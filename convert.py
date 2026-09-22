# pip install pillow numpy

import sys
import struct
import re
import math
import io

import numpy as np
from PIL import Image


# ============================================================
# メタデータ
# ============================================================

def prepare_metadata(metadata_lines):
    if metadata_lines is None:
        metadata_lines = []

    metadata_lines = list(metadata_lines[:32])
    metadata_lines.extend([""] * (32 - len(metadata_lines)))

    return metadata_lines


# ============================================================
# JPEGセグメント解析
# ============================================================

def read_jpeg_segments(data):
    """
    JPEGのAPP1/APP2などのセグメントを取得する。
    """

    segments = []

    if not data.startswith(b"\xff\xd8"):
        raise ValueError("JPEGではありません")

    pos = 2

    while pos < len(data):
        if data[pos] != 0xff:
            pos += 1
            continue

        while pos < len(data) and data[pos] == 0xff:
            pos += 1

        if pos >= len(data):
            break

        marker = data[pos]
        pos += 1

        # SOI / EOI / RST
        if marker in [0xd8, 0xd9] or 0xd0 <= marker <= 0xd7:
            continue

        if pos + 2 > len(data):
            break

        length = struct.unpack(">H", data[pos:pos + 2])[0]

        if length < 2 or pos + length > len(data):
            break

        segment_data = data[pos + 2:pos + length]

        segments.append({
            "marker": marker,
            "data": segment_data,
            "start": pos - 2,
            "end": pos + length
        })

        # SOS以降は通常のJPEG画像データなのでここでは終了
        if marker == 0xda:
            break

        pos += length

    return segments


# ============================================================
# XMP
# ============================================================

def extract_xmp(data):
    """
    JPEG APP1からXMPを取り出す。
    """

    segments = read_jpeg_segments(data)

    xmp_chunks = []

    for segment in segments:
        if segment["marker"] != 0xe1:
            continue

        payload = segment["data"]

        if payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
            xmp = payload[len(b"http://ns.adobe.com/xap/1.0/\x00"):]
            xmp_chunks.append(xmp)

    if not xmp_chunks:
        return None

    return b"".join(xmp_chunks).decode("utf-8", errors="ignore")


def get_xmp_value(xmp, name):
    """
    hdrgm:GainMapMin などのXMP属性を取得。
    """

    patterns = [
        rf'hdrgm:{re.escape(name)}\s*=\s*"([^"]+)"',
        rf'hdrgm:{re.escape(name)}\s*=\s*\'([^\']+)\'',
    ]

    for pattern in patterns:
        match = re.search(pattern, xmp)

        if match:
            return match.group(1)

    return None


def parse_float_value(xmp, name, default=None):
    value = get_xmp_value(xmp, name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def parse_bool_value(xmp, name, default=False):
    value = get_xmp_value(xmp, name)

    if value is None:
        return default

    return value.lower() == "true"


# ============================================================
# MPF
# ============================================================

def parse_mpf(data):
    """
    MPF(APP2)から画像エントリを取得する。

    Ultra HDR JPEGでは、MPFの2枚目のJPEGがGain Map。
    """

    segments = read_jpeg_segments(data)

    mpf_data = None
    mpf_segment_start = None

    for segment in segments:
        if segment["marker"] == 0xe2:
            payload = segment["data"]

            if payload.startswith(b"MPF\x00"):
                mpf_data = payload
                mpf_segment_start = segment["start"]
                break

    if mpf_data is None:
        return None

    # MPF:
    # 0～3   : "MPF\0"
    # 4～    : TIFFヘッダ
    tiff = mpf_data[4:]

    endian = tiff[:2]

    if endian == b"II":
        byte_order = "<"
    elif endian == b"MM":
        byte_order = ">"
    else:
        raise ValueError("MPF TIFFのバイトオーダーが不正です")

    if struct.unpack(
        byte_order + "H",
        tiff[2:4]
    )[0] != 42:
        raise ValueError("MPF TIFFヘッダが不正です")

    ifd_offset = struct.unpack(
        byte_order + "I",
        tiff[4:8]
    )[0]

    ifd_pos = ifd_offset

    entry_count = struct.unpack(
        byte_order + "H",
        tiff[ifd_pos:ifd_pos + 2]
    )[0]

    mp_entry_offset = None
    mp_entry_count = None

    pos = ifd_pos + 2

    for _ in range(entry_count):
        tag, typ, count = struct.unpack(
            byte_order + "HHI",
            tiff[pos:pos + 8]
        )

        value = tiff[pos + 8:pos + 12]

        # MPEntry = 0xB002
        if tag == 0xB002:
            mp_entry_count = count

            if typ == 7:
                mp_entry_offset = struct.unpack(
                    byte_order + "I",
                    value
                )[0]

        pos += 12

    if mp_entry_offset is None:
        return None

    # MP Entry 1つ = 16 bytes
    entry_size = 16

    entries = []

    for i in range(mp_entry_count // entry_size):
        p = mp_entry_offset + i * entry_size

        if p + entry_size > len(tiff):
            break

        attributes = struct.unpack(
            byte_order + "I",
            tiff[p:p + 4]
        )[0]

        size = struct.unpack(
            byte_order + "I",
            tiff[p + 4:p + 8]
        )[0]

        offset = struct.unpack(
            byte_order + "I",
            tiff[p + 8:p + 12]
        )[0]

        entries.append({
            "attributes": attributes,
            "size": size,
            "offset": offset
        })

    if len(entries) < 2:
        return None

    # libultrahdrのMPF生成方式では、
    # 2枚目の画像offsetはMPF APP2セグメント終端付近を
    # 基準として扱う。
    #
    # ただし実装によって基準位置が異なる可能性があるため、
    # 複数候補を後で検証する。
    return {
        "entries": entries,
        "mpf_segment_start": mpf_segment_start,
        "mpf_data": mpf_data
    }


# ============================================================
# Gain Map JPEGの抽出
# ============================================================

def find_gainmap_jpeg(data, mpf):
    """
    MPFからGain Map JPEGを取り出す。

    Ultra HDRでは2番目のMP EntryがGain Map。
    """

    entry = mpf["entries"][1]

    size = entry["size"]
    offset = entry["offset"]

    candidates = []

    # MPF offsetの解釈候補
    candidates.append(offset)
    candidates.append(mpf["mpf_segment_start"] + offset)
    candidates.append(mpf["mpf_segment_start"] + 2 + offset)

    # 実際のJPEG SOI(FFD8)を探す
    for start in candidates:

        if start < 0 or start >= len(data):
            continue

        end = start + size

        if end > len(data):
            continue

        candidate = data[start:end]

        if candidate.startswith(b"\xff\xd8"):
            try:
                Image.open(io.BytesIO(candidate)).verify()
                return candidate
            except Exception:
                pass

    # MPFオフセットを使えない場合の最終フォールバック
    # JPEGの2個目のSOIを探す。
    first_soi = data.find(b"\xff\xd8")

    if first_soi >= 0:
        second_soi = data.find(b"\xff\xd8", first_soi + 2)

        if second_soi >= 0:
            candidate = data[second_soi:second_soi + size]

            try:
                Image.open(io.BytesIO(candidate)).verify()
                return candidate
            except Exception:
                pass

    return None


# ============================================================
# Gain MapのHDR値計算
# ============================================================

def calculate_hdr_gain_map(
    gainmap_array,
    gain_map_min,
    gain_map_max,
    gamma,
    offset_sdr,
    offset_hdr,
    hdr_capacity_min,
    hdr_capacity_max,
    max_display_boost=1.0
):
    """
    Ultra HDR仕様に従ってGain MapからHDRブースト量を計算する。

    gainmap_array:
        0～255のGain Map画素

    戻り値:
        0～255に正規化したhh
    """

    recovery = gainmap_array.astype(np.float32) / 255.0

    # Ultra HDR仕様:
    #
    # log_recovery = pow(recovery, 1 / gamma)
    #
    log_recovery = np.power(
        recovery,
        1.0 / gamma
    )

    log_boost = (
        gain_map_min * (1.0 - log_recovery)
        + gain_map_max * log_recovery
    )

    # max_display_boost=1.0の場合、
    # SDR表示ではweight_factor=0になり、
    # HDRブーストは適用されない。
    #
    # 今回は「Gain Mapが持つ最大HDR補正量」をhhにするため、
    # weight_factorは1.0に固定する。
    weight_factor = 1.0

    effective_log_boost = log_boost * weight_factor

    gain_factor = np.exp2(effective_log_boost)

    # GainMapMin～GainMapMaxを
    # 実際の倍率に戻した範囲。
    min_gain = 2.0 ** gain_map_min
    max_gain = 2.0 ** gain_map_max

    if max_gain == min_gain:
        hh = np.zeros_like(gain_factor, dtype=np.uint8)
    else:
        hh_float = (
            (gain_factor - min_gain)
            / (max_gain - min_gain)
            * 255.0
        )

        hh = np.clip(
            np.round(hh_float),
            0,
            255
        ).astype(np.uint8)

    return hh


# ============================================================
# Ultra HDR JPEG読み込み
# ============================================================

def load_ultra_hdr(input_image_path):
    with open(input_image_path, "rb") as f:
        jpeg_data = f.read()

    # SDR主画像
    main_img = Image.open(io.BytesIO(jpeg_data)).convert("RGB")

    width, height = main_img.size

    # XMP
    xmp = extract_xmp(jpeg_data)

    if xmp is None:
        print("情報: XMPが見つかりません。HDRデータなしとして処理します。")
        return main_img, np.zeros(
            (height, width),
            dtype=np.uint8
        )

    version = get_xmp_value(xmp, "Version")

    if version is None:
        print("情報: hdrgm:Versionが見つかりません。HDRデータなしとして処理します。")
        return main_img, np.zeros(
            (height, width),
            dtype=np.uint8
        )

    print(f"Ultra HDR Gain Map Version: {version}")

    # 必須HDRメタデータ
    gain_map_min = parse_float_value(
        xmp,
        "GainMapMin"
    )

    gain_map_max = parse_float_value(
        xmp,
        "GainMapMax"
    )

    gamma = parse_float_value(
        xmp,
        "Gamma",
        1.0
    )

    offset_sdr = parse_float_value(
        xmp,
        "OffsetSDR",
        0.0
    )

    offset_hdr = parse_float_value(
        xmp,
        "OffsetHDR",
        0.0
    )

    hdr_capacity_min = parse_float_value(
        xmp,
        "HDRCapacityMin",
        gain_map_min
    )

    hdr_capacity_max = parse_float_value(
        xmp,
        "HDRCapacityMax",
        gain_map_max
    )

    base_rendition_is_hdr = parse_bool_value(
        xmp,
        "BaseRenditionIsHDR",
        False
    )

    required = [
        gain_map_min,
        gain_map_max,
        gamma,
        offset_sdr,
        offset_hdr,
        hdr_capacity_min,
        hdr_capacity_max
    ]

    if any(value is None for value in required):
        print("警告: Gain Mapの必要なXMP情報が不足しています。")
        return main_img, np.zeros(
            (height, width),
            dtype=np.uint8
        )

    print(f"GainMapMin: {gain_map_min}")
    print(f"GainMapMax: {gain_map_max}")
    print(f"Gamma: {gamma}")
    print(f"OffsetSDR: {offset_sdr}")
    print(f"OffsetHDR: {offset_hdr}")
    print(f"HDRCapacityMin: {hdr_capacity_min}")
    print(f"HDRCapacityMax: {hdr_capacity_max}")

    # MPF
    mpf = parse_mpf(jpeg_data)

    if mpf is None:
        print("警告: MPFが見つかりません。Gain Mapを取得できません。")

        return main_img, np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # Gain Map JPEG
    gainmap_jpeg = find_gainmap_jpeg(
        jpeg_data,
        mpf
    )

    if gainmap_jpeg is None:
        print("警告: Gain Map JPEGを取得できません。")

        return main_img, np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # Gain Map画像
    gainmap_img = Image.open(
        io.BytesIO(gainmap_jpeg)
    ).convert("L")

    print(
        f"Gain Map解像度: "
        f"{gainmap_img.width}x{gainmap_img.height}"
    )

    # Ultra HDR仕様ではGain Mapの解像度が
    # 主画像と異なっていてもよい。
    #
    # その場合はバイリニア以上でサンプリングする。
    gainmap_img = gainmap_img.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    gainmap_array = np.array(
        gainmap_img,
        dtype=np.uint8
    )

    # Ultra HDR仕様に従ってHDR補正量を算出
    hh_array = calculate_hdr_gain_map(
        gainmap_array,
        gain_map_min,
        gain_map_max,
        gamma,
        offset_sdr,
        offset_hdr,
        hdr_capacity_min,
        hdr_capacity_max
    )

    return main_img, hh_array


# ============================================================
# TXT変換
# ============================================================

def extract_hdr_gainmap_to_txt(
    input_image_path,
    output_txt_path,
    metadata_lines=None
):

    metadata_lines = prepare_metadata(
        metadata_lines
    )

    # --------------------------------------------------------
    # Ultra HDR読み込み
    # --------------------------------------------------------

    main_img, hdr_array = load_ultra_hdr(
        input_image_path
    )

    rgb_array = np.array(
        main_img,
        dtype=np.uint8
    )

    height, width = rgb_array.shape[:2]

    # --------------------------------------------------------
    # TXT出力
    # --------------------------------------------------------

    with open(
        output_txt_path,
        "w",
        encoding="utf-8",
        newline="\n"
    ) as f:

        # 32行のメタデータ
        for line in metadata_lines:
            f.write(line + "\n")

        # 画素データ
        for y in range(height):

            lines = []

            for x in range(width):

                r, g, b = rgb_array[y, x]

                hh = hdr_array[y, x]

                lines.append(
                    f"0x{r:02x}{g:02x}{b:02x} {hh:02x}\n"
                )

            f.writelines(lines)

    print(
        f"処理完了: {output_txt_path} "
        f"(解像度: {width}x{height})"
    )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":

    metadata = [
        "0.1",
        "0.25",
        "2160",
        "3840",
        "RGB",
        "none"
    ]

    extract_hdr_gainmap_to_txt(
        "PXL_20260730_050457969.jpg",
        "testOutput3.txt",
        metadata
    )
