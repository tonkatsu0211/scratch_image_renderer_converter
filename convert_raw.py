import sys
import struct
import re
import io

import numpy as np
from PIL import Image


def prepare_metadata(metadata_lines):
    if metadata_lines is None:
        metadata_lines = []

    metadata_lines = list(metadata_lines[:32])
    metadata_lines.extend([""] * (32 - len(metadata_lines)))

    return metadata_lines


def read_jpeg_segments(data):
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

        if marker == 0xda:
            break

        pos += length

    return segments


def extract_xmp_packets(data):
    segments = read_jpeg_segments(data)

    standard_xmp = []
    extended_chunks = {}

    extended_guid = None

    for segment in segments:
        if segment["marker"] != 0xe1:
            continue

        payload = segment["data"]

        standard_header = b"http://ns.adobe.com/xap/1.0/\x00"
        extended_header = b"http://ns.adobe.com/xmp/extension/\x00"

        if payload.startswith(standard_header):
            xmp = payload[len(standard_header):]
            standard_xmp.append(xmp)

        elif payload.startswith(extended_header):
            p = len(extended_header)

            if len(payload) < p + 32 + 4 + 4:
                continue

            guid = payload[p:p + 32].decode(
                "ascii",
                errors="ignore"
            )

            p += 32

            full_length = struct.unpack(
                ">I",
                payload[p:p + 4]
            )[0]

            p += 4

            offset = struct.unpack(
                ">I",
                payload[p:p + 4]
            )[0]

            p += 4

            chunk = payload[p:]

            extended_guid = guid

            if guid not in extended_chunks:
                extended_chunks[guid] = {
                    "length": full_length,
                    "chunks": {}
                }

            extended_chunks[guid]["chunks"][offset] = chunk

    standard_text = b"".join(
        standard_xmp
    ).decode(
        "utf-8",
        errors="ignore"
    )

    extended_text = ""

    for guid, info in extended_chunks.items():
        full_length = info["length"]
        chunks = info["chunks"]

        if not chunks:
            continue

        buffer = bytearray(full_length)

        complete = True

        for offset, chunk in chunks.items():
            end = offset + len(chunk)

            if end > full_length:
                complete = False
                break

            buffer[offset:end] = chunk

        if complete:
            extended_text += bytes(buffer).decode(
                "utf-8",
                errors="ignore"
            )

    return standard_text, extended_text


def extract_xmp(data):
    standard_xmp, extended_xmp = extract_xmp_packets(data)

    if not standard_xmp and not extended_xmp:
        return None

    return standard_xmp + extended_xmp


def get_xmp_value(xmp, name):
    patterns = [
        rf'hdrgm:{re.escape(name)}\s*=\s*"([^"]+)"',
        rf"hdrgm:{re.escape(name)}\s*=\s*'([^']+)'"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            xmp,
            re.S
        )

        if match:
            return match.group(1)

    return None


def parse_float_value(xmp, name, default=None):
    value = get_xmp_value(
        xmp,
        name
    )

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def parse_bool_value(xmp, name, default=False):
    value = get_xmp_value(
        xmp,
        name
    )

    if value is None:
        return default

    return value.lower() == "true"


def get_gainmap_length_from_xmp(xmp):
    patterns = [
        r'<Container:Item\b[^>]*'
        r'Item:Semantic\s*=\s*"GainMap"'
        r'[^>]*Item:Length\s*=\s*"(\d+)"',

        r'<Container:Item\b[^>]*'
        r'Item:Length\s*=\s*"(\d+)"'
        r'[^>]*Item:Semantic\s*=\s*"GainMap"'
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            xmp,
            re.S
        )

        if match:
            return int(match.group(1))

    return None


def find_jpeg_end(data, start):
    if start < 0 or start + 2 > len(data):
        return None

    if data[start:start + 2] != b"\xff\xd8":
        return None

    pos = start + 2

    while pos < len(data):
        marker_start = data.find(
            b"\xff",
            pos
        )

        if marker_start < 0:
            return None

        p = marker_start + 1

        while p < len(data) and data[p] == 0xff:
            p += 1

        if p >= len(data):
            return None

        marker = data[p]
        p += 1

        if marker == 0xd9:
            return p

        if marker == 0xda:
            if p + 2 > len(data):
                return None

            length = struct.unpack(
                ">H",
                data[p:p + 2]
            )[0]

            if length < 2 or p + length > len(data):
                return None

            pos = p + length

            while pos < len(data):
                ff = data.find(
                    b"\xff",
                    pos
                )

                if ff < 0:
                    return None

                q = ff + 1

                if q >= len(data):
                    return None

                if data[q] == 0x00:
                    pos = q + 1
                    continue

                if data[q] == 0xff:
                    pos = q
                    continue

                marker = data[q]

                if 0xd0 <= marker <= 0xd7:
                    pos = q + 1
                    continue

                if marker == 0xd9:
                    return q + 1

                return None

        elif marker in [0xd8] or 0xd0 <= marker <= 0xd7:
            pos = p

        else:
            if p + 2 > len(data):
                return None

            length = struct.unpack(
                ">H",
                data[p:p + 2]
            )[0]

            if length < 2 or p + length > len(data):
                return None

            pos = p + length

    return None


def find_gainmap_jpeg(data, gainmap_length):
    if gainmap_length is None:
        return None

    first_end = find_jpeg_end(
        data,
        0
    )

    if first_end is None:
        return None

    search_pos = first_end

    while True:
        start = data.find(
            b"\xff\xd8",
            search_pos
        )

        if start < 0:
            break

        end = start + gainmap_length

        if end <= len(data):
            candidate = data[start:end]

            if (
                len(candidate) == gainmap_length
                and candidate.startswith(b"\xff\xd8")
                and candidate.endswith(b"\xff\xd9")
            ):
                try:
                    with Image.open(
                        io.BytesIO(candidate)
                    ) as image:
                        image.verify()

                    return candidate

                except Exception:
                    pass

        actual_end = find_jpeg_end(
            data,
            start
        )

        if actual_end is None:
            search_pos = start + 2
        else:
            search_pos = actual_end

    return None


def get_gainmap_image(gainmap_jpeg):
    image = Image.open(
        io.BytesIO(gainmap_jpeg)
    )

    image.load()

    print(
        f"Gain Map解像度: "
        f"{image.width}x{image.height}"
    )

    print(
        f"Gain Map mode: "
        f"{image.mode}"
    )

    if image.mode == "L":
        return image

    if image.mode in (
        "RGB",
        "RGBA"
    ):
        return image.convert("L")

    return image.convert("L")


def resize_gainmap_to_primary(
    gainmap_image,
    width,
    height
):
    if gainmap_image.size == (
        width,
        height
    ):
        return gainmap_image

    return gainmap_image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )


def load_ultra_hdr(input_image_path):
    with open(
        input_image_path,
        "rb"
    ) as f:
        jpeg_data = f.read()

    main_img = Image.open(
        io.BytesIO(jpeg_data)
    ).convert("RGB")

    width, height = main_img.size

    primary_xmp = extract_xmp(
        jpeg_data
    )

    if primary_xmp is None:
        print(
            "情報: 主画像XMPが見つかりません。"
        )

        return (
            main_img,
            np.zeros(
                (height, width),
                dtype=np.uint8
            )
        )

    version = get_xmp_value(
        primary_xmp,
        "Version"
    )

    if version != "1.0":
        print(
            f"情報: Ultra HDR Gain Map Version "
            f"1.0ではありません: {version}"
        )

        return (
            main_img,
            np.zeros(
                (height, width),
                dtype=np.uint8
            )
        )

    print(
        f"Ultra HDR Gain Map Version: {version}"
    )

    gainmap_length = get_gainmap_length_from_xmp(
        primary_xmp
    )

    if gainmap_length is None:
        print(
            "警告: GContainerからGain MapのLengthを取得できません。"
        )

        return (
            main_img,
            np.zeros(
                (height, width),
                dtype=np.uint8
            )
        )

    print(
        f"Gain Map JPEG Length: "
        f"{gainmap_length} bytes"
    )

    gainmap_jpeg = find_gainmap_jpeg(
        jpeg_data,
        gainmap_length
    )

    if gainmap_jpeg is None:
        print(
            "警告: GContainerで指定されたGain Map JPEGを取得できません。"
        )

        return (
            main_img,
            np.zeros(
                (height, width),
                dtype=np.uint8
            )
        )

    print(
        "Gain Map JPEGを取得しました。"
    )

    standard_xmp, extended_xmp = extract_xmp_packets(
        jpeg_data
    )

    gain_map_min = parse_float_value(
        standard_xmp + extended_xmp,
        "GainMapMin"
    )

    gain_map_max = parse_float_value(
        standard_xmp + extended_xmp,
        "GainMapMax"
    )

    gamma = parse_float_value(
        standard_xmp + extended_xmp,
        "Gamma"
    )

    offset_sdr = parse_float_value(
        standard_xmp + extended_xmp,
        "OffsetSDR"
    )

    offset_hdr = parse_float_value(
        standard_xmp + extended_xmp,
        "OffsetHDR"
    )

    hdr_capacity_min = parse_float_value(
        standard_xmp + extended_xmp,
        "HDRCapacityMin"
    )

    hdr_capacity_max = parse_float_value(
        standard_xmp + extended_xmp,
        "HDRCapacityMax"
    )

    base_rendition_is_hdr = parse_bool_value(
        standard_xmp + extended_xmp,
        "BaseRenditionIsHDR",
        False
    )

    print(
        f"GainMapMin: {gain_map_min}"
    )

    print(
        f"GainMapMax: {gain_map_max}"
    )

    print(
        f"Gamma: {gamma}"
    )

    print(
        f"OffsetSDR: {offset_sdr}"
    )

    print(
        f"OffsetHDR: {offset_hdr}"
    )

    print(
        f"HDRCapacityMin: {hdr_capacity_min}"
    )

    print(
        f"HDRCapacityMax: {hdr_capacity_max}"
    )

    print(
        f"BaseRenditionIsHDR: "
        f"{base_rendition_is_hdr}"
    )

    gainmap_img = get_gainmap_image(gainmap_jpeg)
    gainmap_img = resize_gainmap_to_primary(gainmap_img, width, height)
    gainmap_array = np.array(gainmap_img, dtype=np.float32)

    if gain_map_min is None:
        gain_map_min = 0.0
    if gain_map_max is None:
        gain_map_max = 1.0
    if gamma is None or gamma <= 0:
        gamma = 1.0
    if hdr_capacity_min is None:
        hdr_capacity_min = 0.0
    if hdr_capacity_max is None:
        hdr_capacity_max = gain_map_max

    max_display_boost = 4.0

    if hdr_capacity_max > hdr_capacity_min:
        unclamped_weight_factor = (
            np.log2(max_display_boost) - hdr_capacity_min
        ) / (hdr_capacity_max - hdr_capacity_min)
        weight_factor = np.clip(unclamped_weight_factor, 0.0, 1.0)
    else:
        weight_factor = 1.0

    recovery = gainmap_array / 255.0
    log_recovery = np.power(recovery, 1.0 / gamma)

    log_boost = (
        gain_map_min * (1.0 - log_recovery)
        + gain_map_max * log_recovery
    )

    gain = np.exp2(log_boost * weight_factor)

    max_gain = 4.0
    gain = np.clip(gain, 1.0, max_gain)

    hh_normalized = np.log2(gain) / np.log2(max_gain)
    hdr_array = np.clip(
        np.round(hh_normalized * 255.0),
        0.0,
        255.0
    ).astype(np.uint8)

    print(f"Gain Map weight factor: {weight_factor:.6f}")
    print(f"HDR倍率範囲: {gain.min():.4f}x ～ {gain.max():.4f}x")

    return main_img, hdr_array


def extract_hdr_gainmap_to_txt(
    input_image_path,
    output_txt_path,
    metadata_lines=None
):
    metadata_lines = prepare_metadata(
        metadata_lines
    )

    main_img, hdr_array = load_ultra_hdr(
        input_image_path
    )

    rgb_array = np.array(
        main_img,
        dtype=np.uint8
    )

    height, width = rgb_array.shape[:2]

    with open(
        output_txt_path,
        "w",
        encoding="utf-8",
        newline="\n"
    ) as f:

        for line in metadata_lines:
            f.write(
                line + "\n"
            )

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


def debug_ultra_hdr_structure(
    input_image_path
):
    with open(
        input_image_path,
        "rb"
    ) as f:
        data = f.read()

    print("================================")
    print("JPEG構造")
    print("================================")

    segments = read_jpeg_segments(
        data
    )

    for i, segment in enumerate(segments):
        marker = segment["marker"]
        payload = segment["data"]

        if 0xe0 <= marker <= 0xef:
            name = f"APP{marker - 0xe0}"
        else:
            name = "APP-"

        print(
            f"{i:02d}: "
            f"{name} "
            f"marker=0x{marker:02x} "
            f"size={len(payload)}"
        )

        if marker == 0xe1:
            standard_header = (
                b"http://ns.adobe.com/xap/1.0/\x00"
            )

            extended_header = (
                b"http://ns.adobe.com/xmp/extension/\x00"
            )

            if payload.startswith(
                standard_header
            ):
                print(
                    "    XMP detected"
                )

                xmp = payload[
                    len(standard_header):
                ]

                xmp_text = xmp.decode(
                    "utf-8",
                    errors="ignore"
                )

                for key in [
                    "hdrgm:",
                    "GainMap",
                    "Container:",
                    "Item:",
                    "Directory",
                    "HasExtendedXMP"
                ]:
                    if key in xmp_text:
                        print(
                            f"    contains: {key}"
                        )

                print(
                    xmp_text[:5000]
                )

            elif payload.startswith(
                extended_header
            ):
                print(
                    "    Extended XMP detected"
                )

        elif marker == 0xe2:
            if payload.startswith(
                b"MPF\x00"
            ):
                print(
                    "    MPF detected"
                )

    standard_xmp, extended_xmp = extract_xmp_packets(
        data
    )

    print("================================")
    print("XMP解析結果")
    print("================================")

    print(
        f"Standard XMP: "
        f"{len(standard_xmp)} bytes"
    )

    print(
        f"Extended XMP: "
        f"{len(extended_xmp)} bytes"
    )

    gainmap_length = get_gainmap_length_from_xmp(
        standard_xmp + extended_xmp
    )

    print(
        f"Gain Map Length: "
        f"{gainmap_length}"
    )

    print(
        f"GainMapMin: "
        f"{parse_float_value(standard_xmp + extended_xmp, 'GainMapMin')}"
    )

    print(
        f"GainMapMax: "
        f"{parse_float_value(standard_xmp + extended_xmp, 'GainMapMax')}"
    )

    print(
        f"Gamma: "
        f"{parse_float_value(standard_xmp + extended_xmp, 'Gamma')}"
    )

    print(
        f"OffsetSDR: "
        f"{parse_float_value(standard_xmp + extended_xmp, 'OffsetSDR')}"
    )

    print(
        f"OffsetHDR: "
        f"{parse_float_value(standard_xmp + extended_xmp, 'OffsetHDR')}"
    )

    print(
        f"HDRCapacityMin: "
        f"{parse_float_value(standard_xmp + extended_xmp, 'HDRCapacityMin')}"
    )

    print(
        f"HDRCapacityMax: "
        f"{parse_float_value(standard_xmp + extended_xmp, 'HDRCapacityMax')}"
    )

    print("================================")


if __name__ == "__main__":
    metadata = [
        "0.1",
        "0.25",
        "2160",
        "3840",
        "RGB",
        "none"
    ]

    input_path = (
        "/kaggle/input/datasets/tonkatsu0211/"
        "testfile3/PXL_20260730_050457969.jpg"
    )

    output_path = (
        "/kaggle/working/testOutput5.txt"
    )

    debug_ultra_hdr_structure(
        input_path
    )

    extract_hdr_gainmap_to_txt(
        input_path,
        output_path,
        metadata
    )
