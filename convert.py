# pip install pillow numpy

import io
import numpy as np
from PIL import Image

def extract_hdr_gainmap_to_txt(input_jpg_path, output_txt_path, metadata_lines=None):
    if metadata_lines is None:
        metadata_lines = [""] * 32

    # --- 1. JPEGバイナリの読み込み ---
    with open(input_jpg_path, 'rb') as f:
        data = f.read()

    # --- 2. JPEGの開始マーカー (FF D8) の位置を検索 ---
    soi_offsets = []
    idx = 0
    while True:
        idx = data.find(b'\xff\xd8', idx)
        if idx == -1:
            break
        soi_offsets.append(idx)
        idx += 2

    # EXIFやAPPセグメントのサムネイル（小さなFF D8）を除外するため、10KB以上の間隔があるSOIのみを採用
    valid_sois = []
    for i, offset in enumerate(soi_offsets):
        # 最後のSOIか、次のSOIとの間隔が10KB以上ある場合を有効なフレーム開始位置とする
        if i == len(soi_offsets) - 1 or (soi_offsets[i+1] - offset) > 10240:
            valid_sois.append(offset)

    sdr_img = None
    gainmap_img = None

    if len(valid_sois) >= 2:
        print(f"検出: Ultra HDR構造（複数フレーム）を検出しました ({len(valid_sois)}個)。")
        
        # 1番目のフレーム (SDRメイン画像): ファイル先頭〜2番目のフレーム直前まで
        sdr_bytes = data[valid_sois[0]:valid_sois[1]]
        sdr_img = Image.open(io.BytesIO(sdr_bytes)).convert('RGB')
        
        # 2番目のフレーム (HDRゲインマップ): 2番目のフレーム〜末尾（または3番目の直前）まで
        gain_end = valid_sois[2] if len(valid_sois) > 2 else len(data)
        gain_bytes = data[valid_sois[1]:gain_end]
        gainmap_img = Image.open(io.BytesIO(gain_bytes)).convert('L')
    else:
        print("検出: 単一のJPEG画像として読み込みます。")
        sdr_img = Image.open(input_jpg_path).convert('RGB')

    # --- 3. サイズの自動調整 ---
    width, height = sdr_img.size
    rgb_array = np.array(sdr_img, dtype=np.uint8)

    if gainmap_img is not None:
        if gainmap_img.size != (width, height):
            gainmap_img = gainmap_img.resize((width, height), Image.Resampling.BILINEAR)
        gain_array = np.array(gainmap_img, dtype=np.uint8)
    else:
        gain_array = np.zeros((height, width), dtype=np.uint8)

    # --- 4. テキストファイルへの書き出し ---
    print(f"書き出し開始: {width}x{height} ピクセル...")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for line in metadata_lines:
            f.write(f"{line}\n")

        for y in range(height):
            lines = []
            for x in range(width):
                r, g, b = rgb_array[y, x]
                hh = gain_array[y, x]
                lines.append(f"0x{r:02x}{g:02x}{b:02x} {hh:02x}\n")
            f.writelines(lines)

    print(f"処理完了: {output_txt_path}")

# --- 実行 ---2
if __name__ == "__main__":
    # Ultra HDR(JPEG) や HDR表示対応の画像をそのまま指定します
    metadata = [
        "0.1",
        "0.25",
        "2160",
        "3840",
        "RGB",
        "none"
    ]
    extract_hdr_gainmap_to_txt("PXL_20260730_050457969.jpg", "testOutput3.txt", metadata)
