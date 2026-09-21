# pip install pillow numpy

import struct
import numpy as np
from PIL import Image
import io

def extract_ultra_hdr_jpeg(input_jpg_path, output_txt_path, metadata_lines=None):
    if metadata_lines is None:
        metadata_lines = [""] * 32

    # --- 1. JPEGバイナリの読み込み ---
    with open(input_jpg_path, 'rb') as f:
        data = f.read()

    # --- 2. MPF (Multi-Picture Format) または 埋め込みJPEGの抽出 ---
    # Ultra HDR JPEGは、主画像(SDR)の後にゲインマップ(JPEG)が結合されています (FF D8 ... FF D9)
    jpeg_markers = []
    idx = 0
    while True:
        idx = data.find(b'\xff\xd8', idx)
        if idx == -1:
            break
        jpeg_markers.append(idx)
        idx += 2

    sdr_img = None
    gainmap_img = None

    if len(jpeg_markers) >= 2:
        print(f"検出: 複数のJPEGフレームが見つかりました ({len(jpeg_markers)}個)。Ultra HDRデータとして解析します。")
        # 1番目のJPEG (SDRメイン画像)
        sdr_bytes = data[jpeg_markers[0]:jpeg_markers[1]]
        sdr_img = Image.open(io.BytesIO(sdr_bytes)).convert('RGB')
        
        # 2番目のJPEG (HDRゲインマップ)
        gainmap_bytes = data[jpeg_markers[1]:]
        gainmap_img = Image.open(io.BytesIO(gainmap_bytes)).convert('L') # グレースケールで読み込み
    else:
        print("警告: 埋め込みゲインマップが見つかりませんでした。通常のSDR画像として処理します。")
        sdr_img = Image.open(input_jpg_path).convert('RGB')

    # --- 3. サイズの整合処理 ---
    width, height = sdr_img.size
    rgb_array = np.array(sdr_img, dtype=np.uint8)

    if gainmap_img is not None:
        # ゲインマップ画像はメイン画像より解像度が小さい場合があるため、リサイズして合わせる
        if gainmap_img.size != (width, height):
            gainmap_img = gainmap_img.resize((width, height), Image.Resampling.BILINEAR)
        gain_array = np.array(gainmap_img, dtype=np.uint8)
    else:
        # ゲインマップが無い場合は 00
        gain_array = np.zeros((height, width), dtype=np.uint8)

    # --- 4. テキストファイルへの書き出し (0xrrggbb hh 形式) ---
    print(f"書き出し開始: {width}x{height} ピクセル...")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        # メタデータ 32行を出力
        for line in metadata_lines:
            f.write(f"{line}\n")

        # ピクセルデータの高速出力 (1行ずつまとめて書き込み)
        for y in range(height):
            lines = []
            for x in range(width):
                r, g, b = rgb_array[y, x]
                hh = gain_array[y, x]
                lines.append(f"0x{r:02x}{g:02x}{b:02x} {hh:02x}\n")
            f.writelines(lines)

    print(f"処理完了: {output_txt_path}")

# --- 実行 ---
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
