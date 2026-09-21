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

    # --- 2. 完全なJPEGフレームの切り出し (FF D8 ... FF D9) ---
    jpeg_frames = []
    cursor = 0
    
    while True:
        soi = data.find(b'\xff\xd8', cursor)
        if soi == -1:
            break
        
        eoi = data.find(b'\xff\xd9', soi)
        if eoi == -1:
            break
        
        # FF D9 (2バイト) を含めて切り出す
        frame_data = data[soi:eoi + 2]
        
        # 10KB以上のフレームのみを対象にする (サムネイルなどの微小フレームを排除)
        if len(frame_data) > 10240:
            jpeg_frames.append(frame_data)
        
        cursor = eoi + 2

    sdr_img = None
    gainmap_img = None

    if len(jpeg_frames) >= 2:
        print(f"検出: 有効なJPEGフレームが {len(jpeg_frames)} 個見つかりました。Ultra HDRデータとして切り出します。")
        # 1番目のフレーム (SDRメイン画像)
        sdr_img = Image.open(io.BytesIO(jpeg_frames[0])).convert('RGB')
        
        # 2番目のフレーム (HDRゲインマップ)
        gainmap_img = Image.open(io.BytesIO(jpeg_frames[1])).convert('L')
    else:
        print("検出: 単一のJPEG画像として読み込みます。")
        sdr_img = Image.open(io.BytesIO(data) if len(jpeg_frames) == 0 else io.BytesIO(jpeg_frames[0])).convert('RGB')

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
