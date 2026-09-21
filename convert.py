# pip install pillow numpy

import io
import numpy as np
from PIL import Image

def extract_hdr_gainmap_to_txt(input_jpg_path, output_txt_path, metadata_lines=None):
    # --- 1. メタデータを厳密に32行に固定（不足分は空行、超過分は切り捨て） ---
    if metadata_lines is None:
        metadata_lines = []
    
    # リストをコピーして加工
    formatted_metadata = [str(item) for item in metadata_lines[:32]]
    while len(formatted_metadata) < 32:
        formatted_metadata.append("none")  # 32行に満たない場合は "none" (または空文字列 "") で埋める

    # --- 2. JPEGバイナリの読み込み ---
    with open(input_jpg_path, 'rb') as f:
        data = f.read()

    # --- 3. JPEGフレームの全検索 ---
    soi_offsets = []
    idx = 0
    while True:
        idx = data.find(b'\xff\xd8', idx)
        if idx == -1:
            break
        soi_offsets.append(idx)
        idx += 2

    # 各分割ブロックを画像として検証
    candidates = []
    for i, offset in enumerate(soi_offsets):
        end_offset = soi_offsets[i+1] if i + 1 < len(soi_offsets) else len(data)
        chunk = data[offset:end_offset]
        try:
            img = Image.open(io.BytesIO(chunk))
            img.verify() # 画像として壊れていないか検証
            # 再度読み込み直し
            img = Image.open(io.BytesIO(chunk))
            candidates.append(img)
        except Exception:
            continue

    # メイン画像（1番目に大きい解像度の画像）を取得
    if not candidates:
        raise ValueError("有効なJPEG画像が検出できませんでした。")

    sdr_img = candidates[0].convert('RGB')
    width, height = sdr_img.size
    aspect_ratio = width / height

    # ゲインマップの特定（アスペクト比がほぼ一致し、メイン画像より小さいか同等の画像）
    gainmap_img = None
    for cand in candidates[1:]:
        c_w, c_h = cand.size
        c_aspect = c_w / c_h
        # アスペクト比の差が5%以内のものをゲインマップと判定
        if abs(c_aspect - aspect_ratio) < 0.05:
            gainmap_img = cand.convert('L')
            print(f"検出: ゲインマップ画像を検出しました ({c_w}x{c_h})")
            break

    # --- 4. サイズのアライメント調整 ---
    rgb_array = np.array(sdr_img, dtype=np.uint8)

    if gainmap_img is not None:
        if gainmap_img.size != (width, height):
            gainmap_img = gainmap_img.resize((width, height), Image.Resampling.BILINEAR)
        gain_array = np.array(gainmap_img, dtype=np.uint8)
    else:
        print("警告: ゲインマップが検出されなかったため、ゲイン00で出力します。")
        gain_array = np.zeros((height, width), dtype=np.uint8)

    # --- 5. テキストファイルへの書き出し ---
    print(f"書き出し開始: {width}x{height} ピクセル...")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        # メタデータ 32行を正確に出力
        for line in formatted_metadata:
            f.write(f"{line}\n")

        # ピクセルデータの書き出し (0xrrggbb hh)
        for y in range(height):
            lines = []
            for x in range(width):
                r, g, b = rgb_array[y, x]
                hh = gain_array[y, x]
                lines.append(f"0x{r:02x}{g:02x}{b:02x} {hh:02x}\n")
            f.writelines(lines)

    print(f"処理完了: {output_txt_path}")

# --- 実行 ---3
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
