# pip install pillow numpy

import sys
import numpy as np
from PIL import Image, ImageOps

def extract_hdr_gainmap_to_txt(input_image_path, output_txt_path, metadata_lines=None):
    # --- 1. メタデータ行の設定 (32行) ---
    if metadata_lines is None:
        metadata_lines = [""] * 32
    elif len(metadata_lines) < 32:
        metadata_lines.extend([""] * (32 - len(metadata_lines)))
    elif len(metadata_lines) > 32:
        metadata_lines = metadata_lines[:32]

    # --- 2. メイン画像 (SDR RGB) の読み込み ---
    try:
        main_img = Image.open(input_image_path)
    except Exception as e:
        print(f"エラー: 画像を開けませんでした - {e}")
        return

    # 基本のRGB色データを取得
    sdr_rgb_img = main_img.convert('RGB')
    width, height = sdr_rgb_img.size

    # --- 3. メタデータからのゲインマップ (HDR輝度データ) の抽出 ---
    gainmap_img = None
    
    # Ultra HDR JPEG や Apple HDR (HEIC/JPEG) のマルチフレーム / MPF / 補助画像の解析
    try:
        # 画像内に複数のフレーム/サブ画像（ゲインマップ）が埋め込まれているかチェック
        if hasattr(main_img, 'n_frames') and main_img.n_frames > 1:
            for frame in range(1, main_img.n_frames):
                main_img.seek(frame)
                # 2番目以降のフレームに輝度データ(Luminance/GainMap)が入っている
                gainmap_candidate = main_img.convert('L')
                if gainmap_candidate.size == (width, height):
                    gainmap_img = gainmap_candidate
                    break
                else:
                    # ゲインマップが縮小サイズで保存されている場合はメインサイズに拡大
                    gainmap_img = gainmap_candidate.resize((width, height), Image.Resampling.BILINEAR)
                    break
    except Exception as e:
        pass

    # ゲインマップが検出されなかった場合 (SDR画像などの場合)
    if gainmap_img is None:
        print("情報: HDRメタデータ（ゲインマップ）が検出されませんでした。hh はすべて 00 に設定します。")
        gainmap_array = np.zeros((height, width), dtype=np.uint8)
    else:
        print("情報: HDRゲインマップ（輝度補正データ）を抽出しました。")
        gainmap_array = np.array(gainmap_img, dtype=np.uint8)

    # SDR RGBデータの配列化
    rgb_array = np.array(sdr_rgb_img, dtype=np.uint8)

    # --- 4. txtファイルへの書き出し ---
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        # メタデータ 32行を出力
        for line in metadata_lines:
            f.write(f"{line}\n")

        # ピクセルごとに [0xrrggbb] [hh] を出力
        for y in range(height):
            lines = []
            for x in range(width):
                r, g, b = rgb_array[y, x]
                hh = gainmap_array[y, x]  # メタデータから直接取得したHDR輝度値(00〜ff)
                
                lines.append(f"0x{r:02x}{g:02x}{b:02x} {hh:02x}\n")
            f.writelines(lines)

    print(f"処理完了: {output_txt_path} (解像度: {width}x{height})")

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
