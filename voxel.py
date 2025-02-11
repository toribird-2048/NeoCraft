import numpy as np
from PIL import Image

def read_vox_file(filepath):
    """
    .voxファイル形式の3Dボクセルデータを読み込む

    Args:
        filepath (str): ファイルパス

    Returns:
        numpy.ndarray: 3次元numpy配列 (shape: (16, 16, 16, 3), dtype: np.uint8)
                       ボクセルデータ
    """
    with open(filepath, 'rb') as f:
        # ヘッダ読み込み
        magic_number = f.read(4)
        if magic_number != b"VOX\0":
            raise ValueError("Invalid magic number. Not a .vox file.")
        version = f.read(1)[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        resolution_x = f.read(1)[0]
        resolution_y = f.read(1)[0]
        resolution_z = f.read(1)[0]

        if resolution_x != 16 or resolution_y != 16 or resolution_z != 16:
            raise ValueError(f"Unsupported resolution: ({resolution_x}, {resolution_y}, {resolution_z}). Only 16x16x16 is supported.")


        # データ読み込み
        data = np.zeros((16, 16, 16, 3), dtype=np.uint8)
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    rgb_bytes = f.read(3)
                    if not rgb_bytes: # EOFチェック
                        raise ValueError("Unexpected end of file.")
                    data[x, y, z] = np.frombuffer(rgb_bytes, dtype=np.uint8)
    return data

def write_vox_file(filepath, data):
    """
    .voxファイル形式で3Dボクセルデータを書き込む

    Args:
        filepath (str): ファイルパス
        data (numpy.ndarray): 3次元numpy配列 (shape: (16, 16, 16, 3), dtype: np.uint8)
                              各要素はRGB値 [R, G, B]
    """
    if data.shape != (16, 16, 16, 3) or data.dtype != np.uint8:
        raise ValueError("Data must be a numpy array with shape (16, 16, 16, 3) and dtype np.uint8")

    with open(filepath, 'wb') as f:
        # ヘッダ書き込み
        f.write(b"VOX\0")  # マジックナンバー
        f.write(bytes([1]))   # バージョン
        f.write(bytes([16]))  # 解像度 X
        f.write(bytes([16]))  # 解像度 Y
        f.write(bytes([16]))  # 解像度 Z

        # データ書き込み (Z -> Y -> X 順)
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    f.write(data[x, y, z].tobytes()) # RGBデータを書き込む

def read_vox_file(filepath):
    """
    .voxファイル形式の3Dボクセルデータを読み込む

    Args:
        filepath (str): ファイルパス

    Returns:
        numpy.ndarray: 3次元numpy配列 (shape: (16, 16, 16, 3), dtype: np.uint8)
                       ボクセルデータ
    """
    with open(filepath, 'rb') as f:
        # ヘッダ読み込み
        magic_number = f.read(4)
        if magic_number != b"VOX\0":
            raise ValueError("Invalid magic number. Not a .vox file.")
        version = f.read(1)[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        resolution_x = f.read(1)[0]
        resolution_y = f.read(1)[0]
        resolution_z = f.read(1)[0]

        if resolution_x != 16 or resolution_y != 16 or resolution_z != 16:
            raise ValueError(f"Unsupported resolution: ({resolution_x}, {resolution_y}, {resolution_z}). Only 16x16x16 is supported.")


        # データ読み込み
        data = np.zeros((16, 16, 16, 3), dtype=np.uint8)
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    rgb_bytes = f.read(3)
                    if not rgb_bytes: # EOFチェック
                        raise ValueError("Unexpected end of file.")
                    data[x, y, z] = np.frombuffer(rgb_bytes, dtype=np.uint8)
    return data

def array_to_image(arr):
    """NumPy配列をPIL Imageオブジェクトに変換"""
    return Image.fromarray(arr, 'RGB')

def vox_to_gif(vox_filepath, gif_filepath):
    """
    .voxファイルをXYスライスしてZ方向をGIFアニメーションとして出力する

    Args:
        vox_filepath (str): 入力 .vox ファイルのパス
        gif_filepath (str): 出力 GIF ファイルのパス
    """
    try:
        data = read_vox_file(vox_filepath)
    except FileNotFoundError:
        print(f"エラー：ファイル '{vox_filepath}' が見つかりません。")
        return
    except ValueError as e:
        print(f"エラー：.voxファイルの読み込みに失敗しました: {e}")
        return

    images = []
    for z in range(16):
        # XY平面でスライス (z軸方向のスライス)
        slice_2d = data[:, :, z, :] # shape: (16, 16, 3)
        image = array_to_image(slice_2d)
        images.append(image)

    if not images:
        print("エラー：有効な画像データが生成されませんでした。")
        return

    try:
        # GIFアニメーションとして保存
        images[0].save(gif_filepath,
                       save_all=True,
                       append_images=images[1:],
                       loop=0,  # 0: 無限ループ, 1: 1回ループ
                       duration=100) # 各フレームの間隔 (ミリ秒)
        print(f"GIFアニメーション '{gif_filepath}' を作成しました。")
    except Exception as e:
        print(f"エラー：GIFファイルの保存に失敗しました: {e}")