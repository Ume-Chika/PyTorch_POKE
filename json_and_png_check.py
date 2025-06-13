import os
from pprint import pprint
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from get_data_from_POKEAPI import get_poke_data_from_json

def pad_tensor_to_96x96(tensor_original):
    C, H_original, W_original = tensor_original.shape
    H_target, W_target = 96, 96

    # 高さ方向のパディング量
    pad_h_total = H_target - H_original
    pad_h_top = pad_h_total // 2
    pad_h_bottom = pad_h_total - pad_h_top

    # 幅方向のパディング量
    pad_w_total = W_target - W_original
    pad_w_left = pad_w_total // 2
    pad_w_right = pad_w_total - pad_w_left

    # F.pad の引数は (左, 右, 上, 下) の順
    padding_values = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
    
    tensor_padded = F.pad(tensor_original, padding_values, "constant", 0)
    return tensor_padded

def get_RGB_tensor_by_png(png_path, show_result=False):
    try:
        # PILで画像を読み込む
        pil_image = Image.open(png_path).convert('RGBA') # グレースケール画像の場合も'RGB'に変換しておくと後の処理が楽
        transform = transforms.ToTensor()

        # Transformを適用してテンソルに変換
        RGBA_tensor = transform(pil_image)

        # 前処理, AでRGBをフィルタリング
        RGB_tensor = RGBA_tensor[0:3] * RGBA_tensor[3]

        # 可視化（オプション）
        # matplotlibは (H, W, C) と [0.0, 1.0] の範囲を期待するので、
        # テンソルを(H, W, C)に転置し、CPUに移動させてnumpy配列に変換します。
        if show_result:
            for i in range(3):
                plt.imshow(RGB_tensor.permute(1, 2, 0).cpu().numpy()[:,:,i])
                plt.title(f"Loaded PNG as PyTorch Tensor (normalized), color:{i}")
                plt.colorbar()
                plt.axis('off')
                plt.show()
            plt.imshow(RGB_tensor.permute(1, 2, 0).cpu().numpy())
            plt.title(f"Loaded PNG as PyTorch Tensor (normalized), color:all")
            plt.colorbar()
            plt.axis('off')
            plt.show()

        return pad_tensor_to_96x96(RGB_tensor)
    except FileNotFoundError:
        print(f"エラー: ファイル '{png_path}' が見つかりませんでした。パスを確認してください。")
        return None
    except Exception as e:
        print(f"画像の読み込み中にエラーが発生しました: {e}")
        return None


if __name__ == "__main__":
    data_broken_id = [ # 破損 or 学習に使えないデータ
        
    ]

    # for id in torch.randint(1, 1025, size=(4,)):
    for id in range(1, 1025+1):
        if id in data_broken_id:
            continue
        id = int(id)
        json_path = "poke_data.json"
        poke_data = get_poke_data_from_json(json_path)

        print(id, len(poke_data[str(id)]['type']))
        # pprint(poke_data[str(id)])
        """ # 出力例
        1025
        {'generation': 1,
        'id': 25,
        'name': 'ピカチュウ',
        'state': {'attack': 55,
                'defense': 40,
                'hp': 35,
                'special-attack': 50,
                'special-defense': 50,
                'speed': 90},
        'type': ['electric']}
        """

        # PNGファイルのパス

        png_path = f'poke_png/poke_{id}.png' # ここを実際のPNGファイルパスに置き換えてください
        image_tensor = get_RGB_tensor_by_png(png_path, show_result=False)