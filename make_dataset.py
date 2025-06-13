from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from get_data_from_POKEAPI import get_poke_data_from_json
from json_and_png_check import get_RGB_tensor_by_png

class PokeDataset(Dataset):
    def __init__(self, id_list, poke_data, transform=None, label_type = 'type'):
        """
        カスタムデータセットのコンストラクタ。

        Args:
            id_list (list): データセットに含まれるIDのリスト (train_id_list または test_id_list)。
            data (dict): JSONから読み込んだデータ辞書 (data[id][data_type]でアクセス可能)。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
        self.id_list = id_list
        self.poke_data = poke_data
        self.transform = transform
        self.label_type = label_type

    def __len__(self):
        """
        データセット内のサンプルの総数を返します。
        """
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        指定されたインデックスのサンプルを返します。

        Args:
            idx (int): サンプルのインデックス。

        Returns:
            tuple: (tensor, label) の形式のタプル。
                   tensor: get_tensor_by_id(id) で得られる (3, 96, 96) のテンソル。
                   label: データの種類に応じて適切なラベル。ここではダミーとして0を返します。
                          必要に応じてdata[id][data_type]から取得してください。
        """
        png_path = f'poke_png/poke_{self.id_list[idx]}.png'
        image_rgba = Image.open(png_path).convert('RGBA')
        image_np = np.array(image_rgba)
        alpha_channel = image_np[:, :, 3]
        transparent_pixels = alpha_channel == 0
        image_np[transparent_pixels, :3] = 0
        image = Image.fromarray(image_np).convert('RGB')

        # JSONデータから追加情報を取得 (例: data_type_A の値)
        label = self.poke_data[str(self.id_list[idx])][self.label_type]

        if self.label_type == 'type':
            type_idx_dict = {
                'normal': 0, #（ノーマル）
                'fire'  : 1, #（ほのお） 
                'water' : 2, #（みず）
                'grass' : 3, #（くさ）
                'electric':4,#（でんき）
                'ice'   : 5, #（こおり）
                'fighting':6,#（かくとう）
                'poison': 7, #（どく）
                'ground': 8, #（じめん） 
                'flying': 9, #（ひこう）
                'psychic':10,#（エスパー）
                'bug'   : 11,#（むし） 
                'rock'  :12, #（いわ）
                'ghost' :13, #（ゴースト）
                'dragon':14, #（ドラゴン）
                'steel' :15, #（はがね）
                'dark'  :16, #（あく）
                'fairy' :17, #（フェアリー）
            }
            label_tensor = torch.zeros(len(type_idx_dict))
            for add_label in label:
                label_tensor[type_idx_dict[add_label]] += 1
            # if len(label) != 1:
            #     label_tensor /= len(label)

        # 変換が指定されていれば適用
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
            ])(image)

        # タスクに応じて、必要なら追加のデータを返すように修正
        # 例: return tensor_data, label, current_id
        return image, label_tensor

# --- データセットとデータローダーの作成例 ---

if __name__ == "__main__":
    json_path = 'poke_data.json'
    poke_data = get_poke_data_from_json(json_path)
    id_list = [each_poke_data['id'] for each_poke_data in poke_data.values() if each_poke_data['generation'] != 9]

    # transform = transforms.Compose([
    #     transforms.RandomRotation(degrees=10),  # -10度から+10度の間でランダム回転
    #     transforms.RandomResizedCrop((96, 96), scale=(0.5, 1.0), ratio=(0.75, 1.33)), # 移動したり切り取ったり
    #     transforms.RandomHorizontalFlip(p=0.5), # 50%の確率で水平反転
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # 色調
    #     transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1), # 10%の確率で白黒に変換
    #     transforms.ToTensor()
    # ])

    transform = transforms.Compose([
        transforms.RandomResizedCrop((96, 96), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5), # RandAugmentに含まれないので残す
        transforms.RandAugment(num_ops=2, magnitude=1), # これを導入。ColorJitter, Rotation, Grayscaleは通常削除
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # データセットの作成
    dataset = PokeDataset(id_list=id_list, poke_data=poke_data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 訓練データローダーからバッチを取得して表示
    print("\nデータローダーからバッチを取得:")
    for imgs, labels in dataloader:
        for idx in range(len(imgs)):
            plt.imshow(imgs[idx].permute(1, 2, 0))
            plt.show()
        break