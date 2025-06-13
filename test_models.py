import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from get_data_from_POKEAPI import get_poke_data_from_json
from make_dataset import PokeDataset
from utils import type_to_idx, idx_to_type

def check_id_usable(id_list):
    new_id_list = []
    black_list = [
        678,
    ]
    for id in id_list:
        if os.path.exists(f'poke_png/poke_{id}.png') and not id in black_list:
            new_id_list.append(id)
    return new_id_list

if __name__ == "__main__":
    json_path = 'poke_data.json'
    poke_data = get_poke_data_from_json(json_path)
    """
        学習データ
            第1〜8世代のポケモン（905件）
        テストデータ
            第9世代のポケモン（120件）
    """
    test_id_list  = check_id_usable([each_poke_data['id'] for each_poke_data in poke_data.values() if each_poke_data['generation'] == 9])

    # テスト用データセットの作成
    test_dataset = PokeDataset(id_list=test_id_list, poke_data=poke_data)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

    model_select = 'warp'
    if model_select == 'class_balanced':
        from class_balanced_type_predictor import TypePredictor, setting_TypePredictor
    elif model_select == 'warp':
        from warp_type_predictor import TypePredictor, setting_TypePredictor
    model = TypePredictor(**setting_TypePredictor)
    model.load_state_dict(torch.load('model_states/class_balanced.pth'))

    collect = 0
    data_num = 0
    for img, label in test_dataloader:
        output = model(img)
        predicted_probs = torch.sigmoid(output)
        threshold = 0.5
        predicted_labels = (predicted_probs > threshold).float()
        for idx in range(len(img)):
            plt.imshow(img[idx].permute(1, 2, 0))
            plt.show()
            plt.bar([idx_to_type[i] for i in range(18)], predicted_probs[idx].detach().numpy())
            plt.xticks(rotation=90)
            plt.show()
        break

    # 予測結果の集計
    all_predicted_labels = []
    all_true_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs
            labels = labels
            outputs = model(imgs)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).float() # スレッショルド0.5で二値化

            all_predicted_labels.append(predicted_labels.cpu())
            all_true_labels.append(labels.cpu())

        all_predicted_labels = torch.cat(all_predicted_labels, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)

        # 精度 (Accuracy) の計算例: 各サンプルの全タイプを正しく予測できたか
        correct_samples = (all_predicted_labels == all_true_labels).all(dim=1).sum().item()
        total_samples = all_true_labels.size(0)
        overall_accuracy = correct_samples / total_samples
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

        # 各タイプごとのF1スコア計算 (scikit-learnを使用)
        from sklearn.metrics import f1_score
        f1_micro = f1_score(all_true_labels, all_predicted_labels, average='micro')
        f1_macro = f1_score(all_true_labels, all_predicted_labels, average='macro')
        print(f"Micro F1-score: {f1_micro:.4f}")
        print(f"Macro F1-score: {f1_macro:.4f}")