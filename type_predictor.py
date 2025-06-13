import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt

from get_data_from_POKEAPI import get_poke_data_from_json
from make_dataset import PokeDataset

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
    train_id_list = check_id_usable([each_poke_data['id'] for each_poke_data in poke_data.values() if each_poke_data['generation'] != 9])
    test_id_list  = check_id_usable([each_poke_data['id'] for each_poke_data in poke_data.values() if each_poke_data['generation'] == 9])
    # 訓練用データセットの作成
    train_dataset = PokeDataset(id_list=train_id_list, poke_data=poke_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # テスト用データセットの作成
    test_dataset = PokeDataset(id_list=test_id_list, poke_data=poke_data)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    conv_model = nn.Sequential(
        # 入力チャンネル3 (R,G,B), 出力チャンネル16, カーネルサイズ3, パディング1
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        # カーネルサイズ2, ストライド2
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    model = nn.Sequential( # (畳み込み+ReLU+max pooling)を2〜3回追加して
        nn.Linear(16*48*48, 100),
        nn.Tanh(),
        nn.Linear(100, 18)
    )
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss() # 二値交差エントロピー
    n_epochs = 100

    i = 0
    train_loss_log = []
    test_loss_log_dict = {}
    accuracy_log_dict = {}
    for epoch in range(n_epochs):
        if i == 0 or epoch == n_epochs-1:
            # ====== テストフェーズ ======
            model.eval()  # モデルを評価モードに設定
            test_loss = 0.0
            correct = 0
            total = 0

            # 勾配計算を無効にしてテストを実行
            with torch.no_grad(): # 勾配計算を無効にしてテストを実行
                for imgs, labels in test_dataloader:
                    # imgsを畳み込み & ReLU & max pooling
                    imgs = conv_model(imgs)
                    batch_size = imgs.shape[0]
                    outputs = model(imgs.view(batch_size, -1))
                    loss = loss_fn(outputs, labels)
                    test_loss += loss.item()

                    # --- 正解率(Accuracy)の計算 ---
                    # ロジットにシグモイド関数を適用して確率に変換
                    probs = torch.sigmoid(outputs)
                    # 確率が0.5より大きいかを判定し、0か1の予測値を取得
                    predicted = (probs > 0.1).float()

                    total += labels.size(0)
                    correct += torch.all(predicted == labels, dim=1).sum().item()
            # テスト結果を表示
            avg_loss = test_loss / len(test_dataloader)
            accuracy = 100 * correct / total
            print(f'   === Epoch {epoch} Test Results ===')
            print(f'        ・Accuracy: {accuracy:.2f} %')
            test_loss_log_dict[epoch] = float(avg_loss)
            accuracy_log_dict[epoch] = float(accuracy)
        i = (i + 1) % 10

        # ===== 学習フェーズ =====
        model.train()
        for imgs, labels in train_dataloader:
            imgs = conv_model(imgs)
            batch_size = imgs.shape[0]
            outputs = model(imgs.view(batch_size, -1))
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss_log.append(float(loss))
        print(f'Epoch: {epoch}, Loss: {float(loss)}')

    # グラフを表示
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_log)
    plt.xlabel = 'epoch'
    plt.ylabel = 'loss'
    plt.suptitle = 'loss with train data'
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(test_loss_log_dict.keys(), test_loss_log_dict.values())
    plt.xlabel = 'epoch'
    plt.ylabel = 'loss'
    plt.suptitle = 'loss with test data'
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(accuracy_log_dict.keys(), accuracy_log_dict.values())
    plt.xlabel = 'epoch'
    plt.ylabel = 'accuracy'
    plt.suptitle = 'accuracy'
    plt.show()

    # 数件の推論結果を表示
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
    idx_to_type = {v: k for k, v in type_idx_dict.items()}
    for name, dataloader in zip(["train", "test"], [train_dataloader, test_dataloader]):
        fig, axes = plt.subplots(4, 1, figsize = (8, 8))
        model.eval()
        imgs_list = []
        labels_list = []
        pred_list = []
        with torch.no_grad(): # 勾配計算を無効にしてテストを実行
            for imgs, labels in dataloader:
                update_imgs = conv_model(imgs)
                batch_size = update_imgs.shape[0]
                outputs = model(update_imgs.view(batch_size, -1))

                probs = torch.sigmoid(outputs)
                # 確率が0.5より大きいかを判定し、0か1の予測値を取得
                predicted = (probs > 0.1).float()

        for i, ax in enumerate(axes):
            # --- 正解タイプのテキスト変換 ---
            true_indices = labels[i].nonzero(as_tuple=True)[0]
            true_types = ", ".join([idx_to_type[idx.item()] for idx in true_indices])

            # --- 予測タイプのテキスト変換 ---
            pred_indices = predicted[i].nonzero(as_tuple=True)[0]
            pred_types = ", ".join([idx_to_type[idx.item()] for idx in pred_indices])
            if not pred_types: # 1つもタイプを予測しなかった場合の処理
                pred_types = "None"
            ax.imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f"Correct type: {true_types}\nPredicted type: {pred_types}", fontsize=14)
            ax.axis('off')
        fig.suptitle(f'pred in {name} data', fontsize=16)
        fig.tight_layout()
        plt.show()