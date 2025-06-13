import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

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

class TypePredictor(nn.Module):
    def __init__(self, 
                out_channels1, out_channels2,
                hidden_dim, output_dim, 
                dropout_rate=0.5):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 48×48に
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 24×24に
        )
        self.poolpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear((3+out_channels2) * 24 * 24, hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = self.conv_block(x) # 48 * 38
        x1 = self.conv_block2(x1) # out_channels2 * 24 * 24
        x2 = self.poolpool(x) # 3 * 24 * 24
        x_conbined = torch.cat((x1, x2), dim=1) # (3 + out_channesl2) * 24 * 24
        x_flattend = x_conbined.view(x_conbined.size(0), -1) # ここで平坦化（バッチ次元は維持）
        x_out = self.fc1(x_flattend)
        x_out = self.tanh(x_out)
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        return x_out

setting_TypePredictor = {
    'out_channels1':16,
    'out_channels2':32,
    'hidden_dim':256,
    'output_dim':18,
    'dropout_rate':0.5
}

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

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((96, 96), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=1), # いい感じにデータを嵩ます
        transforms.ToTensor()
    ])

    # 訓練用データセットの作成
    train_dataset = PokeDataset(id_list=train_id_list, poke_data=poke_data, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # テスト用データセットの作成
    test_dataset = PokeDataset(id_list=test_id_list, poke_data=poke_data)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 例: 各タイプのサンプル数をカウント
    pos_sample = torch.zeros(18) # そのタイプを持つポケモンの数
    poke_num = 0
    for imgs, labels in train_dataloader:
        poke_num += len(labels)
        pos_sample += labels.sum(dim=0)
    nega_sample = torch.ones(18) * poke_num - pos_sample # そのタイプを持たないポケモンの数
    pos_weight = nega_sample/pos_sample
    print(pos_weight)

    model = TypePredictor(**setting_TypePredictor)

    learning_rate = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # バイナリクロスエントロピー with ロジッツ損失
    # ※ ロジッツは出力層から直接出てくる生の値のことらしい(シグモイドやソフトマックス適応前)
    n_epochs = 100

    # --- ReduceLROnPlateau スケジューラの初期化 ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # 監視対象はテスト損失 (mode='min')
        factor=0.5, # 学習率を0.5倍にする
        patience=3, # 3エポック改善がなければ学習率を減衰
        threshold=1e-2, # 改善されたとみなすライン
        min_lr=1e-5, # 1e-6: 学習率の下限
        verbose=True # 学習率が変更されたらメッセージを出力
    )

    # ===== 学習フェーズ =====
    i = 0
    train_losses = []
    test_losses = []

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        model.train()
        running_train_loss = 0.0
        for imgs, labels in train_dataloader:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad() # 勾配のリセット
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        train_losses.append(running_train_loss / len(train_dataloader))

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for imgs, labels in test_dataloader:
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                running_test_loss += loss.item()

        test_losses.append(running_test_loss / len(test_dataloader))
        scheduler.step(test_losses[-1])
        if optimizer.param_groups[0]['lr'] <= 1e-5:
            print(f"Learning rate has reached its minimum ({1e-5:.6f}). Stopping training.")
            break # 学習ループを終了

        pbar.set_description(f'loss | (train, test) = ({float(train_losses[-1]):.2f}, {float(test_losses[-1]):.2f}) lr: {optimizer.param_groups[0]['lr']:.6f}')

    plt.figure(figsize = (10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train loss')
    plt.plot(range(1, len(test_losses)+1), test_losses, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    PATH = 'model_states/class_balanced.pth'
    torch.save(model.state_dict(), PATH)