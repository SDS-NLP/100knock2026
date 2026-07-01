import torch
import torch.nn as nn

from knock70 import build_embedding_matrix
from knock71 import TRAIN_PATH, build_dataset, load_sst2

# 72. BoWモデル(ロジスティック回帰)の構築
# 文ベクトル = 単語埋め込みの平均 → Linear(300→1) → sigmoid で「ポジティブ確率」。


class LogisticRegression(nn.Module):
    def __init__(self, E):
        super().__init__()  # 親 nn.Module の初期化(必須)

        # 埋め込み層 = 70の行列E の「E[ids] 引き」を層にしたもの。
        # freeze=True : 73の指示どおり学習中は固定。
        # padding_idx=0: PAD行(id=0)は常にゼロベクトル扱い。
        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(E), freeze=True, padding_idx=0
        )
        demb = self.emb.embedding_dim  # 300

        # 300次元の特徴 → 1値(スコア)。重み w(300個)とバイアス b を持つ線形層。
        self.fc = nn.Linear(demb, 1)

    def forward(self, input_ids):
        # input_ids: (系列長,) の long テンソル
        vecs = self.emb(input_ids)            # (系列長, 300) ← E[ids] と同じ引き
        feat = vecs.mean(dim=0)               # (300,) 単語方向に平均 = 文ベクトル
        prob = torch.sigmoid(self.fc(feat))   # スコア → 0〜1 の確率
        return prob


if __name__ == "__main__":
    E, word2id, id2word = build_embedding_matrix()
    train = build_dataset(load_sst2(TRAIN_PATH), word2id)

    model = LogisticRegression(E)
    x = train[0]["input_ids"]
    print("input_ids:", x)
    print("予測確率 p:", model(x))  # 未学習なのでデタラメ(0.5付近)。73で学習する
