"""第8章 knock79: 多層パーセプトロンを用いる極性分類器。"""

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    """系列長の降順に並べ、ID 0（<PAD>）でパディングする。"""
    sorted_batch = sorted(batch, key=lambda example: len(example["input_ids"]), reverse=True)
    max_length = max(len(example["input_ids"]) for example in sorted_batch)
    input_ids = torch.zeros((len(sorted_batch), max_length), dtype=torch.long)
    labels = []
    for row, example in enumerate(sorted_batch):
        token_ids = example["input_ids"]
        input_ids[row, : len(token_ids)] = token_ids
        labels.append(example["label"])
    return {"input_ids": input_ids, "label": torch.stack(labels)}


class MLPBoWClassifier(nn.Module):
    """平均単語埋め込みを2層MLPに入力して極性を分類する。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_size: int,
        fine_tune_embedding: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=not fine_tune_embedding, padding_idx=0
        )
        embedding_dim = embedding_matrix.size(1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).unsqueeze(-1)
        word_vectors = self.embedding(input_ids) * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_vector = word_vectors.sum(dim=1) / lengths
        return self.classifier(mean_vector)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch(batch, device)
            predictions = (model(batch["input_ids"]) >= 0).long()
            labels = batch["label"].long()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="多層パーセプトロンによるSST-2極性分析")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--fine-tune-embedding", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    torch.manual_seed(42)
    chapter_dir = Path(__file__).resolve().parent
    device = select_device(args.device)
    embedding_matrix = torch.load(chapter_dir / "embedding_matrix.pt", map_location="cpu")
    train_dataset = torch.load(chapter_dir / "train_dataset.pt", map_location="cpu", weights_only=False)
    dev_dataset = torch.load(chapter_dir / "dev_dataset.pt", map_location="cpu", weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = MLPBoWClassifier(
        embedding_matrix,
        hidden_size=args.hidden_size,
        fine_tune_embedding=args.fine_tune_embedding,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr
    )
    print(model)
    print(f"使用デバイス: {device}")
    print(f"単語埋め込みを更新するか: {model.embedding.weight.requires_grad}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            loss = criterion(model(batch["input_ids"]), batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % args.log_interval == 0:
                print(f"epoch={epoch}\tstep={step}/{len(train_loader)}\tloss={total_loss / step:.4f}")

        accuracy = evaluate(model, dev_loader, device)
        print(f"epoch={epoch}\ttrain_loss={total_loss / len(train_loader):.4f}\tdev_accuracy={accuracy:.4f}")

    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, chapter_dir / "mlp_bow_model.pt")
    print(f"モデルを保存しました: {chapter_dir / 'mlp_bow_model.pt'}")


if __name__ == "__main__":
    main()
