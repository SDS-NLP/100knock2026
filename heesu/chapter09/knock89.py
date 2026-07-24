import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from knock85 import DATA_DIR, MODEL_NAME, load_split
from knock87 import SEED, build_loader, evaluate, train


class BertMaxPoolClassifier(nn.Module):
    """A classification head that max-pools over token hidden states instead of
    using the [CLS] pooler that AutoModelForSequenceClassification relies on
    (knock87). Every token's representation gets to contribute its strongest
    feature activations to the sentence vector."""

    def __init__(self, model_name=MODEL_NAME, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        # Fill padding positions with -inf so they can never win the max.
        mask = attention_mask.unsqueeze(-1).bool()
        hidden = hidden.masked_fill(~mask, float("-inf"))
        pooled = hidden.max(dim=1).values
        logits = self.classifier(self.dropout(pooled))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_texts, train_labels = load_split(DATA_DIR / "train.tsv")
    dev_texts, dev_labels = load_split(DATA_DIR / "dev.tsv")

    train_loader = build_loader(train_texts, train_labels, tokenizer, shuffle=True)
    dev_loader = build_loader(dev_texts, dev_labels, tokenizer, shuffle=False)

    model = BertMaxPoolClassifier().to(device)
    train(model, train_loader, dev_loader, device)

    dev_acc = evaluate(model, dev_loader, device)
    print(f"\nfinal dev accuracy (max-pooling): {dev_acc:.4f}")


if __name__ == "__main__":
    main()
