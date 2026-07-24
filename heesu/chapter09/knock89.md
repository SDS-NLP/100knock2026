# knock89

Notation used throughout:
`B` = batch size,
`T` = padded sequence length of a minibatch,
`H` = hidden size (768 for `bert-base-uncased`),
`C` = number of classes (2).

## 1. Task

Problem 89 asks for a classifier whose architecture **differs from knock87**,
suggesting either a `[CLS]`-based read-out or per-token max-pooling. Because
knock87 already reduces the sequence to the `[CLS]` token, I chose the other
suggestion — **masked max-pooling over the full sequence of token hidden
states** — so the two experiments genuinely contrast at the read-out layer while
sharing the same encoder and training recipe.

## 2. Baseline read-out (knock87)

knock87 is `AutoModelForSequenceClassification` -> `BertForSequenceClassification`.
Its sequence representation is BERT's **pooler output**, computed from a single
position:

```
h        = encoder(input)                # (B, T, H)  last_hidden_state
cls      = h[:, 0]                        # (B, H)     the [CLS] vector
pooled87 = tanh(W_pool · cls + b_pool)    # (B, H)     BertPooler (pretrained via NSP)
logits   = W_cls · dropout(pooled87)      # (B, C)     classification head
```

The entire sentence is funneled through position 0. `W_pool` (the pooler dense
layer) is inherited from pretraining; only `W_cls` is randomly initialized.

## 3. Proposed read-out (knock89)

knock89 discards the pooler and instead aggregates **every** token position by an
element-wise maximum over the time axis:

```
h            = encoder(input)                       # (B, T, H)
h_masked     = mask_pad(h, attention_mask, -inf)    # neutralize padding
pooled89[b,c]= max_{t in valid(b)}  h_masked[b,t,c] # (B, H)   per-channel max
logits       = W_cls · dropout(pooled89)            # (B, C)
```

Each of the `H` feature channels independently selects its strongest activation
across the sentence. There is **no pretrained pooler and no `tanh`** — the only
new parameters are the `H×C` classification head; everything else is the
fine-tuned encoder.

| | knock87 | knock89 |
|---|---|---|
| Base class | `AutoModelForSequenceClassification` | custom `nn.Module` over `AutoModel` |
| Positions used | `[CLS]` only | all non-padding tokens |
| Aggregation | pretrained dense + `tanh` on pos. 0 | element-wise max over `T` |
| New parameters | pooler (pretrained) + `W_cls` | `W_cls` only |
| Padding handling | irrelevant (uses pos. 0) | **must** be masked before pooling |

## 4. Implementation walkthrough

**Constructor** — `knock89.py:22-25`. `AutoModel` loads the bare encoder (no task
head, no reliance on the pooler); we attach our own linear classifier.

```python
self.bert = AutoModel.from_pretrained(model_name)   # encoder only
hidden = self.bert.config.hidden_size               # 768
self.dropout = nn.Dropout(dropout)
self.classifier = nn.Linear(hidden, num_labels)     # 768 -> 2
```

**Encoder pass** — `knock89.py:28-32`. We keep `last_hidden_state` (per-token
contextual vectors) and ignore `pooler_output`. `token_type_ids` is forwarded for
generality (all-zeros here, single-sentence input).

```python
hidden = self.bert(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids).last_hidden_state   # (B, T, H)
```

**Padding mask** — `knock89.py:34-35`. Minibatches are *dynamically padded* to the
longest sentence (see knock87's collate), so trailing positions are `[PAD]` and
must not influence the max.

```python
mask = attention_mask.unsqueeze(-1).bool()          # (B, T, 1), broadcast over H
hidden = hidden.masked_fill(~mask, float("-inf"))   # pad positions -> -inf
```

Why `-inf` and not `0`? Max-pooling's identity (neutral) element is `-inf`: it can
never be selected as long as any real token exists, which is guaranteed because
every sequence contains at least `[CLS]` and `[SEP]`. Filling with `0` would be a
bug — a padding `0` can exceed a legitimately negative activation and silently
poison that channel. (Practical caveat: under fp16 autocast, prefer a large finite
negative like `-1e4` to avoid `-inf`-induced NaNs; this run is fp32, so `-inf` is
safe.)

**Pool, classify, package output** — `knock89.py:36-39`.

```python
pooled = hidden.max(dim=1).values                   # (B, H); .values drops argmax
logits = self.classifier(self.dropout(pooled))      # (B, C)
loss = None if labels is None else F.cross_entropy(logits, labels)
return SequenceClassifierOutput(loss=loss, logits=logits)
```

`torch.max(dim=1)` returns a `(values, indices)` namedtuple; we take `.values`.
Returning a `SequenceClassifierOutput` (not a bare tuple) is deliberate — see §5.
`labels` is optional so the *same* `forward` serves both training (returns
`.loss`) and inference (`.logits` only).

## 5. Results and interpretation

| Read-out | Dev accuracy |
|---|---|
| `[CLS]` pooler (knock87) | **0.9220** |
| Masked max-pool (knock89) | **0.9197** |
