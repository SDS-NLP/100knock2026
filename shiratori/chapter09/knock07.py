import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def main():
    train = pd.read_csv("data/SST-2/train.tsv", sep="\t")
    dev = pd.read_csv("data/SST-2/dev.tsv", sep="\t")

    train_dataset = Dataset.from_pandas(train)
    dev_dataset = Dataset.from_pandas(dev)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    dev_dataset = dev_dataset.rename_column("label", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # モデル
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # 学習設定
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    result = trainer.evaluate()
    print(result)


if __name__ == "__main__":
    main()



# Key                                        | Status     | 
# -------------------------------------------+------------+-
# cls.predictions.transform.dense.bias       | UNEXPECTED | 
# cls.predictions.transform.dense.weight     | UNEXPECTED | 
# cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
# cls.predictions.bias                       | UNEXPECTED | 
# cls.seq_relationship.weight                | UNEXPECTED | 
# cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
# cls.seq_relationship.bias                  | UNEXPECTED | 
# classifier.bias                            | MISSING    | 
# classifier.weight                          | MISSING    | 

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# [transformers] `logging_dir` is deprecated and will be removed in v5.2. Please set `TENSORBOARD_LOGGING_DIR` instead.
#   0%|                                                                                                                       | 0/16838 [00:00<?, ?it/s]/Users/shiratorihanae/Downloads/2026/2026春夏/100本ノック/100knock2026/shiratori/venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:752: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
#   super().__init__(loader)
# {'loss': '0.5457', 'grad_norm': '11.08', 'learning_rate': '1.988e-05', 'epoch': '0.005939'}                                                           
# {'loss': '0.4556', 'grad_norm': '13.96', 'learning_rate': '1.976e-05', 'epoch': '0.01188'}                                                            
# {'loss': '0.5686', 'grad_norm': '28.85', 'learning_rate': '1.964e-05', 'epoch': '0.01782'}                                                            
# {'loss': '0.4733', 'grad_norm': '0.4892', 'learning_rate': '1.953e-05', 'epoch': '0.02376'}                                                           
# {'loss': '0.4629', 'grad_norm': '2.276', 'learning_rate': '1.941e-05', 'epoch': '0.02969'}                                                            
# {'loss': '0.5368', 'grad_norm': '93.3', 'learning_rate': '1.929e-05', 'epoch': '0.03563'}                                                             
# {'loss': '0.4179', 'grad_norm': '0.07904', 'learning_rate': '1.917e-05', 'epoch': '0.04157'}                                                          
# {'loss': '0.5356', 'grad_norm': '12.12', 'learning_rate': '1.905e-05', 'epoch': '0.04751'}                                                            
# {'loss': '0.3702', 'grad_norm': '0.2227', 'learning_rate': '1.893e-05', 'epoch': '0.05345'}                                                           
# {'loss': '0.3904', 'grad_norm': '0.142', 'learning_rate': '1.881e-05', 'epoch': '0.05939'}                                                            
# {'loss': '0.4753', 'grad_norm': '12.49', 'learning_rate': '1.869e-05', 'epoch': '0.06533'}                                                            
# {'loss': '0.427', 'grad_norm': '91.36', 'learning_rate': '1.858e-05', 'epoch': '0.07127'}                                                             
# {'loss': '0.4733', 'grad_norm': '77.28', 'learning_rate': '1.846e-05', 'epoch': '0.07721'}                                                            
# {'loss': '0.5385', 'grad_norm': '0.292', 'learning_rate': '1.834e-05', 'epoch': '0.08315'}                                                            
# {'loss': '0.4', 'grad_norm': '0.1309', 'learning_rate': '1.822e-05', 'epoch': '0.08908'}                                                              
# {'loss': '0.4722', 'grad_norm': '2.393', 'learning_rate': '1.81e-05', 'epoch': '0.09502'}                                                             
# {'loss': '0.373', 'grad_norm': '0.2362', 'learning_rate': '1.798e-05', 'epoch': '0.101'}                                                              
# {'loss': '0.5188', 'grad_norm': '0.3878', 'learning_rate': '1.786e-05', 'epoch': '0.1069'}                                                            
# {'loss': '0.3668', 'grad_norm': '4.262', 'learning_rate': '1.774e-05', 'epoch': '0.1128'}                                                             
# {'loss': '0.5346', 'grad_norm': '110.3', 'learning_rate': '1.763e-05', 'epoch': '0.1188'}                                                             
# {'loss': '0.3877', 'grad_norm': '0.5517', 'learning_rate': '1.751e-05', 'epoch': '0.1247'}                                                            
# {'loss': '0.395', 'grad_norm': '1.363', 'learning_rate': '1.739e-05', 'epoch': '0.1307'}                                                              
# {'loss': '0.4139', 'grad_norm': '122.3', 'learning_rate': '1.727e-05', 'epoch': '0.1366'}                                                             
# {'loss': '0.48', 'grad_norm': '0.3841', 'learning_rate': '1.715e-05', 'epoch': '0.1425'}                                                              
# {'loss': '0.3798', 'grad_norm': '1.517', 'learning_rate': '1.703e-05', 'epoch': '0.1485'}                                                             
# {'loss': '0.3333', 'grad_norm': '25.39', 'learning_rate': '1.691e-05', 'epoch': '0.1544'}                                                             
# {'loss': '0.4', 'grad_norm': '3.762', 'learning_rate': '1.679e-05', 'epoch': '0.1604'}                                                                
# {'loss': '0.3917', 'grad_norm': '0.09459', 'learning_rate': '1.668e-05', 'epoch': '0.1663'}                                                           
# {'loss': '0.3999', 'grad_norm': '14.57', 'learning_rate': '1.656e-05', 'epoch': '0.1722'}                                                             
# {'loss': '0.35', 'grad_norm': '0.1978', 'learning_rate': '1.644e-05', 'epoch': '0.1782'}                                                              
# {'loss': '0.3601', 'grad_norm': '0.2914', 'learning_rate': '1.632e-05', 'epoch': '0.1841'}                                                            
# {'loss': '0.3622', 'grad_norm': '20.98', 'learning_rate': '1.62e-05', 'epoch': '0.19'}                                                                
# {'loss': '0.3995', 'grad_norm': '0.04297', 'learning_rate': '1.608e-05', 'epoch': '0.196'}                                                            
# {'loss': '0.3771', 'grad_norm': '0.4975', 'learning_rate': '1.596e-05', 'epoch': '0.2019'}                                                            
# {'loss': '0.3711', 'grad_norm': '0.4444', 'learning_rate': '1.584e-05', 'epoch': '0.2079'}                                                            
# {'loss': '0.4819', 'grad_norm': '0.5179', 'learning_rate': '1.573e-05', 'epoch': '0.2138'}                                                            
# {'loss': '0.3889', 'grad_norm': '0.415', 'learning_rate': '1.561e-05', 'epoch': '0.2197'}                                                             
# {'loss': '0.3153', 'grad_norm': '0.0682', 'learning_rate': '1.549e-05', 'epoch': '0.2257'}                                                            
# {'loss': '0.3447', 'grad_norm': '0.1041', 'learning_rate': '1.537e-05', 'epoch': '0.2316'}                                                            
# {'loss': '0.4261', 'grad_norm': '19.3', 'learning_rate': '1.525e-05', 'epoch': '0.2376'}                                                              
# {'loss': '0.3014', 'grad_norm': '0.5579', 'learning_rate': '1.513e-05', 'epoch': '0.2435'}                                                            
# {'loss': '0.4182', 'grad_norm': '1.029', 'learning_rate': '1.501e-05', 'epoch': '0.2494'}                                                             
# {'loss': '0.375', 'grad_norm': '292.9', 'learning_rate': '1.489e-05', 'epoch': '0.2554'}                                                              
# {'loss': '0.3819', 'grad_norm': '0.4389', 'learning_rate': '1.477e-05', 'epoch': '0.2613'}                                                            
# {'loss': '0.3543', 'grad_norm': '2.892', 'learning_rate': '1.466e-05', 'epoch': '0.2673'}                                                             
# {'loss': '0.2686', 'grad_norm': '0.06059', 'learning_rate': '1.454e-05', 'epoch': '0.2732'}                                                           
# {'loss': '0.3422', 'grad_norm': '0.3118', 'learning_rate': '1.442e-05', 'epoch': '0.2791'}                                                            
# {'loss': '0.3727', 'grad_norm': '458.4', 'learning_rate': '1.43e-05', 'epoch': '0.2851'}                                                              
# {'loss': '0.325', 'grad_norm': '0.4849', 'learning_rate': '1.418e-05', 'epoch': '0.291'}                                                              
# {'loss': '0.381', 'grad_norm': '230.5', 'learning_rate': '1.406e-05', 'epoch': '0.2969'}                                                              
# {'loss': '0.3461', 'grad_norm': '17.35', 'learning_rate': '1.394e-05', 'epoch': '0.3029'}                                                             
# {'loss': '0.3295', 'grad_norm': '0.3965', 'learning_rate': '1.382e-05', 'epoch': '0.3088'}                                                            
# {'loss': '0.3087', 'grad_norm': '0.02597', 'learning_rate': '1.371e-05', 'epoch': '0.3148'}                                                           
# {'loss': '0.2963', 'grad_norm': '0.0677', 'learning_rate': '1.359e-05', 'epoch': '0.3207'}                                                            
# {'loss': '0.2163', 'grad_norm': '0.06121', 'learning_rate': '1.347e-05', 'epoch': '0.3266'}                                                           
# {'loss': '0.3965', 'grad_norm': '0.1929', 'learning_rate': '1.335e-05', 'epoch': '0.3326'}                                                            
# {'loss': '0.3769', 'grad_norm': '8.492', 'learning_rate': '1.323e-05', 'epoch': '0.3385'}                                                             
# {'loss': '0.2668', 'grad_norm': '0.05741', 'learning_rate': '1.311e-05', 'epoch': '0.3445'}                                                           
# {'loss': '0.3159', 'grad_norm': '273.6', 'learning_rate': '1.299e-05', 'epoch': '0.3504'}                                                             
# {'loss': '0.3457', 'grad_norm': '0.05423', 'learning_rate': '1.287e-05', 'epoch': '0.3563'}                                                           
# {'loss': '0.3495', 'grad_norm': '5.109', 'learning_rate': '1.276e-05', 'epoch': '0.3623'}                                                             
# {'loss': '0.346', 'grad_norm': '0.09259', 'learning_rate': '1.264e-05', 'epoch': '0.3682'}                                                            
# {'loss': '0.2637', 'grad_norm': '0.07704', 'learning_rate': '1.252e-05', 'epoch': '0.3742'}                                                           
# {'loss': '0.2726', 'grad_norm': '30.76', 'learning_rate': '1.24e-05', 'epoch': '0.3801'}                                                              
# {'loss': '0.4231', 'grad_norm': '10.34', 'learning_rate': '1.228e-05', 'epoch': '0.386'}                                                              
# {'loss': '0.2873', 'grad_norm': '0.08077', 'learning_rate': '1.216e-05', 'epoch': '0.392'}                                                            
# {'loss': '0.2789', 'grad_norm': '0.05617', 'learning_rate': '1.204e-05', 'epoch': '0.3979'}                                                           
# {'loss': '0.2395', 'grad_norm': '0.04921', 'learning_rate': '1.192e-05', 'epoch': '0.4038'}                                                           
# {'loss': '0.2696', 'grad_norm': '12.85', 'learning_rate': '1.181e-05', 'epoch': '0.4098'}                                                             
# {'loss': '0.311', 'grad_norm': '0.08355', 'learning_rate': '1.169e-05', 'epoch': '0.4157'}                                                            
# {'loss': '0.2106', 'grad_norm': '0.02879', 'learning_rate': '1.157e-05', 'epoch': '0.4217'}                                                           
# {'loss': '0.362', 'grad_norm': '12.98', 'learning_rate': '1.145e-05', 'epoch': '0.4276'}                                                              
# {'loss': '0.3417', 'grad_norm': '54.22', 'learning_rate': '1.133e-05', 'epoch': '0.4335'}                                                             
# {'loss': '0.3175', 'grad_norm': '14.75', 'learning_rate': '1.121e-05', 'epoch': '0.4395'}                                                             
# {'loss': '0.2511', 'grad_norm': '0.3834', 'learning_rate': '1.109e-05', 'epoch': '0.4454'}                                                            
# {'loss': '0.3092', 'grad_norm': '21.05', 'learning_rate': '1.097e-05', 'epoch': '0.4514'}                                                             
# {'loss': '0.2779', 'grad_norm': '0.05934', 'learning_rate': '1.086e-05', 'epoch': '0.4573'}                                                           
# {'loss': '0.3338', 'grad_norm': '0.1223', 'learning_rate': '1.074e-05', 'epoch': '0.4632'}                                                            
# {'loss': '0.2269', 'grad_norm': '263.5', 'learning_rate': '1.062e-05', 'epoch': '0.4692'}                                                             
# {'loss': '0.3244', 'grad_norm': '687.9', 'learning_rate': '1.05e-05', 'epoch': '0.4751'}                                                              
# {'loss': '0.4317', 'grad_norm': '1.338', 'learning_rate': '1.038e-05', 'epoch': '0.4811'}                                                             
# {'loss': '0.3631', 'grad_norm': '0.1855', 'learning_rate': '1.026e-05', 'epoch': '0.487'}                                                             
# {'loss': '0.2068', 'grad_norm': '37.58', 'learning_rate': '1.014e-05', 'epoch': '0.4929'}                                                             
# {'loss': '0.4085', 'grad_norm': '0.564', 'learning_rate': '1.002e-05', 'epoch': '0.4989'}                                                             
# {'loss': '0.1992', 'grad_norm': '33.51', 'learning_rate': '9.905e-06', 'epoch': '0.5048'}                                                             
# {'loss': '0.2495', 'grad_norm': '19.65', 'learning_rate': '9.786e-06', 'epoch': '0.5107'}                                                             
# {'loss': '0.279', 'grad_norm': '0.2076', 'learning_rate': '9.667e-06', 'epoch': '0.5167'}                                                             
# {'loss': '0.2852', 'grad_norm': '0.2064', 'learning_rate': '9.549e-06', 'epoch': '0.5226'}                                                            
# {'loss': '0.3126', 'grad_norm': '60.17', 'learning_rate': '9.43e-06', 'epoch': '0.5286'}                                                              
# {'loss': '0.1618', 'grad_norm': '0.1032', 'learning_rate': '9.311e-06', 'epoch': '0.5345'}                                                            
# {'loss': '0.3158', 'grad_norm': '0.09042', 'learning_rate': '9.192e-06', 'epoch': '0.5404'}                                                           
# {'loss': '0.287', 'grad_norm': '0.07154', 'learning_rate': '9.074e-06', 'epoch': '0.5464'}                                                            
# {'loss': '0.303', 'grad_norm': '0.4223', 'learning_rate': '8.955e-06', 'epoch': '0.5523'}                                                             
# {'loss': '0.3026', 'grad_norm': '0.5787', 'learning_rate': '8.836e-06', 'epoch': '0.5583'}                                                            
# {'loss': '0.203', 'grad_norm': '0.05638', 'learning_rate': '8.717e-06', 'epoch': '0.5642'}                                                            
# {'loss': '0.3286', 'grad_norm': '0.1482', 'learning_rate': '8.598e-06', 'epoch': '0.5701'}                                                            
# {'loss': '0.2178', 'grad_norm': '0.0359', 'learning_rate': '8.48e-06', 'epoch': '0.5761'}                                                             
# {'loss': '0.2737', 'grad_norm': '0.1715', 'learning_rate': '8.361e-06', 'epoch': '0.582'}                                                             
# {'loss': '0.3029', 'grad_norm': '0.07328', 'learning_rate': '8.242e-06', 'epoch': '0.588'}                                                            
# {'loss': '0.2818', 'grad_norm': '2.99', 'learning_rate': '8.123e-06', 'epoch': '0.5939'}                                                              
# {'loss': '0.2382', 'grad_norm': '0.138', 'learning_rate': '8.005e-06', 'epoch': '0.5998'}                                                             
# {'loss': '0.2988', 'grad_norm': '0.2597', 'learning_rate': '7.886e-06', 'epoch': '0.6058'}                                                            
# {'loss': '0.2896', 'grad_norm': '24.82', 'learning_rate': '7.767e-06', 'epoch': '0.6117'}                                                             
# {'loss': '0.2213', 'grad_norm': '16.03', 'learning_rate': '7.648e-06', 'epoch': '0.6177'}                                                             
# {'loss': '0.3181', 'grad_norm': '0.3948', 'learning_rate': '7.529e-06', 'epoch': '0.6236'}                                                            
# {'loss': '0.3205', 'grad_norm': '0.1517', 'learning_rate': '7.411e-06', 'epoch': '0.6295'}                                                            
# {'loss': '0.3227', 'grad_norm': '0.3502', 'learning_rate': '7.292e-06', 'epoch': '0.6355'}                                                            
# {'loss': '0.2569', 'grad_norm': '236.4', 'learning_rate': '7.173e-06', 'epoch': '0.6414'}                                                             
# {'loss': '0.2136', 'grad_norm': '0.02348', 'learning_rate': '7.054e-06', 'epoch': '0.6473'}                                                           
# {'loss': '0.2186', 'grad_norm': '1.502', 'learning_rate': '6.936e-06', 'epoch': '0.6533'}                                                             
# {'loss': '0.2435', 'grad_norm': '0.1876', 'learning_rate': '6.817e-06', 'epoch': '0.6592'}                                                            
# {'loss': '0.1886', 'grad_norm': '0.02139', 'learning_rate': '6.698e-06', 'epoch': '0.6652'}                                                           
# {'loss': '0.2532', 'grad_norm': '0.04123', 'learning_rate': '6.579e-06', 'epoch': '0.6711'}                                                           
# {'loss': '0.3411', 'grad_norm': '0.2076', 'learning_rate': '6.46e-06', 'epoch': '0.677'}                                                              
# {'loss': '0.3017', 'grad_norm': '401.4', 'learning_rate': '6.342e-06', 'epoch': '0.683'}                                                              
# {'loss': '0.253', 'grad_norm': '0.2347', 'learning_rate': '6.223e-06', 'epoch': '0.6889'}                                                             
# {'loss': '0.3604', 'grad_norm': '0.1679', 'learning_rate': '6.104e-06', 'epoch': '0.6949'}                                                            
# {'loss': '0.2206', 'grad_norm': '11.09', 'learning_rate': '5.985e-06', 'epoch': '0.7008'}                                                             
# {'loss': '0.2836', 'grad_norm': '0.151', 'learning_rate': '5.866e-06', 'epoch': '0.7067'}                                                             
# {'loss': '0.3061', 'grad_norm': '0.03154', 'learning_rate': '5.748e-06', 'epoch': '0.7127'}                                                           
# {'loss': '0.2636', 'grad_norm': '104.5', 'learning_rate': '5.629e-06', 'epoch': '0.7186'}                                                             
# {'loss': '0.2693', 'grad_norm': '0.0521', 'learning_rate': '5.51e-06', 'epoch': '0.7246'}                                                             
# {'loss': '0.1961', 'grad_norm': '0.2344', 'learning_rate': '5.391e-06', 'epoch': '0.7305'}                                                            
# {'loss': '0.3006', 'grad_norm': '0.05672', 'learning_rate': '5.273e-06', 'epoch': '0.7364'}                                                           
# {'loss': '0.2574', 'grad_norm': '0.2307', 'learning_rate': '5.154e-06', 'epoch': '0.7424'}                                                            
# {'loss': '0.3209', 'grad_norm': '0.2151', 'learning_rate': '5.035e-06', 'epoch': '0.7483'}                                                            
# {'loss': '0.2837', 'grad_norm': '0.3025', 'learning_rate': '4.916e-06', 'epoch': '0.7542'}                                                            
# {'loss': '0.2597', 'grad_norm': '0.1326', 'learning_rate': '4.797e-06', 'epoch': '0.7602'}                                                            
# {'loss': '0.2103', 'grad_norm': '0.07472', 'learning_rate': '4.679e-06', 'epoch': '0.7661'}                                                           
# {'loss': '0.2668', 'grad_norm': '0.04817', 'learning_rate': '4.56e-06', 'epoch': '0.7721'}                                                            
# {'loss': '0.2551', 'grad_norm': '0.08704', 'learning_rate': '4.441e-06', 'epoch': '0.778'}                                                            
# {'loss': '0.2032', 'grad_norm': '36.26', 'learning_rate': '4.322e-06', 'epoch': '0.7839'}                                                             
# {'loss': '0.2839', 'grad_norm': '0.2697', 'learning_rate': '4.204e-06', 'epoch': '0.7899'}                                                            
# {'loss': '0.2675', 'grad_norm': '0.07014', 'learning_rate': '4.085e-06', 'epoch': '0.7958'}                                                           
# {'loss': '0.2887', 'grad_norm': '0.04107', 'learning_rate': '3.966e-06', 'epoch': '0.8018'}                                                           
# {'loss': '0.3052', 'grad_norm': '157.8', 'learning_rate': '3.847e-06', 'epoch': '0.8077'}                                                             
# {'loss': '0.2775', 'grad_norm': '0.128', 'learning_rate': '3.728e-06', 'epoch': '0.8136'}                                                             
# {'loss': '0.3307', 'grad_norm': '0.1037', 'learning_rate': '3.61e-06', 'epoch': '0.8196'}                                                             
# {'loss': '0.2912', 'grad_norm': '0.3668', 'learning_rate': '3.491e-06', 'epoch': '0.8255'}                                                            
# {'loss': '0.1592', 'grad_norm': '251', 'learning_rate': '3.372e-06', 'epoch': '0.8315'}                                                               
# {'loss': '0.2633', 'grad_norm': '0.0358', 'learning_rate': '3.253e-06', 'epoch': '0.8374'}                                                            
# {'loss': '0.2605', 'grad_norm': '0.09594', 'learning_rate': '3.135e-06', 'epoch': '0.8433'}                                                           
# {'loss': '0.2461', 'grad_norm': '0.006678', 'learning_rate': '3.016e-06', 'epoch': '0.8493'}                                                          
# {'loss': '0.2741', 'grad_norm': '0.02918', 'learning_rate': '2.897e-06', 'epoch': '0.8552'}                                                           
# {'loss': '0.2602', 'grad_norm': '0.05741', 'learning_rate': '2.778e-06', 'epoch': '0.8611'}                                                           
# {'loss': '0.1897', 'grad_norm': '0.03616', 'learning_rate': '2.659e-06', 'epoch': '0.8671'}                                                           
# {'loss': '0.253', 'grad_norm': '0.01936', 'learning_rate': '2.541e-06', 'epoch': '0.873'}                                                             
# {'loss': '0.255', 'grad_norm': '0.06742', 'learning_rate': '2.422e-06', 'epoch': '0.879'}                                                             
# {'loss': '0.2653', 'grad_norm': '0.02468', 'learning_rate': '2.303e-06', 'epoch': '0.8849'}                                                           
# {'loss': '0.1733', 'grad_norm': '0.2317', 'learning_rate': '2.184e-06', 'epoch': '0.8908'}                                                            
# {'loss': '0.2403', 'grad_norm': '0.1973', 'learning_rate': '2.066e-06', 'epoch': '0.8968'}                                                            
# {'loss': '0.2109', 'grad_norm': '0.0163', 'learning_rate': '1.947e-06', 'epoch': '0.9027'}                                                            
# {'loss': '0.2874', 'grad_norm': '3.733', 'learning_rate': '1.828e-06', 'epoch': '0.9087'}                                                             
# {'loss': '0.2', 'grad_norm': '0.1185', 'learning_rate': '1.709e-06', 'epoch': '0.9146'}                                                               
# {'loss': '0.2703', 'grad_norm': '0.1444', 'learning_rate': '1.59e-06', 'epoch': '0.9205'}                                                             
# {'loss': '0.1415', 'grad_norm': '0.07188', 'learning_rate': '1.472e-06', 'epoch': '0.9265'}                                                           
# {'loss': '0.2455', 'grad_norm': '0.02096', 'learning_rate': '1.353e-06', 'epoch': '0.9324'}                                                           
# {'loss': '0.2902', 'grad_norm': '0.2892', 'learning_rate': '1.234e-06', 'epoch': '0.9384'}                                                            
# {'loss': '0.1916', 'grad_norm': '0.0574', 'learning_rate': '1.115e-06', 'epoch': '0.9443'}                                                            
# {'loss': '0.3079', 'grad_norm': '0.2098', 'learning_rate': '9.966e-07', 'epoch': '0.9502'}                                                            
# {'loss': '0.2448', 'grad_norm': '288.6', 'learning_rate': '8.778e-07', 'epoch': '0.9562'}                                                             
# {'loss': '0.3019', 'grad_norm': '0.05552', 'learning_rate': '7.59e-07', 'epoch': '0.9621'}                                                            
# {'loss': '0.2918', 'grad_norm': '0.1315', 'learning_rate': '6.402e-07', 'epoch': '0.968'}                                                             
# {'loss': '0.2991', 'grad_norm': '1.229', 'learning_rate': '5.214e-07', 'epoch': '0.974'}                                                              
# {'loss': '0.1894', 'grad_norm': '34.14', 'learning_rate': '4.027e-07', 'epoch': '0.9799'}                                                             
# {'loss': '0.2355', 'grad_norm': '0.1044', 'learning_rate': '2.839e-07', 'epoch': '0.9859'}                                                            
# {'loss': '0.1339', 'grad_norm': '0.04083', 'learning_rate': '1.651e-07', 'epoch': '0.9918'}                                                           
# {'loss': '0.2545', 'grad_norm': '0.1191', 'learning_rate': '4.632e-08', 'epoch': '0.9977'}                                                            
# {'eval_loss': '0.3657', 'eval_accuracy': '0.9255', 'eval_runtime': '34.25', 'eval_samples_per_second': '25.46', 'eval_steps_per_second': '6.365', 'epoch': '1'}                                                                                                                                              
# Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.03s/it]
# {'train_runtime': '1.558e+04', 'train_samples_per_second': '4.323', 'train_steps_per_second': '1.081', 'train_loss': '0.3154', 'epoch': '1'}          
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 16838/16838 [4:19:37<00:00,  1.08it/s]
# /Users/shiratorihanae/Downloads/2026/2026春夏/100本ノック/100knock2026/shiratori/venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:752: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
#   super().__init__(loader)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 218/218 [00:29<00:00,  7.35it/s]
# {'eval_loss': 0.3657359778881073, 'eval_accuracy': 0.9254587155963303, 'eval_runtime': 30.4216, 'eval_samples_per_second': 28.664, 'eval_steps_per_second': 7.166, 'epoch': 1.0}