from knock70 import Word2Vectors
import csv
import torch



def process_tsv_to_dict(file_path, word2id) -> list:

    dataset = []

    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            
            # 'sentence'列と'label'列の値を取得
            text  = row.get('sentence', '') 
            label = float(row.get('label', ''))

            # make input_ids
            input_ids = []
            words = text.split()
            for word in words:
                if word in word2id:
                    input_ids.append(word2id[word])
            
            if len(input_ids) == 0:
                continue
            
            # 指定されたフォーマットの辞書オブジェクトを作成
            instance = {
                'text'      : text,
                'label'     : torch.tensor([label]),
                'input_ids' : torch.tensor(input_ids)

            }
            
            dataset.append(instance)
            
    return dataset


if __name__=="__main__":

    file_path_embedding = "GoogleNews-vectors-negative300.bin"
    file_path_train     = "SST-2/train.tsv"
    file_path_test      = "SST-2/dev.tsv"

    word2id, id2word, E = Word2Vectors(file_path_embedding)

    train_dataset = process_tsv_to_dict(file_path_train, word2id)
    test_dataset  = process_tsv_to_dict(file_path_test, word2id)

    print("First 5 of train_data\n")
    for i in range(5):
        print(f"{train_dataset[i]}\n")


        


