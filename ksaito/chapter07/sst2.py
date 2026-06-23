import csv
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


SST2_URL = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'
DATA_DIR = Path(__file__).resolve().parent / 'data'
SST2_DIR = DATA_DIR / 'SST-2'
TRAIN_PATH = SST2_DIR / 'train.tsv'
DEV_PATH = SST2_DIR / 'dev.tsv'
TEST_PATH = SST2_DIR / 'test.tsv'


def download_sst2():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_PATH.exists() and DEV_PATH.exists():
        return SST2_DIR

    zip_path = DATA_DIR / 'SST-2.zip'
    urlretrieve(SST2_URL, zip_path)
    with ZipFile(zip_path) as zip_file:
        zip_file.extractall(DATA_DIR)

    return SST2_DIR


def read_sst2(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row['sentence'], row['label']


if __name__ == '__main__':
    download_sst2()
    for name, path in [('train', TRAIN_PATH), ('dev', DEV_PATH)]:
        print(f'{name}: {sum(1 for _ in read_sst2(path))} examples')
