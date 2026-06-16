"""60. データの入手・取得

GLUE の Web サイトから SST-2 データセットを取得し、
学習データ（train.tsv）と検証データ（dev.tsv）について
ポジティブ (1) とネガティブ (0) の事例数をカウントする。
"""

from collections import Counter
from pathlib import Path
import csv
import shutil
import tempfile
import ssl
import urllib.request
import zipfile


SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
SST2_DIR = DATA_DIR / "SST-2"


def build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that works across local Python installs."""
    try:
        import certifi
    except ImportError:
        return ssl._create_unverified_context()

    return ssl.create_default_context(cafile=certifi.where())


def download_sst2() -> None:
    train_path = SST2_DIR / "train.tsv"
    dev_path = SST2_DIR / "dev.tsv"
    if train_path.exists() and dev_path.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)

    try:
        context = build_ssl_context()
        with urllib.request.urlopen(SST2_URL, context=context) as response:
            with zip_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
    finally:
        if zip_path.exists():
            zip_path.unlink()


def count_labels(tsv_path: Path) -> Counter:
    counts = Counter()
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return counts
        for row in reader:
            if not row:
                continue
            counts[row[-1]] += 1
    return counts


def main() -> None:
    download_sst2()

    for split_name in ("train", "dev"):
        counts = count_labels(SST2_DIR / f"{split_name}.tsv")
        print(f"{split_name}.tsv")
        print(f"  1 (positive): {counts['1']}")
        print(f"  0 (negative): {counts['0']}")


if __name__ == "__main__":
    main()
