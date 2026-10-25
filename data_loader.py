"""读取UCI干豆数据。训练步骤在train.py中，这里只处理ZIP文件。"""
from hashlib import sha256
from io import StringIO
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
from scipy.io import arff

DATA_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DATA_SHA256 = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"


def load_dry_bean_data(data_directory):
    """ZIP缺失时下载；校验版本后，返回一个pandas表格。"""
    data_directory = Path(data_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    zip_path = data_directory / "dry_bean_dataset.zip"
    if not zip_path.exists():
        urlretrieve(DATA_URL, zip_path)
    if sha256(zip_path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("原始ZIP不完整或版本不符，请重新下载官方数据。")

    with ZipFile(zip_path) as archive:
        arff_name = next(name for name in archive.namelist() if name.endswith(".arff"))
        text = archive.read(arff_name).decode("utf-8")
    records, _ = arff.loadarff(StringIO(text))
    data = pd.DataFrame(records)
    data["Class"] = data["Class"].map(lambda value: value.decode("utf-8"))
    return data
