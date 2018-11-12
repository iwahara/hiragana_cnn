# -*- coding: utf-8 -*-
import urllib.request
import sys
import zipfile
from pathlib import Path

from PIL import Image
import numpy as np

DATASET_URL = "http://lab.ndl.go.jp/dataset/hiragana73.zip"
DATASET_ZIP_PATH = "dataset/hiragana73.zip"
DATASET_EXTRAXT_PATH = "dataset"
DATASET_PATH = "dataset/hiragana73"
GRAYSCALED_DATASET_PATH = "dataset/gray_scaled"

categorical_list = list("あいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわゐゑをん")

zip_path = Path(DATASET_ZIP_PATH)
if not zip_path.exists():
    print("downdload dataset.", flush=True)
    urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP_PATH)
    print("downdloaded.", flush=True)

p = Path(DATASET_PATH)
dir_list = sorted(list(p.glob("*")))
if not p.exists() or not len(dir_list) == len(categorical_list):
    with zipfile.ZipFile(DATASET_ZIP_PATH) as existing_zip:
        print("extract dataset", flush=True)
        existing_zip.extractall(DATASET_EXTRAXT_PATH)
        print("extracted", flush=True)


data_path = Path(GRAYSCALED_DATASET_PATH) / "hiragana.npz"
category_path = Path(GRAYSCALED_DATASET_PATH) / "label.npz"

if not data_path.exists() or not category_path.exists():
    data_list = []
    category_list = []
    for (dir, category) in zip(dir_list, categorical_list):
        print("{}をグレースケール中…".format(category), flush=True)
        png_list = sorted(list(dir.glob("*.png")))

        for png in png_list:
            im = Image.open(png)
            new_im = np.array(im.convert('L')) # グレースケール化してnumpy化
            data_list.append(new_im)
            category_list.append(categorical_list.index(category))


    data_path.parent.mkdir(parents=True, exist_ok=True)

    np_data = np.array(data_list)
    np_category = np.array(category_list)
    print(np_data.shape)
    np.savez(data_path, np_data)
    np.savez(category_path, np_category)
    print(np_category)

    print("gray scaled dataset.")
