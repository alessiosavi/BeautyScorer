import random
from glob import glob
from typing import Dict, Union

import numpy as np
import torch
import yaml
from deepface import DeepFace
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.io import decode_image

import dataset

MIN_SCORE = 1


def load_raw_conf(file):
    with open(file, "r") as file:
        return yaml.safe_load(file)


def load_conf(file):
    CONF = load_raw_conf(file)
    # Channel, height and width for the photos
    C = CONF["C"]
    H = CONF["H"]
    W = CONF["W"]

    # Height and width for the extracted faces
    F_H = CONF["F_H"]
    F_W = CONF["F_W"]

    # Max photos allowed for score a person
    N_MAX = CONF["N_MAX"]
    # Number of classes (scores)
    N_CLASSES = CONF["N_CLASSES"]

    return CONF, C, H, W, F_H, F_W, N_MAX, N_CLASSES


def read_photo(photo):
    return np.asarray(Image.open(photo))


def load_image(path, toPil=True):
    img = decode_image(path)
    return img.permute(1, 2, 0) if toPil else img


def toPil(img):
    if img.shape[0] == 3:
        img = (
            img.permute(1, 2, 0)
            if torch.is_tensor(img)
            else np.transpose(img, (1, 2, 0))
        )
    return img


def show_img(img, title=""):
    if isinstance(img, str):
        img = load_image(img)
    plt.imshow(img)
    plt.plot()
    plt.title(title)
    plt.show()


def extract_face(img, threshold=0.3):
    img = toPil(img)
    img = img.numpy() if torch.is_tensor(img) else img
    res = DeepFace.extract_faces(
        img,
        color_face="bgr",
        detector_backend="yolov12l",
        enforce_detection=False,
        expand_percentage=50,
    )
    if len(res) != 1 or res[0]["confidence"] < threshold:
        return None
    return torch.from_numpy(res[0]["face"])


def get_face_embedding(img: str, threshold=0.3):
    res = DeepFace.represent(
        img,
        detector_backend="yolov12l",
        model_name="Dlib",
        enforce_detection=False,
    )
    if len(res) != 1 or res[0]["face_confidence"] < threshold:
        return None
    return torch.Tensor(res[0]["embedding"])


def compute_class_weights(df):
    value_counts = df["score"].value_counts().sort_index()
    num_classes = df["score"].nunique()
    min_class = df["score"].min()

    class_counts = torch.ones(num_classes) * 1e-6  # Avoid zero division
    for score, count in value_counts.items():
        class_idx = score - min_class
        class_counts[class_idx] = count
    total_samples = len(df)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.clamp(class_weights, max=5)  # Cap to prevent instability
    return class_weights


"""
The dataset file is a CSV with two column: the folder where to find the photos, and the score associated to the person
"""


def load_raw_dataset(df, BASEPATH):
    raw_dataset = []
    for row in df[["folder_name", "score"]].fillna(-1).to_dict(orient="records"):
        images = glob(f'{BASEPATH}/{row["folder_name"]}/*')
        if len(images) > 0:
            raw_dataset.append(
                {"id": row["folder_name"], "score": row["score"], "photos": images}
            )
    return raw_dataset


def load_data(person_id, basepath, device=torch.device("cpu")):
    images = list(glob(f"{basepath}/{person_id}/*"))
    d = [{"id": person_id, "score": 0, "photos": images}]
    ds = dataset.BeautyDataset(d)
    dl = DataLoader(ds, batch_size=1, pin_memory=True, collate_fn=ds.collate_fn)
    batch = next(iter(dl))
    for k in batch:
        batch[k] = batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k]
    del batch["ids"]
    del batch["scores"]
    return batch


def show_person(person_id, basepath, n=9):
    images = list(glob(f"{basepath}/{person_id}/*"))
    [
        show_img(photo, person_id)
        for photo in random.sample(images, k=min(n, len(images)))
    ]
