import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import models

from utils import extract_face, load_conf, load_image

_, C, H, W, F_H, F_W, N_MAX, N_CLASSES = load_conf("conf.yaml")
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)


class BeautyDataset(Dataset):
    def __init__(self, data):
        super(BeautyDataset, self).__init__()
        self.data = data  # List of dicts: {'Id': int, 'photos': list of photos path, 'score': int/float}
        self.preprocess = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        B = len(batch)
        photos_tensor = torch.zeros((B, N_MAX, C, H, W), dtype=torch.float32)
        photos_mask = torch.zeros((B, N_MAX), dtype=torch.bool)
        faces_tensor = torch.zeros((B, N_MAX, C, F_H, F_W), dtype=torch.float32)
        faces_mask = torch.zeros((B, N_MAX), dtype=torch.bool)
        scores = torch.zeros(B, dtype=torch.int32)

        ids = []
        for i, entry in enumerate(batch):
            raw_photos = [load_image(p, toPil=False) for p in entry["photos"]]
            # Change order of the photos
            random.shuffle(raw_photos)
            raw_photos = torch.stack(raw_photos)

            # Apply transforms to photos
            photo_tensors = self.preprocess(raw_photos)

            raw_faces = [extract_face(photo) for photo in raw_photos]

            for j, (photo, face) in enumerate(zip(photo_tensors, raw_faces)):
                photos_tensor[i, j] = photo
                photos_mask[i, j] = True

                if face is not None:
                    faces_tensor[i, j] = self.preprocess(face.permute(2, 0, 1))
                    faces_mask[i, j] = True

            scores[i] = entry["score"]
            ids.append(entry["id"])
        out = {
            "ids": np.asarray(ids),
            "photos_tensor": photos_tensor,
            "photos_mask": photos_mask,
            "faces_tensor": faces_tensor,
            "faces_mask": faces_mask,
            "scores": torch.IntTensor(scores),
        }
        return out
