import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import models
from psycopg import Cursor
from utils import extract_face, load_conf, load_image, get_face_embedding
import pandas as pd
_, C, H, W, F_H, F_W, N_MAX, N_CLASSES = load_conf("conf.yaml")
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)


class BeautyDataset(Dataset):
    def __init__(self, data, cursor:Cursor):
        super(BeautyDataset, self).__init__()
        self.data = data  # List of dicts: {'Id': int, 'photos': list of photos path, 'score': int/float}
        self.preprocess = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
        self.cursor = cursor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        B = len(batch)
        photos_tensor = torch.zeros((B, N_MAX, C, H, W), dtype=torch.float32)
        photos_mask = torch.zeros((B, N_MAX), dtype=torch.bool)
        faces_tensor = torch.zeros((B, N_MAX, 128), dtype=torch.float32)
        faces_mask = torch.zeros((B, N_MAX), dtype=torch.bool)
        scores = torch.zeros(B, dtype=torch.int32)

        ids = [entry['id'] for entry in batch]
        self.cursor.execute("""select * from photo_embeddings where folder_name = ANY(%s);""",[ids])
     
     
        db_res= pd.DataFrame(self.cursor.fetchall())
        db_res.columns = ['folder_name', 'photo_id', 'embedding']
        res = []
        for l,g in db_res.groupby('folder_name'):
            res.append({'id':l,'embeddings':g['embedding'].to_list()})
        for i, entry in enumerate(batch):
            face_embedding = res[i]['embeddings']
            raw_photos = [load_image(p, toPil=False) for p in entry["photos"]]
            
            # Change order of the photos at each iteration            
            tmp = list(zip(face_embedding,raw_photos))
            random.shuffle(tmp)
            face_embedding,raw_photos = zip(*tmp)
            
            raw_photos = torch.stack(raw_photos)

            # Apply transforms to photos
            photo_tensors = self.preprocess(raw_photos)

            for j, photo in enumerate(photo_tensors):
                photos_tensor[i, j] = photo
                photos_mask[i, j] = True
                # emb = get_face_embedding(entry["photos"][j])
                if len(face_embedding[j]) != 0:
                    faces_tensor[i, j] = torch.Tensor(face_embedding[j])
                    faces_mask[i, j] = True

            scores[i] = entry["score"]
        out = {
            "ids": np.asarray(ids),
            "photos_tensor": photos_tensor,
            "photos_mask": photos_mask,
            "faces_tensor": faces_tensor,
            "faces_mask": faces_mask,
            "scores": torch.IntTensor(scores),
        }
        return out
