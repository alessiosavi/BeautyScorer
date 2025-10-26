#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import random
import shutil
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

tqdm.pandas()

import dataset
import utils
import model as model_imp


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.cuda.set_device(0)


CONF = utils.load_conf("conf.yaml")[0]
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
BASEPATH = CONF["DATASET_PHOTO_BASEPATH"]
CONF


# In[2]:


def load_dataset(dataset):
    df = pd.read_csv(dataset, usecols=["folder_name", "score"])
    df = (
        df[~df["score"].isna()]
        .drop_duplicates()
        .reset_index(drop=True)
        .convert_dtypes()
    )
    return df


df = load_dataset(CONF["DATASET_FILE"])
mask = df["folder_name"].progress_apply(
    lambda x: os.path.exists(os.path.join(BASEPATH, x))
)
df = df[mask]


# In[3]:


def cache_dataset(df, basepath, cache_path):
    try:
        os.makedirs(cache_path, exist_ok=False)
    except:
        pass

    for folder_name in tqdm(list(df["folder_name"].unique())):
        dst = os.path.join(cache_path, folder_name)
        src = os.path.join(basepath, folder_name)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copytree(src, dst, symlinks=True)


cache_dataset(df, BASEPATH, "/dev/shm/image_dataset")
BASEPATH = "/dev/shm/image_dataset"


# In[4]:


t = df["score"].value_counts()
t.plot(kind="barh")


# In[5]:


utils.compute_class_weights(df), df["score"].value_counts().sort_index()


# In[6]:


raw_dataset = utils.load_raw_dataset(df, BASEPATH)
unique_scores = set([row["score"] for row in raw_dataset])
MAX_SCORE = max(unique_scores)
MIN_SCORE = min(unique_scores)
assert MAX_SCORE - MIN_SCORE == CONF["N_CLASSES"] - 1
raw_dataset[0]


# In[7]:


emb = utils.get_face_embedding(raw_dataset[0]["photos"][0])
emb


# In[ ]:


# In[8]:


# import importlib

# importlib.reload(utils)
# importlib.reload(model_imp)
# importlib.reload(dataset)


# In[9]:


raw_ds = dataset.BeautyDataset(raw_dataset)
for x in raw_ds:
    print(x)
    break


# In[10]:


BATCH_SIZE = CONF["BATCH_SIZE"]
raw_dl = DataLoader(
    raw_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    collate_fn=raw_ds.collate_fn,
)
for x in raw_dl:
    # print(x)
    for a, b in zip(x.keys(), x.values()):
        print(f"{a} -> {b.shape}")
    break


# In[11]:


lr = CONF["LR"]
EPOCHS = CONF["EPOCHS"]

model = model_imp.BeautyScoreModel(conf=CONF)
model_params = list(filter(lambda p: p.requires_grad, model.parameters()))

class_weights = utils.compute_class_weights(df)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
optimizer = optim.AdamW(model_params, lr=lr, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

finetune = CONF["MODEL"]
if finetune != "":
    try:
        model.load_state_dict(torch.load(finetune, weights_only=True))
    except Exception as e:
        print(f"Impossible to load model due to {e}. Training from scratch")


# In[ ]:


# # Assert that changing the order of the images does not change the result
# def randomize(t1, t2):
#     idx = torch.randperm(t1.shape[1])
#     t1 = t1[:, idx].view(t1.size())
#     t2 = t2[:, idx].view(t2.size())
#     return t1, t2


# a, b = randomize(x["photos_tensor"], x["photos_mask"])
# c, d = randomize(x["faces_tensor"], x["faces_mask"])
# model.eval()
# res1 = model(x["photos_tensor"], x["photos_mask"], x["faces_tensor"], x["faces_mask"])
# res2 = model(a, b, c, d)
# model.train()
# (res1 - res2).abs().max()


# In[ ]:


N_CLASSES = CONF["N_CLASSES"]
STEPS = len(raw_ds) // BATCH_SIZE
CLASS_DELTA = MAX_SCORE - N_CLASSES
PATIENCE = CONF["PATIENCE"]


def train(model, dl_train):
    best_loss = 100
    early_stop_counter = 0
    for epoch in tqdm(range(EPOCHS), position=0):
        model.train()
        train_loss = 0
        with tqdm(dl_train, total=STEPS, position=1) as pbar:
            for step, batch in enumerate(pbar):
                photos_tensor = batch["photos_tensor"].to(device)
                photos_mask = batch["photos_mask"].to(device)
                faces_tensor = batch["faces_tensor"].to(device)
                faces_mask = batch["faces_mask"].to(device)
                labels = batch["scores"].to(device) - MIN_SCORE  # 0-8

                optimizer.zero_grad()
                # with torch.amp.autocast(
                #     device_type=torch.device
                # ):
                logits = model(photos_tensor, photos_mask, faces_tensor, faces_mask)
                loss = criterion(logits, labels.long())

                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_description(
                    f"Epoch {epoch+1}/{EPOCHS} - Train - Loss: {round(train_loss / (step+1),4)}"
                )

            avg_train_loss = train_loss / STEPS
            model_imp.score_persons(batch, model, 5, BASEPATH)
            scheduler.step()  # Update LR

            # Early stopping
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                torch.save(model.state_dict(), "models/model_state_dict.pth")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    return model


gc.collect()
torch.cuda.empty_cache()
model.to(device)
model = train(model, raw_dl)
torch.save(model.state_dict(), "models/model_state_dict_final.pt")
torch.save(model, "models/model.pt")


# # INFERENCE

# In[ ]:


import datasets

faces_example = datasets.load_dataset("logasja/lfw", split="train", streaming=True)


# In[ ]:


examples = []
for idx, row in tqdm(enumerate(faces_example)):
    examples.append(row)
    if idx > 1000:
        break


# In[ ]:


import pandas as pd

example_df = pd.DataFrame(examples)


# In[ ]:


import os

tmp_folder = "/dev/shm/example_dataset"
os.makedirs(tmp_folder, exist_ok=True)


# In[ ]:


for label, group in tqdm(
    example_df.groupby("label"), total=example_df["label"].nunique()
):
    if len(group) < 2:
        continue
    os.makedirs(f"{tmp_folder}/{label}", exist_ok=True)
    for idx, row in group.iterrows():
        row["image"].save(f"{tmp_folder}/{label}/{idx}.jpg")


# In[ ]:


from glob import glob

raw_dataset = []
for label in example_df["label"].astype(str).unique():
    raw_dataset.append(
        {"id": label, "score": -1, "photos": glob(f"{tmp_folder}/{label}/*")}
    )
raw_dataset = list(filter(lambda x: len(x["photos"]) > 0, raw_dataset))
raw_dataset[0]


# In[ ]:


example_person = random.choice(raw_dataset)
example_person


# In[ ]:


utils.show_person(example_person["id"], tmp_folder)


# In[ ]:


score, probs = score_person(example_person["id"], model, tmp_folder)
print(f"Score: {score} | Probabilities: {probs}")
