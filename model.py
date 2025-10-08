import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
import utils
from tqdm.auto import tqdm
import random

class BeautyScoreModel(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_encoder_layers=4,
        num_heads=8,
        ff_dim=2048,
        dropout=0.2,
        conf=None,
    ):
        super().__init__()
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        self.conf = conf

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # for i in range(12, 16):  # Layers 12 to 16
        #     for param in self.feature_extractor[i].parameters():
        #         param.requires_grad = True

        self.feature_dim = 960
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.photo_proj = nn.Linear(self.feature_dim, embed_dim)
        self.face_proj = nn.Linear(self.feature_dim, embed_dim)
        self.photo_cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.face_cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
        )
        self.photo_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.face_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.conf["N_CLASSES"]),
        )

    def forward(self, photos_tensor, photos_mask, faces_tensor, faces_mask):
        batch_size = photos_tensor.size(0)
        photos_flat = photos_tensor.view(
            batch_size * self.conf["N_MAX"],
            self.conf["C"],
            self.conf["H"],
            self.conf["W"],
        )
        faces_flat = faces_tensor.view(
            batch_size * self.conf["N_MAX"],
            self.conf["C"],
            self.conf["F_H"],
            self.conf["F_W"],
        )
        photo_feats_flat = self.feature_extractor(photos_flat)
        face_feats_flat = self.feature_extractor(faces_flat)

        photo_feats_pooled = self.pool(photo_feats_flat).view(-1, self.feature_dim)
        photo_feats = photo_feats_pooled.view(
            batch_size, -1, self.feature_dim
        )  # note: 9 to -1 for general
        photo_emb = self.photo_proj(photo_feats)
        cls_photo = self.photo_cls.repeat(batch_size, 1, 1)
        photo_input = torch.cat([cls_photo, photo_emb], dim=1)
        photo_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=photos_mask.device),
                ~photos_mask,
            ],
            dim=1,
        )
        photo_encoded = self.photo_encoder(
            photo_input.transpose(0, 1), src_key_padding_mask=photo_padding_mask
        ).transpose(0, 1)
        photo_vec = photo_encoded[:, 0, :]

        face_feats_pooled = self.pool(face_feats_flat).view(-1, self.feature_dim)
        face_feats = face_feats_pooled.view(batch_size, -1, self.feature_dim)
        face_emb = self.face_proj(face_feats)
        cls_face = self.face_cls.repeat(batch_size, 1, 1)
        face_input = torch.cat([cls_face, face_emb], dim=1)
        face_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=faces_mask.device),
                ~faces_mask,
            ],
            dim=1,
        )
        face_encoded = self.face_encoder(
            face_input.transpose(0, 1), src_key_padding_mask=face_padding_mask
        ).transpose(0, 1)
        face_vec = face_encoded[:, 0, :]
        combined = torch.cat([photo_vec, face_vec], dim=1)
        score = self.mlp(combined)
        return score


MIN_SCORE = 1


# Example inference
def predict(model, photos_tensor, photos_mask, faces_tensor, faces_mask):
    model.eval()
    with torch.no_grad():
        logits = model(
            photos_tensor, photos_mask, faces_tensor, faces_mask
        )  # [batch_size, 9]
        probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        predicted_classes = torch.argmax(
            logits, dim=-1
        )  # [batch_size], values in [0, 8]
        predicted_scores = predicted_classes + MIN_SCORE  # Convert back to scores 1-9
    model.train()
    return predicted_scores, probabilities

def score_person(person_id, model):
    batch = utils.load_data(person_id)
    score, probability = predict(model, **batch)
    score = score.detach().to("cpu").item()
    probability = probability.detach().to("cpu").squeeze(0)
    prob_idx = torch.argsort(probability, descending=True)
    probs = {
        int(j.item() + MIN_SCORE): round(probability[j].item() * 100, 3)
        for j in prob_idx
    }
    return score, probs


def score_persons(batch, model, k=5):
    max_len = len(batch["ids"])
    sample_ids = random.sample(list(range(max_len)), k=min(max_len, k))
    for idx in tqdm(sample_ids):
        person = batch["ids"][idx]
        score, probability = score_person(person, model)
        print(
            f"Person: {person} -> Score: {score} | Prob: {probability} | RealScore: {batch['scores'][idx]}"
        )
        utils.show_person(person, 1)
