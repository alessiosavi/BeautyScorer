# BeautyScorer: A Permutation-Invariant Beauty Scoring Neural Network

## Overview

BeautyScoreModel is a deep learning model designed to predict beauty scores (as integers from 1 to 9) for individuals based on up to 9 photos and corresponding face crops. The model treats the problem as a multi-class classification task, outputting probabilities over 9 classes. It is built using PyTorch and leverages pre-trained `MobileNetV3` for feature extraction, transformer encoders for aggregating variable-length inputs, and an MLP head for final prediction.

Key features:

- **Input Handling**: Supports variable numbers of photos and faces (up to 9 each), with padding and masks to handle fewer inputs.
- **Permutation Invariance**: The order of photos/faces does not affect the output, making it robust to shuffling (a "hidden gem" for set-based inputs like photo collections).
- **Efficiency**: Uses lightweight components for faster training/inference.
- **Adaptability**: Fine-tuned for beauty scoring, with options for handling class imbalance and further fine-tuning on expanded datasets.

This repository includes the model implementation, preprocessing utilities, training/fine-tuning scripts, and inference examples.

## Architecture

The model processes photos and faces separately before fusing their representations. Here's a high-level breakdown:

1. **Feature Extraction (`MobileNetV3`)**:
   - A pre-trained `MobileNetV3` (features only, no classifier) extracts spatial features from each image.
   - Input: RGB images resized to 224x224.
   - Output per image: A feature map `[960, 7, 7]`, pooled to `[960]` via adaptive average pooling.
   - **Technical Choice**: `MobileNetV3` is lightweight (efficient convolutions with depthwise separables) and pre-trained on ImageNet, providing strong generic features.
   - **Peculiarity**: Shared extractor for photos and faces, but separate downstream processing allows specialization.

2. **Projection Layers**:
   - Linear layers map the 960-D features to an embedding dimension (default: 512).
   - **Technical Choice**: Dimensionality reduction improves efficiency (transformer attention scales quadratically with dimension) and acts as a learnable adapter for task-specific features (e.g., beauty-related traits vs. ImageNet objects).
   - **Why Not Skip?**: Direct 960-D input would increase compute; projection creates a bottleneck for better generalization.

3. **Transformer Encoders**:
   - Two separate encoders (one for photos, one for faces) aggregate embeddings.
   - Input: CLS token + sequence of embeddings [batch, 10, 512] (CLS + up to 9 photos/faces).
   - Uses self-attention (4 layers, 8 heads) with padding masks to ignore invalid inputs.
   - Output: Aggregated vector from the encoded CLS token [batch, 512].
   - **Technical Choice**: Transformers treat inputs as sets, enabling context-aware aggregation. No positional encodings ensure permutation invariance—a key "hidden gem" where shuffling photos (with masks) yields identical outputs.
   - **Peculiarity**: Masks handle variable lengths (e.g., 3 photos + 6 faces); attention ignores padding, focusing on relevant data.

4. **MLP Head**:
   - Concatenates photo and face vectors [batch, 1024] and regresses to 9-class logits via a 2-layer MLP (1024 → 512 → 9).
   - **Technical Choice**: Simple non-linear fusion; dropout (0.2) for regularization.

**Data Flow Summary**:

- Inputs: `photos_tensor` [B,9,3,224,224], `photos_mask` [B,9], `faces_tensor` [B,9,3,224,224], `faces_mask` [B,9].
- Flatten & Extract: Per-image features via `MobileNetV3` + pooling → [B,9,960].
- Project: → [B,9,512].
- Aggregate: Prepend CLS, encode with transformer (masked) → [B,512] per branch.
- Fuse & Predict: Concat → MLP → [B,9] logits.

The model is permutation-invariant because transformers use attention (no order bias) and masks ensure only valid inputs matter.

## Peculiarities and Hidden Gems

- **Variable Inputs**: Up to 9 photos/faces; fewer are padded with zeros and masked (False in mask). This allows flexibility without fixed-size assumptions.
- **Permutation Invariance**: No positional encodings in transformers—photos can be shuffled without changing outputs. Tested via randomization: outputs match in eval mode (dropout disabled).
- **Separate Photo/Face Processing**: Enables different contributions (e.g., photos for composition, faces for details), even if counts differ.
- **Classification Over Regression**: Switched to classification (scores 1-9 → classes 0-8) for better handling discrete scores; uses cross-entropy loss.
- **Class Imbalance Handling**: Computes inverse-frequency weights; caps to avoid instability.
- **Fine-Tuning Strategy**: Unfreeze later `MobileNetV3` layers; use AdamW, cosine annealing scheduler, label smoothing, mixed precision, and early stopping for SOTA performance on expanded datasets.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/alessiosavi/BeautyScorer.git
   cd BeautyScorer
   ```

2. Install dependencies:

   ```bash
   pip install torch torchvision pandas tqdm scikit-learn
   ```

## Usage

### Preprocessing

Use `load_data` to load and normalize images.

### Inference

```python
CONF = utils.load_conf("conf.yaml")[0]
model = BeautyScoreModel(conf = CONF)
model.load_state_dict(torch.load('model_v8_classes_big_finetuned_state_dict.pt', weights_only=True))

score, probability = score_person(person_id, model)
print(
    f"Person: {person_id} -> Score: {score} | Prob: {probability} | RealScore: {batch['scores'][idx]}"
)
utils.show_person(person_id, 1)
```

### Training/Fine-Tuning

See `train()` function for full script. Example:

```python
CONF = utils.load_conf("conf.yaml")[0]
new_data = [{"id": "person1", "score": 5, "photos": glob("path/to/person1/*")}, ...]
raw_ds = dataset.BeautyDataset(raw_dataset)
raw_dl = DataLoader(
    raw_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    collate_fn=raw_ds.collate_fn,
)
model = BeautyScoreModel(conf = CONF)
model_params = list(filter(lambda p: p.requires_grad, model.parameters()))

class_weights = utils.compute_class_weights(df)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
optimizer = optim.AdamW(model_params, lr=lr, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
train(model, raw_dl)
```

## Training Details

- **Dataset**: List of dicts with ID, score (1-9), and photo paths.
- **Loss**: CrossEntropyLoss with weights and label smoothing (0.1).
- **Optimizer**: AdamW (lr=1e-5, weight_decay=0.01).
- **Scheduler**: CosineAnnealingLR.
- **Other**: Mixed precision (AMP), early stopping (patience=3).

## Acknowledgments

- Built with PyTorch and torchvision.
- Inspired by transformer-based set aggregation (e.g., BERT-like CLS token).

For issues, open a GitHub issue. Contributions welcome!
