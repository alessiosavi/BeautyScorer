# Architecture Guide

Deep dive into BeautyScorer's model architectures and design decisions.

## Overview

BeautyScorer uses a dual-stream architecture that processes photos and face crops separately before fusing them for final prediction. This design captures both overall appearance (from full photos) and facial features (from face crops).

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         BeautyScorer Architecture                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Photos (N×3×H×W)              Faces (N×3×H×W)                       │
│        │                              │                              │
│        ▼                              ▼                              │
│  ┌──────────────┐              ┌──────────────┐                      │
│  │   Backbone   │              │   Backbone   │  (shared or separate)│
│  │ (MobileNet/  │              │ (MobileNet/  │                      │
│  │  ViT/etc.)   │              │  ViT/etc.)   │                      │
│  └──────────────┘              └──────────────┘                      │
│        │                              │                              │
│        ▼                              ▼                              │
│  ┌──────────────┐              ┌──────────────┐                      │
│  │  Projection  │              │  Projection  │                      │
│  │   (Linear)   │              │   (Linear)   │                      │
│  └──────────────┘              └──────────────┘                      │
│        │                              │                              │
│        ▼                              ▼                              │
│  ┌──────────────┐              ┌──────────────┐                      │
│  │  Transformer │ ◄──────────► │  Transformer │  (cross-attention)   │
│  │   Encoder    │              │   Encoder    │                      │
│  └──────────────┘              └──────────────┘                      │
│        │                              │                              │
│        └──────────┬──────────────────┘                               │
│                   │                                                  │
│                   ▼                                                  │
│           ┌──────────────┐                                           │
│           │  SE Attention│                                           │
│           │   Concat +   │                                           │
│           └──────────────┘                                           │
│                   │                                                  │
│                   ▼                                                  │
│           ┌──────────────┐                                           │
│           │     MLP      │                                           │
│           │    Head      │                                           │
│           └──────────────┘                                           │
│                   │                                                  │
│                   ▼                                                  │
│              Logits (9)                                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Permutation Invariance

The model treats input photos as an **unordered set**. This is achieved by:

- **No positional encodings** in transformer encoders (by default)
- **CLS token aggregation** that attends equally to all positions
- **Masked attention** that ignores padded positions

```python
# Verification: shuffling photos doesn't change output
model.eval()
perm = torch.randperm(max_photos)
output1 = model(photos, mask, faces, face_mask)
output2 = model(photos[:, perm], mask[:, perm], faces[:, perm], face_mask[:, perm])
assert torch.allclose(output1, output2)  # True!
```

### 2. Variable-Length Inputs

Handles 1-N photos per person using:

- **Padding**: Fixed-size tensors with zero-padding
- **Boolean masks**: `True` for valid positions, `False` for padding
- **Masked attention**: Transformer ignores padded positions

### 3. Dual-Stream Processing

Separate processing for photos and faces allows:

- **Different contributions**: Full photos for composition, faces for details
- **Different counts**: Can have 5 photos but only 3 detected faces
- **Specialized features**: Each stream learns task-relevant representations

## Model Architectures

### 1. MobileNet + Transformer (Default)

**Best for**: General use cases, balanced accuracy/speed trade-off.

```text
Components:
├── Backbone: MobileNetV3-Large (960-dim features)
├── Projection: Linear(960 → 512)
├── Encoders: 4-layer Transformer (8 heads, 2048 FF dim)
├── Cross-Attention: Optional bidirectional attention
└── Head: SE-Attention + 2-layer MLP
```

**Architecture Details**:

```python
class MobileNetTransformerModel:
    def __init__(self, config):
        # Backbone (frozen or partially unfrozen)
        self.backbone = mobilenet_v3_large(pretrained=True).features
        # Output: [B, N, 960] after pooling

        # Project to embedding dimension
        self.photo_proj = nn.Linear(960, 512)
        self.face_proj = nn.Linear(960, 512)

        # Transformer encoders with CLS token
        self.photo_encoder = TransformerEncoder(
            embed_dim=512,
            num_layers=4,
            num_heads=8,
            use_cls_token=True
        )

        # Cross-attention (optional)
        self.cross_attention = CrossAttentionFusion(512, 8)

        # Classification head with SE attention
        self.head = SEAttentionHead(1024, 512, 9)
```

**Why MobileNetV3?**

- Efficient depthwise separable convolutions
- Pre-trained on ImageNet (strong general features)
- Good accuracy/speed trade-off
- SE blocks provide channel attention

### 2. ViT + ArcFace (Advanced)

**Best for**: Maximum accuracy when compute is available.

```text
Components:
├── Backbone: DINOv2 ViT-S/14 (384-dim) or ViT-B/16 (768-dim)
├── Projection: Linear(768 → 768) with LayerNorm
├── Self-Attention: 6-layer Transformer
├── Cross-Attention: Bidirectional photo-face attention
├── Pooling: Attention pooling (not CLS token)
└── Head: SE-Attention + 2-layer MLP
```

**Why DINOv2/ViT?**

- Self-supervised pre-training captures rich semantics
- Attention-based: naturally handles variable-length inputs
- Multi-scale features from different layers
- State-of-the-art transfer learning performance

**Why Attention Pooling?**

```python
class AttentionPooling(nn.Module):
    """Learns to weight different photos based on importance."""
    def __init__(self, embed_dim, num_heads):
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, mask):
        # Learned query attends to all photos
        output, weights = self.attention(
            self.query.expand(batch_size, -1, -1),
            x, x,
            key_padding_mask=~mask
        )
        return output.squeeze(1)  # [B, embed_dim]
```

### 3. Lightweight CPU (Efficient)

**Best for**: CPU inference, edge deployment, resource-constrained environments.

```text
Components:
├── Backbone: MobileNetV2 (no SE blocks) or EfficientNet-Lite
├── Projection: Linear(1280 → 256) with ReLU
├── Aggregation: Mean-Max pooling (no transformer)
├── Optional: Lightweight single-head attention
└── Head: Simple 2-layer MLP
```

**Design Decisions for CPU**:

1. **No SE blocks**: SE attention is slow on CPU
2. **Simple pooling**: Transformers have quadratic complexity
3. **Smaller embeddings**: 256 vs 512 reduces compute
4. **Frozen backbone**: Fewer gradients to compute
5. **No AMP**: Mixed precision doesn't help on CPU

```python
class LightweightPooling(nn.Module):
    """Fast pooling for CPU efficiency."""
    def forward(self, x, mask):
        # Masked mean pooling
        x_masked = x * mask.unsqueeze(-1)
        mean_pool = x_masked.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        # Max pooling
        x_masked[~mask] = float('-inf')
        max_pool = x_masked.max(1)[0]

        # Learned combination
        return self.weight[0] * mean_pool + self.weight[1] * max_pool
```

## Component Deep Dives

### Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout):
        # Learnable CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Pre-norm transformer layers (better training stability)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True  # Pre-LayerNorm
            ),
            num_layers=num_layers
        )

    def forward(self, x, mask):
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Create padding mask (True = ignore)
        padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=x.device),
            ~mask  # Invert: True means valid in input, True means ignore in attention
        ], dim=1)

        # Encode
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Return CLS token representation
        return x[:, 0]
```

### Cross-Attention Fusion

Allows photo and face streams to exchange information:

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        # Photo attends to faces
        self.photo_to_face = nn.MultiheadAttention(embed_dim, num_heads)
        # Face attends to photos
        self.face_to_photo = nn.MultiheadAttention(embed_dim, num_heads)
        # Feed-forward layers
        self.ff_photo = FeedForward(embed_dim)
        self.ff_face = FeedForward(embed_dim)

    def forward(self, photo_emb, face_emb, photo_mask, face_mask):
        # Photo attends to faces
        photo_enhanced, _ = self.photo_to_face(
            query=photo_emb,
            key=face_emb,
            value=face_emb,
            key_padding_mask=~face_mask
        )
        photo_emb = self.ff_photo(photo_emb + photo_enhanced)

        # Face attends to photos
        face_enhanced, _ = self.face_to_photo(
            query=face_emb,
            key=photo_emb,
            value=photo_emb,
            key_padding_mask=~photo_mask
        )
        face_emb = self.ff_face(face_emb + face_enhanced)

        return photo_emb, face_emb
```

### SE-Attention Head

Squeeze-and-Excitation for adaptive feature recalibration:

```python
class SEAttentionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, reduction=16):
        # Channel attention
        self.se = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim),
            nn.Sigmoid()
        )
        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Apply channel attention
        attention = self.se(x)
        x = x * attention
        return self.mlp(x)
```

## Backbone Registry

Supports multiple backbone architectures:

| Backbone | Family | Feature Dim | Best For |
|----------|--------|-------------|----------|
| `mobilenet_v3_large` | MobileNet | 960 | Default, balanced |
| `mobilenet_v3_small` | MobileNet | 576 | Faster, less accurate |
| `mobilenet_v2` | MobileNet | 1280 | CPU-friendly |
| `efficientnet_b0` | EfficientNet | 1280 | Efficient, accurate |
| `efficientnet_lite0` | EfficientNet | 1280 | CPU/mobile |
| `resnet50` | ResNet | 2048 | Classic, robust |
| `vit_b_16` | ViT | 768 | High accuracy |
| `dinov2_vits14` | DINOv2 | 384 | SOTA features |
| `dinov2_vitb14` | DINOv2 | 768 | SOTA, larger |

## Memory Optimization

### Gradient Checkpointing

Trades compute for memory:

```python
config = ModelConfig(use_gradient_checkpointing=True)
model = create_model(config=config)

# During forward pass, intermediate activations are not stored
# They're recomputed during backward pass
```

### Mixed Precision Training

Uses FP16 for most operations:

```python
scaler = torch.amp.GradScaler()

with torch.amp.autocast(device_type="cuda"):
    logits = model(photos, masks, faces, face_masks)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

Simulates larger batches:

```python
# Effective batch size = batch_size × accumulation_steps
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Data Flow Example

```python
# Input shapes for batch_size=4, max_photos=9
photos = torch.randn(4, 9, 3, 224, 224)      # 4 people, up to 9 photos each
photos_mask = torch.ones(4, 9, dtype=bool)    # All valid (could have False for padding)
faces = torch.randn(4, 9, 3, 224, 224)        # Extracted faces
faces_mask = torch.tensor([                    # Some faces might not be detected
    [1, 1, 1, 1, 0, 0, 0, 0, 0],  # Person 1: 4 faces detected
    [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Person 2: all 9 faces detected
    [1, 1, 0, 0, 0, 0, 0, 0, 0],  # Person 3: only 2 faces
    [1, 1, 1, 1, 1, 0, 0, 0, 0],  # Person 4: 5 faces
], dtype=bool)

# Forward pass
logits = model(photos, photos_mask, faces, faces_mask)
# Output: [4, 9] - logits for 9 classes for each person

# Prediction
probs = torch.softmax(logits, dim=-1)
scores = logits.argmax(dim=-1) + 1  # 1-indexed scores
```

## Performance Considerations

| Model | Params | GPU (V100) | CPU (i7) | Memory |
|-------|--------|------------|----------|--------|
| MobileNet+Transformer | ~15M | 50 ms | 200 ms | 2 GB |
| ViT+ArcFace | ~90M | 150 ms | 2000 ms | 8 GB |
| Lightweight CPU | ~5M | 20 ms | 80 ms | 1 GB |

*Inference time per batch of 4 samples with 9 photos each*
