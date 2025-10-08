# Architecture Overview

The `BeautyScore` is a PyTorch neural network designed to predict a beauty score (as a class output) based on up to 9 photos and up to 9 corresponding face crops per sample in a batch. It processes photos and faces separately using a shared feature extractor (`MobileNetV3`), then aggregates features using transformer encoders with attention mechanisms to create order-invariant representations (i.e., the order of photos/faces doesn't affect the output). Masks handle variable numbers of photos/faces by ignoring padding. The final score is produced by concatenating aggregated vectors and passing them through an MLP.

## Key components

- **Feature Extractor**: Pre-trained `MobileNetV3` (features only, no classifier head). It extracts spatial features from images, outputting a tensor of shape [batch_size * 9, 960, 7, 7] for the flattened batch.
- **Projection Layers**: Linear layers (`photo_proj` and `face_proj`) to map the extracted features (after pooling and flattening) to an embedding dimension (default: 512).
- **CLS Tokens**: Learnable tokens prepended to the sequence of photo/face embeddings, acting as aggregates for the transformer.
- **Transformer Encoders**: Two separate `nn.TransformerEncoder` instances (one for photos, one for faces), each with 4 layers, 8 attention heads, and feedforward dim 2048. They use self-attention to combine embeddings, ignoring masked positions, ensuring permutation invariance (no positional encodings are added, so order doesn't matter).
- **MLP Head**: A simple fully connected network to classify the final score from the concatenated aggregated vectors.
- **Masks**: Boolean masks (`photos_mask`, `faces_mask`) ensure attention only considers valid (non-padded) photos/faces.

The model is permutation-invariant because:

- Transformers rely on self-attention, which treats inputs as sets (no inherent order).
- No positional encodings are used.
- Masks prevent padding from influencing attention.
- Shuffling photos (with corresponding mask shuffle) should yield the same output in eval mode (as confirmed in prior debugging).

Hyperparameters like `embed_dim=512`, `num_encoder_layers=4`, `num_heads=8`, `ff_dim=2048`, `dropout=0.2` can be tuned but are set to reasonable defaults.

## Logic and Input Propagation Step by Step

The model takes four inputs in the `forward` method:

- `photos_tensor`: [batch_size, 9, 3, 224, 224] – Batch of photos (padded with zeros if fewer than 9).
- `photos_mask`: [batch_size, 9] – Boolean mask (True for valid photos, False for padding).
- `faces_tensor`: [batch_size, 9, 3, 224, 224] – Similar for faces.
- `faces_mask`: [batch_size, 9] – Mask for faces.

Propagation happens symmetrically for photos and faces, then combines at the end. I'll describe photos first, then note faces are identical, and finally the combination.

1. **Flatten Batch for Feature Extraction (Photos)**:
   - `photos_flat = photos_tensor.view(-1, 3, 224, 224)`: Reshapes to [batch_size * 9, 3, 224, 224]. This flattens the sequence dimension so all images (valid and padded) can be processed in parallel by the CNN.
   - Logic: Treats the batch as a large set of individual images for efficient GPU processing.

2. **Extract Features with `MobileNetV3` (Photos)**:
   - `photo_feats_flat = self.feature_extractor(photos_flat)`: Passes through `MobileNetV3`'s convolutional layers, outputting [batch_size * 9, 960, 7, 7].
   - Logic: `MobileNetV3` is a lightweight CNN that extracts hierarchical features (e.g., edges, textures, objects) from RGB images. The output is a feature map (spatial grid of 7x7 with 960 channels). Padded images (all zeros) will produce near-zero feature maps, but masks handle this later.

3. **Pool and Flatten Features (Photos)**:
   - `photo_feats_pooled = self.pool(photo_feats_flat).view(-1, self.feature_dim)`: Applies adaptive average pooling to reduce [960, 7, 7] to [960, 1, 1], then flattens to [batch_size * 9, 960].
   - Logic: Pooling summarizes the spatial features into a fixed-size vector per image, capturing global information (e.g., overall composition) while discarding fine spatial details. This is standard for image classification/embedding tasks to get a compact representation.

4. **Reshape and Project to Embedding Space (Photos)**:
   - `photo_feats = photo_feats_pooled.view(batch_size, 9, self.feature_dim)`: Reshapes back to [batch_size, 9, 960].
   - `photo_emb = self.photo_proj(photo_feats)`: Linear projection to [batch_size, 9, embed_dim] (e.g., 512).
   - Logic: The projection aligns the CNN features to the transformer's input dimension, allowing for potential dimensionality reduction or adaptation.

5. **Prepend CLS Token (Photos)**:
   - `cls_photo = self.photo_cls.repeat(batch_size, 1, 1)`: Repeats the learnable CLS token to [batch_size, 1, embed_dim].
   - `photo_input = torch.cat([cls_photo, photo_emb], dim=1)`: Concatenates to [batch_size, 10, embed_dim] (CLS + 9 photos).
   - Logic: The CLS token is a special embedding that will aggregate information from all photos via attention. During training, it learns to represent the "summary" of the photo set.

6. **Create Padding Mask (Photos)**:
   - `photo_padding_mask = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=photos_mask.device), ~photos_mask], dim=1)`: [batch_size, 10], where False means "attend to this" (CLS is always attended, valid photos are attended based on mask, padding is ignored).
   - Logic: Transformer attention uses this mask to set attention weights to zero for padded positions, preventing them from influencing the output. `~photos_mask` flips the input mask (original True=valid becomes False=attend).

7. **Transformer Encoding (Photos)**:
   - `photo_input.transpose(0, 1)`: Transposes to [10, batch_size, embed_dim] (sequence-first for transformer).
   - `photo_encoded = self.photo_encoder(photo_input.transpose(0, 1), src_key_padding_mask=photo_padding_mask).transpose(0, 1)`: Outputs [batch_size, 10, embed_dim].
   - Logic: The transformer encoder applies multi-head self-attention across the sequence (CLS + photo embeddings). Each layer computes attention scores between all pairs (query-key dot products, scaled and softmaxed), masked to ignore padding. This allows the CLS token to "attend" to relevant photos, aggregating features in a weighted, context-aware way. Feedforward layers add non-linearity. Since no positional encoding is added, the model treats inputs as a bag-of-embeddings, ensuring order invariance. Dropout (0.2) adds regularization during training.

8. **Extract Aggregated Vector (Photos)**:
   - `photo_vec = photo_encoded[:, 0, :]` : Takes the encoded CLS token as [batch_size, embed_dim].
   - Logic: After encoding, the CLS position holds the pooled representation of all valid photos.

    **Faces Processing (Steps 1-8 Repeated)**:

    - Identical to photos, using `faces_tensor`, `faces_mask`, `self.face_proj`, `self.face_cls`, `self.face_encoder`.
    - Outputs `face_vec`: [batch_size, embed_dim].
    - Logic: Allows separate handling of full photos (e.g., composition, background) vs. face crops (e.g., facial beauty), even if counts differ (X photos, Y faces).

9. **Concatenate and Predict Score**:
   - `combined = torch.cat([photo_vec, face_vec], dim=1)`: [batch_size, 2 * embed_dim].
   - `score = self.mlp(combined)`: Passes through MLP ([2*embed_dim -> 512 -> ReLU -> Dropout -> N_CLASSES]), outputs [batch_size].
   - Logic: Concatenation fuses photo and face summaries. The MLP learns a non-linear mapping to a set of logits for each class, with dropout for regularization.

## Overall Flow Summary

- Inputs → CNN feature extraction (per image) → Pooling/Flattening → Projection → CLS + Sequence → Masked Transformer (set aggregation) → CLS extract → Concat (photos + faces) → MLP → Score.
- In training: Dropout causes output variability; use `.train()` for updates.
- In eval: Deterministic; use `.eval()` for inference.
- Handles variable lengths via masks; permutation-invariant due to attention-only processing.

### EXTRA

#### Why use a linear layer after the `MobileNetV3` feature extractor?

The `self.photo_proj` and `self.face_proj` linear layers (`nn.Linear(self.feature_dim, embed_dim)`) in the `BeautyScoreModel` are used to transform the pooled features from `MobileNetV3` (dimension 960) to a lower-dimensional embedding space (default `embed_dim=512`) before feeding them into the transformer encoder. While it might seem possible to skip these layers and directly input the pooled features (dimension 960) into the transformer, there are several important reasons for including these linear projections.

### Role of the Linear Projection Layers

1. **Dimensionality Reduction**:
   - **Context**: `MobileNetV3`’s feature extractor outputs a feature map of shape [batch_size *9, 960, 7, 7], which is pooled to [batch_size* 9, 960] via adaptive average pooling. This 960-dimensional vector is relatively high-dimensional.
   - **Purpose**: The linear layers (`photo_proj` and `face_proj`) map this to a smaller `embed_dim` (e.g., 512). This reduces the computational burden on the transformer encoder, which has quadratic complexity with respect to the embedding dimension (due to self-attention’s key-query dot products).
   - **Why Necessary**: Feeding 960-dimensional vectors directly into the transformer would increase memory and computation costs (e.g., attention matrices would be 960x960 per head instead of 512x512). Reducing to 512 dimensions makes the model more efficient, especially for larger batch sizes or sequences.

2. **Task-Specific Adaptation**:
   - **Context**: `MobileNetV3` is pre-trained on ImageNet, so its 960-dimensional features are optimized for general object recognition, not beauty scoring or face-specific tasks.
   - **Purpose**: The linear projection layers act as a learnable transformation to adapt these generic features to a representation better suited for the beauty scoring task. During training, the weights of `photo_proj` and `face_proj` are updated to emphasize features relevant to photos and faces in your dataset.
   - **Why Necessary**: Without projection, the transformer would receive raw ImageNet features, which may not capture nuances of beauty (e.g., facial symmetry, photo aesthetics). The linear layer allows the model to learn a task-specific subspace, improving performance. Feeding the 960-dimensional features directly will potentially cause numerical instability (e.g., larger dot products in attention).

3. **Improved Generalization**:
   - **Context**: High-dimensional inputs (960) can lead to overfitting, especially if your dataset is small or the model has many parameters.
   - **Purpose**: The projection to a lower dimension (512) acts as a bottleneck, forcing the model to learn a more compact, generalizable representation of the features.
   - **Why Necessary**: This bottleneck can reduce noise in the `MobileNetV3` features and prevent the transformer from overfitting to irrelevant dimensions, improving robustness.

4. **Separate Processing for Photos and Faces**:
   - **Context**: The model processes photos and faces separately (via `photo_encoder` and `face_encoder`), but both use the same `MobileNetV3` feature extractor, producing identical 960-dimensional outputs.
   - **Purpose**: Separate projection layers (`photo_proj` and `face_proj`) allow the model to learn distinct transformations for photos (e.g., full-body or scene aesthetics) and faces (e.g., facial features). This specialization is critical since ***photos and faces may contribute differently to the beauty score***.
   - **Why Necessary**: Without separate projections, the transformer would receive identical feature spaces for photos and faces, potentially losing the ability to differentiate their roles. The linear layers enable tailored embeddings for each input type.

### Why Not Feed Pooled Features Directly?

Feeding the pooled features (960 dimensions) directly to the transformer encoder without projection would be technically possible by setting `d_model=960` in the transformer. However, this would introduce several issues:

- **Increased Computational Cost**: The transformer’s self-attention scales as O(n²d) (n=sequence length, d=embedding dimension). For d=960 vs. 512, attention computations are ~6.25x more expensive (960²/512²). With a sequence length of 10 (CLS + 9 photos/faces), this significantly slows training/inference and ***increases memory usage***.
- **Poor Task Adaptation**: Raw `MobileNetV3` features are tuned for ImageNet’s 1000-class object recognition, not beauty scoring. Many of the 960 dimensions may be irrelevant (e.g., features for detecting cars or animals). Without projection, the transformer would need to learn to ignore these, which is harder than learning a compact representation via a linear layer.
- **Risk of Overfitting**: High-dimensional inputs increase the model’s capacity unnecessarily, risking overfitting, especially if your dataset isn’t large enough to constrain the transformer’s learning.
- **Numerical Stability**: Larger embedding dimensions can lead to larger dot products in attention, potentially causing instability (e.g., softmax saturation). Projection to a smaller dimension helps stabilize training.
- **Loss of Specialization**: Without separate `photo_proj` and `face_proj`, the model cannot easily differentiate photo and face contributions, reducing its ability to weigh their importance for the final score.

### When Might You Skip the Linear Layer?

In rare cases, you might consider skipping the projection if:

- Your dataset is extremely large, and computational cost isn’t a concern (unlikely for beauty scoring).
- The `MobileNetV3` features are already highly relevant to your task (e.g., after extensive fine-tuning on a similar dataset).
- You’re using a transformer designed for high-dimensional inputs (e.g., with sparse attention), but this isn’t the case for `nn.TransformerEncoder`.

Even then, you’d likely replace the linear layer with another form of dimensionality reduction (e.g., 1x1 convolution or PCA-like transformation) to maintain efficiency and task relevance.

#### TL;DR

The linear projection layers are necessary for:

- Reducing dimensionality for efficiency (mostly memory).
- Adapting ImageNet features to the beauty scoring task.
- Aligning with the transformer’s expected input size.
- Enabling separate processing of photos and faces.
- Improving generalization through a bottleneck.
