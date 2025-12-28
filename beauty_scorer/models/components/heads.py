"""
Classification and regression heads for beauty scoring.

Provides various head architectures including standard classification,
squeeze-and-excitation attention, and regression heads.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Standard classification head with MLP.

    Takes concatenated features and produces class logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 9,
        dropout: float = 0.2,
    ):
        """
        Initialize classification head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Number of output classes.
            dropout: Dropout rate.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim).

        Returns:
            Class logits (batch, num_classes).
        """
        return self.mlp(x)


class SEAttentionHead(nn.Module):
    """
    Classification head with Squeeze-and-Excitation attention.

    Adds channel-wise attention before classification for
    adaptive feature recalibration.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 9,
        dropout: float = 0.2,
        reduction: int = 16,
    ):
        """
        Initialize SE attention head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            reduction: SE reduction ratio.
        """
        super().__init__()

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim),
            nn.Sigmoid(),
        )

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SE attention.

        Args:
            x: Input features (batch, input_dim).

        Returns:
            Class logits (batch, num_classes).
        """
        # Apply SE attention
        attention = self.se(x)
        x = x * attention

        # Classification
        return self.mlp(x)


class RegressionHead(nn.Module):
    """
    Regression head for continuous score prediction.

    Predicts a single continuous value instead of class probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        min_score: float = 1.0,
        max_score: float = 9.0,
    ):
        """
        Initialize regression head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
            min_score: Minimum output score.
            max_score: Maximum output score.
        """
        super().__init__()
        self.min_score = min_score
        self.max_score = max_score

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim).

        Returns:
            Predicted scores (batch, 1) in [min_score, max_score].
        """
        out = self.mlp(x)
        # Scale to score range using sigmoid
        out = torch.sigmoid(out) * (self.max_score - self.min_score) + self.min_score
        return out


class OrdinalHead(nn.Module):
    """
    Ordinal regression head for ordered class prediction.

    Uses multiple binary classifiers for ordinal regression,
    which is suitable for ordered categories like beauty scores.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 9,
        dropout: float = 0.2,
    ):
        """
        Initialize ordinal head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Number of ordinal classes.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_classes = num_classes

        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Binary classifiers for each threshold
        # For K classes, we need K-1 thresholds
        self.thresholds = nn.Linear(hidden_dim // 2, num_classes - 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim).

        Returns:
            Ordinal logits (batch, num_classes - 1) for each threshold.
        """
        features = self.features(x)
        logits = self.thresholds(features)
        return logits

    def predict_class(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert ordinal logits to class predictions.

        Args:
            logits: Ordinal logits (batch, num_classes - 1).

        Returns:
            Predicted classes (batch,) in [0, num_classes - 1].
        """
        # Cumulative probabilities
        cum_probs = torch.sigmoid(logits)
        # Count how many thresholds are exceeded
        predictions = (cum_probs > 0.5).sum(dim=1)
        return predictions

    def to_class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert ordinal logits to class probabilities.

        Args:
            logits: Ordinal logits (batch, num_classes - 1).

        Returns:
            Class probabilities (batch, num_classes).
        """
        batch_size = logits.size(0)
        device = logits.device

        # Cumulative probabilities
        cum_probs = torch.sigmoid(logits)

        # Add boundaries (P(Y >= 0) = 1, P(Y >= K) = 0)
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)
        cum_probs = torch.cat([ones, cum_probs, zeros], dim=1)

        # Class probabilities = P(Y >= k) - P(Y >= k+1)
        class_probs = cum_probs[:, :-1] - cum_probs[:, 1:]

        return class_probs


class MultiTaskHead(nn.Module):
    """
    Multi-task head for joint classification and regression.

    Predicts both class probabilities and continuous scores.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 9,
        dropout: float = 0.2,
    ):
        """
        Initialize multi-task head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Number of output classes.
            dropout: Dropout rate.
        """
        super().__init__()

        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification branch
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Regression branch
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim).

        Returns:
            Tuple of (class_logits, regression_scores).
        """
        shared_features = self.shared(x)
        class_logits = self.classifier(shared_features)
        reg_scores = self.regressor(shared_features)
        return class_logits, reg_scores
