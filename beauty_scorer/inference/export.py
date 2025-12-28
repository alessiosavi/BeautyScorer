"""
Model export utilities for deployment.

Supports exporting to:
- ONNX for cross-platform inference
- TorchScript for deployment
- Quantized INT8 for edge devices
"""

from pathlib import Path

import torch
import torch.nn as nn

from beauty_scorer.models.base import BaseBeautyModel
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model: BaseBeautyModel,
    output_path: str | Path,
    max_photos: int = 9,
    image_size: tuple[int, int] = (224, 224),
    face_size: tuple[int, int] = (224, 224),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """
    Export model to ONNX format.

    Args:
        model: Model to export.
        output_path: Path for output ONNX file.
        max_photos: Maximum number of photos.
        image_size: Input image size (H, W).
        face_size: Face input size (H, W).
        opset_version: ONNX opset version.
        dynamic_batch: Whether to allow dynamic batch size.

    Returns:
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    # Create dummy inputs
    batch_size = 1
    h, w = image_size
    fh, fw = face_size

    dummy_photos = torch.randn(batch_size, max_photos, 3, h, w)
    dummy_photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    dummy_faces = torch.randn(batch_size, max_photos, 3, fh, fw)
    dummy_faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

    # Input/output names
    input_names = ["photos", "photos_mask", "faces", "faces_mask"]
    output_names = ["logits"]

    # Dynamic axes
    if dynamic_batch:
        dynamic_axes = {
            "photos": {0: "batch_size"},
            "photos_mask": {0: "batch_size"},
            "faces": {0: "batch_size"},
            "faces_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }
    else:
        dynamic_axes = None

    # Export
    torch.onnx.export(
        model,
        (dummy_photos, dummy_photos_mask, dummy_faces, dummy_faces_mask),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"Exported model to ONNX: {output_path}")

    # Verify ONNX model
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")
    except ImportError:
        logger.warning("onnx package not installed, skipping verification")
    except Exception as e:
        logger.warning(f"ONNX verification failed: {e}")

    return str(output_path)


def export_to_torchscript(
    model: BaseBeautyModel,
    output_path: str | Path,
    max_photos: int = 9,
    image_size: tuple[int, int] = (224, 224),
    face_size: tuple[int, int] = (224, 224),
    method: str = "trace",
) -> str:
    """
    Export model to TorchScript format.

    Args:
        model: Model to export.
        output_path: Path for output file.
        max_photos: Maximum number of photos.
        image_size: Input image size.
        face_size: Face input size.
        method: Export method ('trace' or 'script').

    Returns:
        Path to exported TorchScript file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    # Create dummy inputs
    batch_size = 1
    h, w = image_size
    fh, fw = face_size

    dummy_photos = torch.randn(batch_size, max_photos, 3, h, w)
    dummy_photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    dummy_faces = torch.randn(batch_size, max_photos, 3, fh, fw)
    dummy_faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

    if method == "trace":
        scripted = torch.jit.trace(
            model,
            (dummy_photos, dummy_photos_mask, dummy_faces, dummy_faces_mask),
        )
    else:
        scripted = torch.jit.script(model)

    # Save
    scripted.save(str(output_path))
    logger.info(f"Exported model to TorchScript: {output_path}")

    return str(output_path)


def export_to_quantized(
    model: BaseBeautyModel,
    output_path: str | Path,
    calibration_data: list | None = None,
    backend: str = "fbgemm",
) -> str:
    """
    Export quantized INT8 model.

    Args:
        model: Model to export.
        output_path: Path for output file.
        calibration_data: Data for calibration (list of input tuples).
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM).

    Returns:
        Path to exported quantized model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    # Set quantization config
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    # Prepare model for quantization
    model_prepared = torch.ao.quantization.prepare(model, inplace=False)

    # Calibrate with data if provided
    if calibration_data:
        logger.info(f"Calibrating with {len(calibration_data)} samples...")
        with torch.no_grad():
            for photos, photos_mask, faces, faces_mask in calibration_data:
                model_prepared(photos, photos_mask, faces, faces_mask)

    # Convert to quantized
    model_quantized = torch.ao.quantization.convert(model_prepared, inplace=False)

    # Save as TorchScript
    scripted = torch.jit.script(model_quantized)
    scripted.save(str(output_path))

    logger.info(f"Exported quantized model: {output_path}")
    return str(output_path)


class ONNXPredictor:
    """
    Predictor using ONNX Runtime for inference.

    Useful for deployment without PyTorch dependency.
    """

    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
    ):
        """
        Initialize ONNX predictor.

        Args:
            model_path: Path to ONNX model.
            providers: Execution providers (e.g., ['CUDAExecutionProvider']).
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(f"ONNX predictor initialized: {model_path}")

    def predict(
        self,
        photos: "np.ndarray",
        photos_mask: "np.ndarray",
        faces: "np.ndarray",
        faces_mask: "np.ndarray",
    ) -> "np.ndarray":
        """
        Run inference.

        Args:
            photos: Photo tensor as numpy array.
            photos_mask: Photo mask as numpy array.
            faces: Face tensor as numpy array.
            faces_mask: Face mask as numpy array.

        Returns:
            Logits as numpy array.
        """
        import numpy as np

        # Prepare inputs
        inputs = {
            "photos": photos.astype(np.float32),
            "photos_mask": photos_mask.astype(bool),
            "faces": faces.astype(np.float32),
            "faces_mask": faces_mask.astype(bool),
        }

        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]


def get_model_size(model: nn.Module) -> dict:
    """
    Get model size information.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with size information.
    """
    import os
    import tempfile

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    # Get saved model size
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        file_size = os.path.getsize(f.name)
        os.unlink(f.name)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_mb": param_size / (1024 * 1024),
        "buffer_memory_mb": buffer_size / (1024 * 1024),
        "total_memory_mb": (param_size + buffer_size) / (1024 * 1024),
        "saved_size_mb": file_size / (1024 * 1024),
    }
