"""ONNX Model Exporter for YOLO Detector."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
    import onnx
except ImportError as e:
    raise ImportError(f"Required export package missing: {e}")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("ExportONNX")


def export_to_onnx(
    model_path: str = "models/weights/best.pt",
    output_dir: str = "models/onnx",
    imgsz: int = 640,
    opset: int = 12,
    dynamic: bool = False,
    simplify: bool = True,
) -> Path:
    """Export PyTorch YOLO model to ONNX format."""
    m_path = Path(model_path)
    if not m_path.exists():
        logger.warning(f"Weights file '{model_path}' not found. Using pretrained 'yolov8n.pt' for export demonstration.")
        m_path = Path("yolov8n.pt")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model '{m_path}' for ONNX export (imgsz={imgsz}, opset={opset}, dynamic={dynamic})...")
    model = YOLO(str(m_path))

    # Ultralytics export
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )

    dest_path = out_root / f"{m_path.stem}.onnx"
    shutil.copy(exported_file, dest_path)

    # Validate ONNX graph
    onnx_model = onnx.load(str(dest_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = dest_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX export successful!\n  Path: {dest_path.resolve()}\n  Size: {file_size_mb:.2f} MB")
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", type=str, default="models/weights/best.pt", help="Path to .pt model")
    parser.add_argument("--output_dir", type=str, default="models/onnx", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
    )
