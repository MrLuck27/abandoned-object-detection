"""TensorRT Engine Builder for Jetson / Workstation Edge Deployment.

Converts an exported ONNX model to a TensorRT optimized engine (.engine).
Checks for TensorRT availability and provides actionable setup guidance
if the native NVIDIA TensorRT SDK is not installed on the host system.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("BuildTensorRT")


def check_tensorrt_available() -> Tuple[bool, str]:
    """Check if TensorRT Python API or trtexec CLI is installed."""
    # Check trtexec CLI
    if shutil.which("trtexec"):
        return True, "trtexec CLI found in system PATH"

    # Check python module
    try:
        import tensorrt as trt
        return True, f"Python tensorrt module found (v{trt.__version__})"
    except ImportError:
        pass

    return False, "TensorRT is not installed or not found in system PATH"


def build_tensorrt_engine(
    onnx_path: str = "models/onnx/yolov8n.onnx",
    output_engine: str = "models/tensorrt/yolov8n.engine",
    fp16: bool = True,
    device_id: int = 0,
):
    """Compile ONNX into TensorRT Engine."""
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        logger.error(f"Input ONNX file not found at: {onnx_file.resolve()}. Run `python scripts/export_onnx.py` first.")
        return False

    out_file = Path(output_engine)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    available, detail = check_tensorrt_available()
    logger.info(f"TensorRT Status: {detail}")

    if not available:
        logger.warning(
            "\n" + "=" * 70 + "\n"
            "TENSORRT COMPILATION PREREQUISITE NOTICE:\n"
            "TensorRT requires an NVIDIA GPU with compatible CUDA/cuDNN drivers and\n"
            "the NVIDIA TensorRT SDK installed.\n\n"
            "To install TensorRT on Windows / Linux:\n"
            "  1. Ensure CUDA Toolkit (11.8+ or 12.x) and cuDNN are installed.\n"
            "  2. Install TensorRT via pip for NVIDIA CUDA 12.x:\n"
            "       pip install --upgrade tensorrt tensorrt-cu12\n"
            "  3. Alternatively, download TensorRT GA from NVIDIA Developer:\n"
            "       https://developer.nvidia.com/tensorrt\n"
            "     and add the TensorRT /bin folder (containing trtexec.exe) to PATH.\n"
            "=" * 70 + "\n"
        )
        return False

    # If trtexec CLI is available
    if shutil.which("trtexec"):
        logger.info(f"Compiling with trtexec: {onnx_file} -> {out_file} (FP16={fp16})...")
        cmd = [
            "trtexec",
            f"--onnx={onnx_file.resolve()}",
            f"--saveEngine={out_file.resolve()}",
            f"--device={device_id}",
        ]
        if fp16:
            cmd.append("--fp16")

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully generated TensorRT engine: {out_file.resolve()}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"trtexec failed with error code: {e.returncode}")
            return False

    # Attempt Python TensorRT Builder
    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_file, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"Parser error: {parser.get_error(error)}")
                return False

        config = builder.create_builder_config()
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build serialized TensorRT engine.")
            return False

        with open(out_file, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"Successfully compiled and saved TensorRT engine to: {out_file.resolve()}")
        return True
    except Exception as e:
        logger.error(f"TensorRT Python build error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT Engine from ONNX")
    parser.add_argument("--onnx", type=str, default="models/onnx/best.onnx")
    parser.add_argument("--output", type=str, default="models/tensorrt/best.engine")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    args = parser.parse_args()

    build_tensorrt_engine(
        onnx_path=args.onnx,
        output_engine=args.output,
        fp16=args.fp16,
        device_id=args.device,
    )
