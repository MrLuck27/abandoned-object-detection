"""Video I/O and stream helper functions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Tuple, Optional
import cv2
import numpy as np


class VideoReader:
    """Wrapper around OpenCV VideoCapture providing safe iteration and metadata."""

    def __init__(self, source: str | int | Path):
        self.source = str(source) if isinstance(source, Path) else source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video stream/file: {self.source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0  # Default nominal FPS
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            timestamp = frame_idx / self.fps
            yield frame_idx, timestamp, frame
            frame_idx += 1

    def release(self):
        """Release underlying OpenCV capture."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Wrapper around OpenCV VideoWriter."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to initialize VideoWriter at: {self.output_path}")

    def write(self, frame: np.ndarray):
        """Write frame to output video."""
        self.writer.write(frame)

    def release(self):
        """Finalize and release video file."""
        if self.writer.isOpened():
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
