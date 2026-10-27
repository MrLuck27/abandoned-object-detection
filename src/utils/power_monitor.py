"""Hardware and Power Consumption Monitor for Edge/Workstation deployment."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Any, Optional
import psutil

# Check if NVML is directly available
try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


class PowerMonitor:
    """Monitors CPU, Memory, GPU Utilization and Power Draw."""

    def __init__(self, gpu_device_id: int = 0):
        self.gpu_device_id = gpu_device_id
        self.nvml_initialized = False
        self.nvml_handle = None
        self._power_samples = []

        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device_id)
                self.nvml_initialized = True
            except Exception:
                self.nvml_initialized = False

    def query_gpu_smi(self) -> Dict[str, Any]:
        """Fallback to nvidia-smi CLI if direct NVML bindings fail."""
        if not shutil.which("nvidia-smi"):
            return {
                "gpu_available": False,
                "power_w": None,
                "gpu_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_temp_c": None,
            }

        try:
            cmd = [
                "nvidia-smi",
                f"--id={self.gpu_device_id}",
                "--query-gpu=power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(cmd, encoding="utf-8", timeout=1.0).strip()
            parts = [p.strip() for p in output.split(",")]
            pwr = float(parts[0]) if parts[0] != "[N/A]" else None
            util = float(parts[1]) if parts[1] != "[N/A]" else None
            mem_u = float(parts[2]) if parts[2] != "[N/A]" else None
            mem_t = float(parts[3]) if parts[3] != "[N/A]" else None
            temp = float(parts[4]) if parts[4] != "[N/A]" else None

            if pwr is not None:
                self._power_samples.append(pwr)

            return {
                "gpu_available": True,
                "power_w": pwr,
                "gpu_util_percent": util,
                "gpu_mem_used_mb": mem_u,
                "gpu_mem_total_mb": mem_t,
                "gpu_temp_c": temp,
            }
        except Exception:
            return {
                "gpu_available": False,
                "power_w": None,
                "gpu_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_temp_c": None,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieve current CPU, RAM, and GPU stats.

        Returns:
            Dictionary with hardware status metrics.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        metrics: Dict[str, Any] = {
            "cpu_util_percent": round(cpu_percent, 1),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_percent": round(mem.percent, 1),
            "gpu_available": False,
            "power_w": None,
            "gpu_util_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_temp_c": None,
        }

        if self.nvml_initialized and self.nvml_handle:
            try:
                # NVML power is in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                power_w = round(power_mw / 1000.0, 2)
                self._power_samples.append(power_w)

                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)

                metrics.update({
                    "gpu_available": True,
                    "power_w": power_w,
                    "gpu_util_percent": util.gpu,
                    "gpu_mem_used_mb": round(mem_info.used / (1024**2), 1),
                    "gpu_mem_total_mb": round(mem_info.total / (1024**2), 1),
                    "gpu_temp_c": temp,
                })
                return metrics
            except Exception:
                pass

        # Fallback to nvidia-smi CLI
        gpu_smi = self.query_gpu_smi()
        metrics.update(gpu_smi)
        return metrics

    def get_power_summary(self, total_frames: int, elapsed_seconds: float) -> Dict[str, Any]:
        """Compute average/peak power and energy efficiency (FPS/Watt)."""
        fps = round(total_frames / max(elapsed_seconds, 1e-4), 2)
        if not self._power_samples:
            return {
                "avg_power_w": "Not Available",
                "peak_power_w": "Not Available",
                "fps": fps,
                "fps_per_watt": "Not Available",
            }

        avg_pwr = sum(self._power_samples) / len(self._power_samples)
        peak_pwr = max(self._power_samples)
        fps_per_watt = (fps / avg_pwr) if avg_pwr > 0 else 0.0

        return {
            "avg_power_w": round(avg_pwr, 2),
            "peak_power_w": round(peak_pwr, 2),
            "fps": fps,
            "fps_per_watt": round(fps_per_watt, 3),
        }

    def close(self):
        """Release NVML bindings."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
