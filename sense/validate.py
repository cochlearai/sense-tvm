"""ONNX vs compiled binary validation utilities."""

from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Dict

import numpy as np

from sense import Sense


def _bench(fn, runs: int = 10, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    end = time.perf_counter()
    return (end - start) / runs


def _build_binary(model_dir: Path) -> bool:
    subprocess.run(
        ["make", "clean"],
        cwd=str(model_dir),
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["make"],
        cwd=str(model_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    return True


def _compare_outputs(onnx_output: np.ndarray, c_output: np.ndarray) -> Dict[str, Any]:
    onnx_flat = onnx_output.flatten()
    c_flat = c_output.flatten()

    abs_diff = np.abs(onnx_flat - c_flat)
    rel_diff = np.abs((onnx_flat - c_flat) / (np.abs(onnx_flat) + 1e-10))
    cosine_sim = float(np.dot(onnx_flat, c_flat) / (np.linalg.norm(onnx_flat) * np.linalg.norm(c_flat)))
    is_close = bool(np.allclose(onnx_flat, c_flat, rtol=1e-4, atol=1e-5))

    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "cosine_similarity": cosine_sim,
        "is_close": is_close,
        "shape_match": onnx_output.shape == c_output.shape,
    }


def _print_validation_result(title: str, result: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    status = "✓ PASS" if result["is_close"] else "❌ FAIL"
    print(f"  Status: {status}")
    print(f"  Max abs diff: {result['max_abs_diff']:.6e}")
    print(f"  Mean abs diff: {result['mean_abs_diff']:.6e}")
    print(f"  Cosine similarity: {result['cosine_similarity']:.8f}")
    print(f"  ONNX latency: {result['onnx_latency_ms']:.3f} ms")
    if result.get("c_latency_ms") is not None:
        print(f"  C latency: {result['c_latency_ms']:.3f} ms")
    else:
        print("  C latency: N/A (stdout parse failed)")
    if result.get("speedup_vs_onnx") is not None:
        print(f"  Speedup (ONNX/C): {result['speedup_vs_onnx']:.2f}x")
    print("=" * 60)


def _run_binary_validation(
    title: str,
    model_dir: Path,
    executable_name: str,
    backend: str,
    input_data: np.ndarray,
    ort_output: np.ndarray,
    ort_time_ms: float,
) -> Dict[str, Any] | None:
    print(f"[INFO] Rebuilding {title} binary...")
    if not _build_binary(model_dir):
        return None

    executable = (model_dir / executable_name).resolve()
    if not executable.exists():
        print(f"Error: executable not found at {executable}")
        return None

    c_latency_ms = None
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_out:
            try:
                input_data.tofile(tmp_in.name)
                tmp_in.flush()
                cmd = [str(executable), tmp_in.name, tmp_out.name]
                if backend == "ncnn":
                    cmd.extend(
                        [
                            str(int(np.prod(ort_output.shape))),
                            "weights.bin",
                        ]
                    )
                result = subprocess.run(
                    cmd,
                    cwd=str(model_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    print(f"Execution failed: {result.stderr}")
                    return None
                c_output = np.fromfile(tmp_out.name, dtype=np.float32).reshape(ort_output.shape)
                match = re.search(
                    r"Inference completed in\s+([0-9]*\.?[0-9]+)\s*ms",
                    result.stdout,
                )
                if match:
                    c_latency_ms = float(match.group(1))
            finally:
                Path(tmp_in.name).unlink(missing_ok=True)
                Path(tmp_out.name).unlink(missing_ok=True)

    metrics = _compare_outputs(ort_output, c_output)
    speedup = None
    if c_latency_ms and c_latency_ms > 0:
        speedup = ort_time_ms / c_latency_ms

    validation_result = {
        **metrics,
        "onnx_latency_ms": ort_time_ms,
        "c_latency_ms": c_latency_ms,
        "speedup_vs_onnx": speedup,
    }
    _print_validation_result(title, validation_result)
    return validation_result


def run_validation(sense: Sense, model_path: Path) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime is required for --validate.")
        return False

    if not sense.parse_result:
        print("Error: Sense pipeline did not complete. Run build before validation.")
        return False

    input_info = sense.parse_result.input_info
    if not input_info:
        print("Error: Input metadata is missing.")
        return False

    output_dir = (
        Path(sense.sense_config.export.output_dir)
        / sense.sense_config.hardware.device
        / sense.sense_config.build_option.backend
    )

    input_name, input_meta = next(iter(input_info.items()))
    input_shape = tuple(input_meta.get("shape", ()))
    input_dtype = np.dtype(input_meta.get("dtype", "float32"))
    rng = np.random.default_rng(42)
    input_data = rng.standard_normal(input_shape).astype(input_dtype)

    ort_session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    ort_outputs = ort_session.run(None, {input_name: input_data})
    if not ort_outputs:
        print("Error: ONNX runtime produced no outputs.")
        return False

    if len(ort_outputs) > 1:
        print("Warning: multiple outputs detected. Only the first will be compared.")
    ort_output = ort_outputs[0]
    ort_time_ms = _bench(lambda: ort_session.run(None, {input_name: input_data})) * 1000.0

    result = _run_binary_validation(
        title="Validation",
        model_dir=output_dir,
        executable_name=sense.sense_config.export.model_name,
        backend=sense.sense_config.build_option.backend,
        input_data=input_data,
        ort_output=ort_output,
        ort_time_ms=ort_time_ms,
    )
    if result is None:
        return False

    return result["is_close"]
