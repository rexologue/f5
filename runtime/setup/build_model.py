#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import site
from shutil import which

from utils import preload_libpython
preload_libpython()

# ---------------- utils ----------------


def _paths_exist(*ps: Path) -> list[str]:
    return [str(p) for p in ps if p and p.exists()]

def make_subproc_env() -> dict:
    """
    Собирает env для сабпроцессов: добавляет
    - conda/lib (libpython*.so) в LD_LIBRARY_PATH (+ LD_PRELOAD при наличии точного .so.1.0)
    - TRT_DIR/bin и targets/.../bin в PATH
    - TRT_DIR/lib и targets/.../lib в LD_LIBRARY_PATH
    """
    env = os.environ.copy()

    # 1) libpython из активного env
    cand_libs = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cand_libs += [Path(conda_prefix)/"lib", Path(conda_prefix)/"lib64"]
    cand_libs += [Path(sys.prefix)/"lib", Path(sys.prefix)/"lib64", Path(sys.exec_prefix)/"lib", Path(sys.exec_prefix)/"lib64"]
    cand_libs = [p for p in cand_libs if p.exists()]

    libdir = None
    for d in cand_libs:
        if list(d.glob(f"libpython{sys.version_info.major}.{sys.version_info.minor}*.so*")):
            libdir = d
            break
    if libdir:
        env["LD_LIBRARY_PATH"] = f"{str(libdir)}:{env.get('LD_LIBRARY_PATH','')}"
        exact = next(iter(libdir.glob(f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0")), None)
        if exact:
            env["LD_PRELOAD"] = f"{str(exact)}:{env.get('LD_PRELOAD','')}" if env.get("LD_PRELOAD") else str(exact)

    # 2) TensorRT bin/lib
    trt_dir = os.environ.get("TRT_DIR")
    if trt_dir:
        trt_dir = Path(trt_dir)
        bin_main = trt_dir / "bin"
        bin_targets = trt_dir / "targets" / "x86_64-linux-gnu" / "bin"
        lib_main = trt_dir / "lib"
        lib_targets = trt_dir / "targets" / "x86_64-linux-gnu" / "lib"

        prepend_bins = _paths_exist(bin_main, bin_targets)
        if prepend_bins:
            env["PATH"] = ":".join(prepend_bins + [env.get("PATH","")])

        prepend_libs = _paths_exist(lib_main, lib_targets)
        if prepend_libs:
            env["LD_LIBRARY_PATH"] = ":".join(prepend_libs + [env.get("LD_LIBRARY_PATH","")])

    return env

def echo(msg: str) -> None:
    print(msg, flush=True)

def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    echo(" ".join(cmd))
    if env is None:
        env = make_subproc_env()  # <-- ключевое
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)

def is_dir_empty(p: Path) -> bool:
    if not p.exists():
        echo(f"Ошибка: '{p}' не существует или не является директорией")
        return True
    if not p.is_dir():
        echo(f"Ошибка: '{p}' не является директорией")
        return True
    return next(p.iterdir(), None) is None

def copy_tree_into(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in src_dir.iterdir():
        src = entry
        dst = dst_dir / entry.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

def first_site_packages() -> Path:
    cands = [Path(p) for p in site.getsitepackages() if "site-packages" in p]
    if cands:
        return cands[0]
    usp = site.getusersitepackages()
    if usp:
        return Path(usp)
    raise RuntimeError("Не удалось определить путь site-packages")

# ---------------- trtexec integration (from export_vocos_trt.sh) ----------------

def find_trtexec() -> str | None:
    # 1) PATH
    from shutil import which
    p = which("trtexec")
    if p:
        return p
    # 2) common locations
    for cand in (
        "/opt/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/bin/trtexec",
    ):
        if Path(cand).exists():
            return cand
    # 3) under TRT_DIR (both bin & targets bin)
    trt_dir = os.environ.get("TRT_DIR", "")
    if trt_dir:
        for sub in ("bin/trtexec", "targets/x86_64-linux-gnu/bin/trtexec"):
            cand = Path(trt_dir) / sub
            if cand.exists():
                return str(cand)
    # 4) quick search nearby
    if trt_dir:
        for pth in Path(trt_dir).rglob("trtexec"):
            if pth.is_file():
                return str(pth)
    return None

def build_vocos_engine_with_trtexec(onnx_path: Path, engine_path: Path, precision: str) -> None:
    trtexec = find_trtexec()
    if not trtexec:
        echo("[-] trtexec не найден. Установите TensorRT или добавьте trtexec в PATH.")
        sys.exit(1)

    onnx_path = onnx_path.resolve()
    engine_path = engine_path.resolve()

    echo(f"[i] trtexec: {trtexec}")
    echo(f"[i] ONNX:    {onnx_path}")
    echo(f"[i] ENGINE:  {engine_path}")
    echo(f"[i] PREC:    {precision}")

    # precision flags
    prec_flags: list[str] = []
    if precision == "fp16":
        prec_flags = ["--fp16"]
    elif precision == "bf16":
        prec_flags = ["--bf16"]
    elif precision == "fp32":
        prec_flags = []
    else:
        echo(f"[-] Неизвестная PRECISION={precision} (ожидаю fp32|fp16|bf16)")
        sys.exit(2)

    # dynamic shapes for mel [B, 100, T]
    MIN_BATCH_SIZE=1; OPT_BATCH_SIZE=1; MAX_BATCH_SIZE=8
    MIN_INPUT_LENGTH=1; OPT_INPUT_LENGTH=1000; MAX_INPUT_LENGTH=3000
    MEL_MIN_SHAPE=f"{MIN_BATCH_SIZE}x100x{MIN_INPUT_LENGTH}"
    MEL_OPT_SHAPE=f"{OPT_BATCH_SIZE}x100x{OPT_INPUT_LENGTH}"
    MEL_MAX_SHAPE=f"{MAX_BATCH_SIZE}x100x{MAX_INPUT_LENGTH}"

    cmd = [
        trtexec,
        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(engine_path)}",
        f"--minShapes=mel:{MEL_MIN_SHAPE}",
        f"--optShapes=mel:{MEL_OPT_SHAPE}",
        f"--maxShapes=mel:{MEL_MAX_SHAPE}",
    ] + prec_flags

    run(cmd)

# ---------------- main pipeline ----------------

def main():
    parser = argparse.ArgumentParser(description="Build TRT-LLM engine and Vocos vocoder (Python port, trtexec integrated)")
    parser.add_argument("PT_CHECKPOINT_PATH", type=Path, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("VOCODER_DIR", type=Path, help="Local path to Vocos vocoder (weights/config)")
    parser.add_argument("OUTPUT_DIR", type=Path, help="Output directory")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "5"), help="CUDA_VISIBLE_DEVICES value (default: 5)")
    parser.add_argument("--precision", choices=["fp32","fp16","bf16"], default=os.environ.get("PRECISION","fp32"), help="precision for vocoder engine (default from $PRECISION or fp32)")
    args = parser.parse_args()

    # env
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # resolve paths
    output_dir = args.OUTPUT_DIR.resolve()
    f5_trt_ckpt = (output_dir / "trt_f5_ckpt").resolve()
    f5_trt_engine = (output_dir / "trt_f5_engine").resolve()
    vocos_onnx = (output_dir / "onnx_vocoder.onnx").resolve()
    vocos_trt_engine = (output_dir / "vocoder_engine.plan").resolve()

    pkg_dir = first_site_packages()
    tensorrt_llm_models_dir = pkg_dir / "tensorrt_llm" / "models"

    # dirs
    f5_trt_ckpt.mkdir(parents=True, exist_ok=True)
    f5_trt_engine.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # base dir of this script to access ./exporting and ./patch
    script_dir = Path(__file__).resolve().parent
    patch_dir = script_dir / "patch"

    # Step 1: Convert checkpoint
    if is_dir_empty(f5_trt_ckpt):
        echo("Converting checkpoint")
        run(
            [
                sys.executable,
                "convert_checkpoint.py",
                "--timm_ckpt", str(args.PT_CHECKPOINT_PATH),
                "--output_dir", str(f5_trt_ckpt),
            ]
        )

    # Step 2: Build engine
    if is_dir_empty(f5_trt_engine):
        echo("Building engine")
        # cp -r patch/* $PKG/tensorrt_llm/models
        if patch_dir.exists():
            copy_tree_into(patch_dir, tensorrt_llm_models_dir)
        else:
            echo(f"Предупреждение: каталог patch не найден: {patch_dir}")

        run(
            [
                "trtllm-build",
                "--checkpoint_dir", str(f5_trt_ckpt),
                "--max_batch_size", "8",
                "--output_dir", str(f5_trt_engine),
                "--remove_input_padding", "disable",
            ]
        )

    # Step 3: Export vocoder ONNX
    if not vocos_onnx.exists():
        echo("Exporting vocos vocoder")
        run(
            [
                sys.executable,
                "export_vocoder_to_onnx.py",
                "--vocoder-path", str(args.VOCODER_DIR),
                "--output-path", str(vocos_onnx),
            ]
        )

    # Step 4: Build vocoder TRT engine (integrated trtexec)
    if not vocos_trt_engine.exists():
        echo("Building vocos vocoder engine (trtexec)")
        build_vocos_engine_with_trtexec(vocos_onnx, vocos_trt_engine, args.precision)

    echo("Done.")

if __name__ == "__main__":
    main()
