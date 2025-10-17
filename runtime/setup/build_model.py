# runtime/setup/build_model.py (ИСПРАВЛЕНО для TRT-LLM 1.0)
#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils import preload_libpython
preload_libpython()

# ---------------- utils ----------------
def echo(msg: str) -> None:
    print(msg, flush=True)

def make_subproc_env() -> dict:
    """Собирает env для сабпроцессов с правильными путями."""
    env = os.environ.copy()
    
    # 1) libpython
    cand_libs = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cand_libs += [Path(conda_prefix) / "lib", Path(conda_prefix) / "lib64"]
    
    cand_libs += [
        Path(sys.prefix) / "lib",
        Path(sys.prefix) / "lib64",
        Path(sys.exec_prefix) / "lib",
        Path(sys.exec_prefix) / "lib64",
    ]
    
    cand_libs = [p for p in cand_libs if p.exists()]
    
    libdir = None
    for d in cand_libs:
        if list(d.glob(f"libpython{sys.version_info.major}.{sys.version_info.minor}*.so*")):
            libdir = d
            break
    
    if libdir:
        ld_path = env.get("LD_LIBRARY_PATH", "")
        parts = [p for p in ld_path.split(":") if p]
        if str(libdir) not in parts:
            parts.insert(0, str(libdir))
        env["LD_LIBRARY_PATH"] = ":".join(parts)
        
        exact = next(iter(libdir.glob(f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0")), None)
        if exact:
            ld_preload = env.get("LD_PRELOAD", "")
            preloads = [p for p in ld_preload.split(":") if p]
            if str(exact) not in preloads:
                preloads.insert(0, str(exact))
            env["LD_PRELOAD"] = ":".join(preloads)
    
    # 2) TensorRT
    trt_dir = os.environ.get("TRT_DIR")
    if trt_dir:
        trt_dir = Path(trt_dir)
        
        bin_main = trt_dir / "bin"
        bin_targets = trt_dir / "targets" / "x86_64-linux-gnu" / "bin"
        prepend_bins = [str(p) for p in [bin_main, bin_targets] if p.exists()]
        if prepend_bins:
            path = env.get("PATH", "")
            path_parts = [p for p in path.split(":") if p]
            for pb in reversed(prepend_bins):
                if pb not in path_parts:
                    path_parts.insert(0, pb)
            env["PATH"] = ":".join(path_parts)
        
        lib_main = trt_dir / "lib"
        lib_targets = trt_dir / "targets" / "x86_64-linux-gnu" / "lib"
        prepend_libs = [str(p) for p in [lib_main, lib_targets] if p.exists()]
        if prepend_libs:
            ld_path = env.get("LD_LIBRARY_PATH", "")
            ld_parts = [p for p in ld_path.split(":") if p]
            for pl in reversed(prepend_libs):
                if pl not in ld_parts:
                    ld_parts.insert(0, pl)
            env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    
    return env

def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    echo(" ".join(cmd))
    if env is None:
        env = make_subproc_env()
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)

def is_dir_empty(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
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
    import site
    cands = [Path(p) for p in site.getsitepackages() if "site-packages" in p]
    if cands:
        return cands[0]
    usp = site.getusersitepackages()
    if usp:
        return Path(usp)
    raise RuntimeError("Не удалось определить путь site-packages")

def find_trtexec() -> str | None:
    from shutil import which
    p = which("trtexec")
    if p:
        return p
    
    for cand in [
        "/opt/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/bin/trtexec",
    ]:
        if Path(cand).exists():
            return cand
    
    trt_dir = os.environ.get("TRT_DIR", "")
    if trt_dir:
        for sub in ["bin/trtexec", "targets/x86_64-linux-gnu/bin/trtexec"]:
            cand = Path(trt_dir) / sub
            if cand.exists():
                return str(cand)
        
        for pth in Path(trt_dir).rglob("trtexec"):
            if pth.is_file():
                return str(pth)
    
    return None

def build_vocos_engine_with_trtexec(onnx_path: Path, engine_path: Path, precision: str) -> None:
    trtexec = find_trtexec()
    if not trtexec:
        echo("[-] trtexec не найден")
        sys.exit(1)
    
    onnx_path = onnx_path.resolve()
    engine_path = engine_path.resolve()
    
    echo(f"[i] trtexec: {trtexec}")
    echo(f"[i] ONNX:    {onnx_path}")
    echo(f"[i] ENGINE:  {engine_path}")
    echo(f"[i] PREC:    {precision}")
    
    prec_flags = []
    if precision == "fp16":
        prec_flags = ["--fp16"]
    elif precision == "bf16":
        prec_flags = ["--bf16"]
    elif precision not in ["fp32", "float32"]:
        echo(f"[-] Неизвестная PRECISION={precision}")
        sys.exit(2)
    
    MIN_BATCH = 1
    OPT_BATCH = 2
    MAX_BATCH = 8
    
    MIN_T = 10
    OPT_T = 500
    MAX_T = 3000
    
    MEL_MIN = f"{MIN_BATCH}x100x{MIN_T}"
    MEL_OPT = f"{OPT_BATCH}x100x{OPT_T}"
    MEL_MAX = f"{MAX_BATCH}x100x{MAX_T}"
    
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=mel:{MEL_MIN}",
        f"--optShapes=mel:{MEL_OPT}",
        f"--maxShapes=mel:{MEL_MAX}",
        "--verbose",
        "--memPoolSize=workspace:4096",
    ] + prec_flags
    
    echo(f"[+] Запуск trtexec")
    run(cmd)
    echo(f"✅ Vocoder engine сохранён: {engine_path}")

def main():
    parser = argparse.ArgumentParser(description="Build TRT-LLM engine and Vocos vocoder")
    parser.add_argument("PT_CHECKPOINT_PATH", type=Path)
    parser.add_argument("VOCODER_DIR", type=Path)
    parser.add_argument("OUTPUT_DIR", type=Path)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-input-len", type=int, default=2048, help="Max input length")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max sequence length (defaults to max_input_len)")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    echo(f"🎯 Используется GPU: {args.cuda_visible_devices}")
    
    output_dir = args.OUTPUT_DIR.resolve()
    f5_trt_ckpt = output_dir / "trt_f5_ckpt"
    f5_trt_engine = output_dir / "trt_f5_engine"
    vocos_onnx = output_dir / "onnx_vocoder.onnx"
    vocos_trt_engine = output_dir / "vocoder_engine.plan"
    
    for d in [f5_trt_ckpt, f5_trt_engine, output_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    script_dir = Path(__file__).resolve().parent
    patch_dir = script_dir / "patch"
    
    # Step 1: Convert checkpoint
    if is_dir_empty(f5_trt_ckpt):
        echo("📦 Конвертация checkpoint → TRT")
        run([
            sys.executable,
            str(script_dir / "convert_checkpoint.py"),
            "--timm_ckpt", str(args.PT_CHECKPOINT_PATH),
            "--output_dir", str(f5_trt_ckpt),
            "--vocab-size", str(args.vocab_size) if args.vocab_size else "0",
        ])
    else:
        echo("✅ Checkpoint уже сконвертирован, пропускаем")
    
    # Step 2: Build DiT engine
    if is_dir_empty(f5_trt_engine):
        echo("🔨 Билд DiT TRT engine")
        
        if patch_dir.exists():
            pkg_dir = first_site_packages()
            trt_models_dir = pkg_dir / "tensorrt_llm" / "models"
            copy_tree_into(patch_dir, trt_models_dir)
            echo(f"✅ Patch скопирован в {trt_models_dir}")
        
        # Определяем max_seq_len
        max_seq_len = args.max_seq_len if args.max_seq_len else args.max_input_len
        
        # TRT-LLM 1.0 совместимые аргументы
        build_cmd = [
            "trtllm-build",
            "--checkpoint_dir", str(f5_trt_ckpt),
            "--max_batch_size", str(args.max_batch_size),
            "--output_dir", str(f5_trt_engine),
            "--remove_input_padding", "disable",  # ВАЖНО для DiT
            "--max_beam_width", "1",
            "--gpt_attention_plugin", "float16",
            "--gemm_plugin", "float16",
            "--max_input_len", str(args.max_input_len),
            "--max_seq_len", str(max_seq_len),
            "--workers", "1",
        ]
        
        run(build_cmd, env=make_subproc_env())
    else:
        echo("✅ DiT engine уже собран, пропускаем")
    
    # Step 3: Export vocoder ONNX
    if not vocos_onnx.exists():
        echo("🎵 Экспорт Vocos → ONNX")
        run([
            sys.executable,
            str(script_dir / "export_vocoder_to_onnx.py"),
            "--vocoder-path", str(args.VOCODER_DIR),
            "--output-path", str(vocos_onnx),
        ])
    else:
        echo("✅ Vocos ONNX уже экспортирован, пропускаем")
    
    # Step 4: Build vocoder engine
    if not vocos_trt_engine.exists():
        echo("🚀 Билд Vocos TRT engine")
        build_vocos_engine_with_trtexec(vocos_onnx, vocos_trt_engine, args.precision)
    else:
        echo("✅ Vocos engine уже собран, пропускаем")
    
    echo("\n" + "="*80)
    echo("✅ Готово!")
    echo(f"  DiT engine:  {f5_trt_engine}")
    echo(f"  Vocoder:     {vocos_trt_engine}")
    echo("="*80)

if __name__ == "__main__":
    main()