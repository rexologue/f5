#!/usr/bin/env python3
"""
Idempotent TensorRT SDK installer (Linux x86_64) + auto-pick tensorrt-llm.

- Resumes safely: skips steps already done
- Installs tensorrt from SDK wheels; polygraphy/onnx-graphsurgeon from NVIDIA PyPI if not in SDK
- Auto-pins tensorrt-llm by SDK series when known (10.11→1.0.0), otherwise installs latest
- --upgrade forces reinstall/upgrade of Python pkgs (tensorrt, polygraphy, onnx-graphsurgeon, tllm)
"""

import argparse
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.request
import subprocess
from pathlib import Path

# conservative mapping; extend as needed
TLLM_BY_TRT_SERIES = {
    "10.11": "1.0.0",
}

def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def echo(msg: str) -> None:
    print(f"[install_trt] {msg}")

def detect_linux_x86_64():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise SystemExit("This script supports only Linux x86_64.")

def build_default_url(version: str, cuda: str) -> str:
    family = ".".join(version.split(".")[:3])
    return f"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/{family}/tars/TensorRT-{version}.Linux.x86_64-gnu.cuda-{cuda}.tar.gz"

def download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    echo(f"Downloading:\n  {url}\n→ {dst}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return dst

def parse_trt_version_from_name(path_or_url: str) -> str | None:
    m = re.search(r"TensorRT-(\d+\.\d+\.\d+(?:\.\d+)?)[\.-]", path_or_url)
    return m.group(1) if m else None

def series(ver: str) -> str:
    return ".".join(ver.split(".")[:2])

def find_existing_trt_dir(install_dir: Path, want_ver: str | None) -> Path | None:
    cands = sorted([p for p in install_dir.iterdir() if p.is_dir() and p.name.startswith("TensorRT-")])
    if not cands:
        return None
    if not want_ver:
        return cands[-1]
    for p in cands:
        if f"TensorRT-{want_ver}" in p.name:
            return p
    return cands[-1]

def extract_tar_if_needed(tar_path: Path, install_dir: Path, src_hint: str, force: bool=False) -> Path:
    install_dir.mkdir(parents=True, exist_ok=True)
    want_ver = parse_trt_version_from_name(src_hint)
    existing = find_existing_trt_dir(install_dir, want_ver)
    if existing and not force:
        echo(f"Found TensorRT dir: {existing} (skip extraction)")
        return existing
    echo(f"Extracting {tar_path} → {install_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(install_dir)
    trt_dir = find_existing_trt_dir(install_dir, want_ver)
    if not trt_dir:
        raise SystemExit("Extraction finished but no 'TensorRT-*' directory was found.")
    echo(f"Found TensorRT dir: {trt_dir}")
    return trt_dir

def ensure_path_prepend(var: str, new_paths: list[str]) -> None:
    cur = os.environ.get(var, "")
    parts = [p for p in cur.split(":") if p]
    for p in reversed(new_paths):  # maintain order: first in list becomes left-most
        if p and p not in parts:
            parts.insert(0, p)
    os.environ[var] = ":".join(parts)

def export_env_for_process(trt_dir: Path):
    targets_lib = trt_dir / "targets" / "x86_64-linux-gnu" / "lib"
    lib = trt_dir / "lib"
    bin_dir = trt_dir / "bin"
    os.environ["TRT_DIR"] = str(trt_dir)
    ensure_path_prepend("PATH", [str(bin_dir)])
    libs = [str(lib)]
    if targets_lib.exists():
        libs.append(str(targets_lib))
    ensure_path_prepend("LD_LIBRARY_PATH", libs)
    echo(f"Env for process: PATH starts with {bin_dir}, LD_LIBRARY_PATH starts with {lib}")

def persist_env(trt_dir: Path):
    lines = [
        f'export TRT_DIR="{trt_dir}"',
        'export PATH="$TRT_DIR/bin:$PATH"',
        'export LD_LIBRARY_PATH="$TRT_DIR/lib:$LD_LIBRARY_PATH"',
        'if [ -d "$TRT_DIR/targets/x86_64-linux-gnu/lib" ]; then export LD_LIBRARY_PATH="$TRT_DIR/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH"; fi',
    ]
    for rc in [Path.home()/".bashrc", Path.home()/".zshrc"]:
        if not rc.exists():
            continue
        text = rc.read_text(encoding="utf-8")
        marker = "# >>> TensorRT SDK env >>>"
        block = marker + "\n" + "\n".join(lines) + "\n# <<< TensorRT SDK env <<<\n"
        if marker in text:
            start = text.find(marker)
            end = text.find("# <<< TensorRT SDK env <<<")
            new_text = text[:start] + block + text[end + len("# <<< TensorRT SDK env <<<") :]
        else:
            new_text = text + ("\n" if not text.endswith("\n") else "") + block
        rc.write_text(new_text, encoding="utf-8")
        echo(f"Persisted env to {rc}. Reload your shell: exec $SHELL -l")

def get_installed_version(module_name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(module_name.replace("-", "_"))
    except Exception:
        return None

def sdk_wheel_version(py_dir: Path, pkg: str) -> str | None:
    # expects wheel like: tensorrt-10.13.3.9-cp312-none-linux_x86_64.whl
    for whl in py_dir.glob(f"{pkg.replace('_','-')}*-*.whl"):
        name = whl.name
        m = re.match(rf"{re.escape(pkg).replace('_','[-_]')}[-_]([\w\.]+)-", name, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def pip_install_from_nvidia_pypi(pkgs: list[str], upgrade: bool):
    need = []
    for pkg in pkgs:
        ver = get_installed_version(pkg)
        if ver and not upgrade:
            echo(f"{pkg} already installed ({ver}), skip")
        else:
            need.append(pkg)
    if need:
        echo(f"Installing from NVIDIA PyPI: {', '.join(need)}")
        cmd = [sys.executable, "-m", "pip", "install", "--extra-index-url", "https://pypi.nvidia.com"]
        if upgrade:
            cmd.append("--upgrade")
        cmd += need
        sh(cmd)

def pip_install_from_sdk(trt_dir: Path, py_pkgs: list[str], upgrade: bool):
    py_dir = trt_dir / "python"
    if not py_dir.exists():
        raise SystemExit(f"Python wheels dir not found: {py_dir}")

    # Which wheels are present in SDK?
    present_names = {p.name.split("-")[0].lower() for p in py_dir.glob("*.whl")}
    sdk_present = []
    from_pypi = []
    for pkg in py_pkgs:
        base = pkg.lower().replace("_","-")
        if base in present_names:
            sdk_present.append(pkg)
        else:
            from_pypi.append(pkg)

    # Install/skip SDK wheels individually (check version)
    for pkg in sdk_present:
        want = sdk_wheel_version(py_dir, pkg)
        have = get_installed_version(pkg)
        if have and want and have == want and not upgrade:
            echo(f"{pkg} already installed ({have}) from SDK version, skip")
            continue
        echo(f"Installing {pkg} from SDK{' (upgrade)' if upgrade else ''}")
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={py_dir}"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(pkg)
        sh(cmd)

    # Anything not in SDK → NVIDIA PyPI (idempotent)
    if from_pypi:
        pip_install_from_nvidia_pypi(from_pypi, upgrade)

def pip_try_install(spec: str, indexes: list[str], upgrade: bool) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"]
    for url in indexes:
        cmd += ["--extra-index-url", url]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(spec)
    echo(f"Trying: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def tllm_candidates(user_pkg: str | None) -> list[str]:
    # Если явно задано --tllm-pkg — пробуем только его
    if user_pkg:
        return [user_pkg]
    # Иначе — перебор известных вариантов имён пакета
    return ["tensorrt-llm-cu12x", "tensorrt-llm", "tensorrt_llm"]

def tllm_installed_module_version() -> str | None:
    # независимый от имени пакета способ понять, что модуль есть
    try:
        import importlib.util as iu, importlib.metadata as md
        if iu.find_spec("tensorrt_llm") is None:
            return None
        try:
            return md.version("tensorrt-llm-cu12x")
        except Exception:
            try:
                return md.version("tensorrt-llm")
            except Exception:
                try:
                    return md.version("tensorrt_llm")
                except Exception:
                    return "installed (unknown dist name)"
    except Exception:
        return None

def ensure_tllm_auto(src_hint: str, trt_dir: Path, upgrade: bool, pkg_override: str | None, indexes: list[str]):
    # Уже установлен?
    have = tllm_installed_module_version()
    if have and not upgrade:
        echo(f"tensorrt-llm already installed ({have}), skip")
        return

    # Определяем пин версии по серии TRT (если есть)
    parsed = parse_trt_version_from_name(src_hint) or parse_trt_version_from_name(trt_dir.name) or ""
    pinned_ver = None
    if parsed:
        s = series(parsed)
        pinned_ver = TLLM_BY_TRT_SERIES.get(s)
        if pinned_ver:
            echo(f"Detected TensorRT {parsed} (series {s}). Pin tensorrt-llm=={pinned_ver}.")
        else:
            echo(f"Detected TensorRT {parsed} (series {s}). No explicit pin → will try latest tensorrt-llm.")
    else:
        echo("WARN: Could not parse TensorRT version → will try latest tensorrt-llm.")

    # Перебор возможных имён пакета
    tried = []
    for base in tllm_candidates(pkg_override):
        spec = f"{base}=={pinned_ver}" if pinned_ver else base
        if pip_try_install(spec, indexes, upgrade):
            echo(f"Installed {spec}")
            return
        tried.append(spec)

    # Если дошли сюда — не нашли ни одного подходящего колеса
    echo("WARN: Could not install TensorRT-LLM automatically.")
    echo("Tried: " + ", ".join(tried))
    echo("You can: 1) pass --tllm-pkg <name> to override package name; 2) add extra indexes with --tllm-index URL; 3) use --no-tllm and install manually.")

def find_trtexec(trt_dir: Path) -> str | None:
    for p in [
        trt_dir / "bin" / "trtexec",
        trt_dir / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec",
    ]:
        if p.exists():
            return str(p)
    for p in trt_dir.rglob("trtexec"):
        if p.is_file():
            return str(p)
    return None

def main():
    detect_linux_x86_64()

    ap = argparse.ArgumentParser(description="Install NVIDIA TensorRT SDK and auto-pick tensorrt-llm by SDK version (idempotent).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Direct URL to TensorRT tar.gz")
    src.add_argument("--tar", type=Path, help="Path to a local TensorRT tar.gz")
    ap.add_argument("--install-dir", type=Path, default=Path.home() / ".trt", help="Where to extract SDK")
    ap.add_argument("--persist", action="store_true", help="Persist PATH/LD_LIBRARY_PATH into ~/.bashrc / ~/.zshrc")
    ap.add_argument("--no-python", action="store_true", help="Skip installing Python wheels (tensorrt, polygraphy, onnx-graphsurgeon)")
    ap.add_argument("--no-tllm", action="store_true", help="Skip installing tensorrt-llm")
    ap.add_argument("--upgrade", action="store_true", help="Force reinstall/upgrade Python packages")
    ap.add_argument("--force-extract", action="store_true", help="Force re-extract SDK even if a TensorRT-* dir exists")
    ap.add_argument("--tllm-pkg", default=None, help="Override package base name for tensorrt-llm (e.g., 'tensorrt-llm')")
    ap.add_argument("--tllm-index", action="append", default=["https://pypi.nvidia.com"], help="Extra index URL(s) for tllm; can be passed multiple times")

    args = ap.parse_args()

    # Resolve tarball
    if args.tar:
        tar_path = args.tar.expanduser()
        if not tar_path.exists():
            raise SystemExit(f"Tarball not found: {tar_path}")
        src_hint = str(tar_path.name)
    else:
        url = args.url
        tar_path = args.install_dir.expanduser() / "TensorRT.tar.gz"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        if not tar_path.exists():
            tar_path = download(url, tar_path)
        else:
            echo(f"Tarball already exists: {tar_path} (skip download)")
        src_hint = url

    # Extract (idempotent)
    trt_dir = extract_tar_if_needed(tar_path, args.install_dir.expanduser(), src_hint, force=args.force_extract)

    # Env for this process (idempotent prepend)
    export_env_for_process(trt_dir)

    # Python wheels
    if not args.no_python:
        pip_install_from_sdk(trt_dir, ["tensorrt", "polygraphy", "onnx-graphsurgeon"], upgrade=args.upgrade)

    # tensorrt-llm auto
    if not args.no_tllm:
        ensure_tllm_auto(src_hint, trt_dir, upgrade=args.upgrade, pkg_override=args.tllm_pkg, indexes=args.tllm_index)

    # Persist env
    if args.persist:
        persist_env(trt_dir)

    # Sanity info
    exe = find_trtexec(trt_dir)
    if exe:
        echo(f"trtexec candidate: {exe}")
        try:
            out = subprocess.check_output([exe, "--version"], text=True).strip()
            echo("trtexec OK: " + out)
        except Exception as e:
            echo(f"Note: trtexec check failed: {e}")
    else:
        echo("Note: trtexec not found under TRT_DIR (yet).")

    try:
        code = "import tensorrt as trt; print('tensorrt', trt.__version__)"
        out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        echo(out)
    except Exception as e:
        echo(f"Note: Python import check failed (did wheels install?): {e}")

    echo("Done.")

if __name__ == "__main__":
    main()
