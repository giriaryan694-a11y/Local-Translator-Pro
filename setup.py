#!/usr/bin/env python3
"""
setup.py — Interactive installer for Offline Translator Pro (creates venv, installs Python packages,
and attempts to install system deps: tesseract, ffmpeg, build tools). Uses os + subprocess.

Run:
    python setup.py
"""

import os
import sys
import subprocess
import platform
import shutil
import venv
from pathlib import Path
from getpass import getuser

LOG = []

def run(cmd, check=True, shell=False):
    LOG.append("RUN: " + (" ".join(cmd) if isinstance(cmd, (list,tuple)) else str(cmd)))
    try:
        if shell:
            res = subprocess.run(cmd, shell=True, check=check)
        else:
            res = subprocess.run(cmd, check=check)
        return res.returncode == 0
    except subprocess.CalledProcessError as e:
        LOG.append(f"ERROR: {e}")
        return False

def which(cmd):
    return shutil.which(cmd) is not None

def print_header():
    print("="*60)
    print("Offline Translator Pro — Installer".center(60))
    print("Made by Aryan Giri".center(60))
    print("="*60)

def ask_bool(prompt, default=True):
    d = "Y/n" if default else "y/N"
    ans = input(f"{prompt} [{d}] ").strip().lower()
    if ans == "":
        return default
    return ans in ("y","yes")

def safe_input(prompt, default=""):
    v = input(f"{prompt} ").strip()
    return v if v else default

def create_venv(venv_path:Path):
    print(f"\nCreating virtual environment in: {venv_path}")
    if venv_path.exists():
        print("Virtualenv already exists — skipping creation.")
        return True
    try:
        venv.create(venv_path, with_pip=True)
        print("✅ Virtualenv created.")
        return True
    except Exception as e:
        print("❌ Failed to create virtualenv:", e)
        return False

def get_venv_pip(venv_path:Path):
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"

def get_venv_python(venv_path:Path):
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def install_system_packages(pkg_list):
    system = platform.system().lower()
    print(f"\nInstalling system packages: {', '.join(pkg_list)} (detected OS: {system})")
    ok = False
    if system == "linux":
        # try apt, yum, pacman
        if which("apt-get"):
            cmd = ["sudo","apt-get","update"]
            run(cmd)
            cmd = ["sudo","apt-get","install","-y"] + pkg_list
            ok = run(cmd)
        elif which("yum"):
            cmd = ["sudo","yum","install","-y"] + pkg_list
            ok = run(cmd)
        elif which("pacman"):
            cmd = ["sudo","pacman","-Sy"] + pkg_list
            ok = run(cmd)
        else:
            print("No known package manager found (apt/yum/pacman). Please install packages manually.")
    elif system == "darwin":
        if which("brew"):
            cmd = ["brew","install"] + pkg_list
            ok = run(cmd)
        else:
            print("Homebrew not found. Install Homebrew (https://brew.sh/) then re-run installer or install packages manually.")
    elif system == "windows":
        # try winget or choco
        if which("winget"):
            for p in pkg_list:
                run(f"winget install -e --id {p}", shell=True)
            ok = True
        elif which("choco"):
            cmd = ["choco","install","-y"] + pkg_list
            ok = run(cmd)
        else:
            print("Neither winget nor choco found. Please install Tesseract and FFmpeg manually on Windows.")
            ok = False
    else:
        print("Unsupported OS for automated system package install. Please install packages manually.")
    if ok:
        print("✅ System packages installation step completed (or attempted).")
    else:
        print("⚠️ System packages may not have been installed automatically. See messages above.")
    return ok

def pip_install(pip_path, packages):
    pip_cmd = [str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"]
    print("\nUpgrading pip, setuptools, wheel...")
    run(pip_cmd)
    # install packages in chunks to show progress
    for pkg in packages:
        print(f"\nInstalling Python package: {pkg}")
        ok = run([str(pip_path), "install", pkg])
        if not ok:
            print(f"⚠️ Failed to install {pkg}. You may need to install it manually.")

def attempt_torch_install(pip_path, prefer_gpu):
    print("\nInstalling PyTorch — this step is tricky due to CUDA versions.")
    if prefer_gpu:
        print("Attempting to install common CUDA-enabled PyTorch wheels (best-effort).")
        # try a few common cuda versions
        cuda_candidates = [
            ("cu117","https://download.pytorch.org/whl/cu117"),
            ("cu118","https://download.pytorch.org/whl/cu118"),
            ("cu121","https://download.pytorch.org/whl/cu121"),
        ]
        for label, url in cuda_candidates:
            print(f"Trying {label} wheel index...")
            ok = run([str(pip_path), "install", "torch", "torchaudio", "torchvision", "--index-url", url])
            if ok:
                print(f"✅ Installed torch with {label}.")
                return True
        print("⚠️ GPU-enabled wheels failed or not compatible. Falling back to CPU-only torch.")
    # CPU fallback
    ok = run([str(pip_path), "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    if ok:
        print("✅ Installed CPU-only torch.")
    else:
        print("❌ Failed to install torch. You may need to install it manually following instructions at https://pytorch.org/")
    return ok

def main():
    print_header()

    # 1) create venv?
    use_venv = ask_bool("Create and use a virtual environment for this project?", default=True)
    venv_path = Path("venv") if use_venv else None

    if use_venv:
        created = create_venv(venv_path)
        if not created:
            print("Failed to create virtualenv — aborting.")
            sys.exit(1)
        pip_path = get_venv_pip(venv_path)
        python_path = get_venv_python(venv_path)
    else:
        pip_path = shutil.which("pip") or shutil.which("pip3")
        python_path = shutil.which("python") or shutil.which("python3")
        if not pip_path or not python_path:
            print("pip or python not found in PATH. Please install Python and pip or enable virtualenv.")
            sys.exit(1)

    # 2) system packages
    if ask_bool("Attempt to install system packages (Tesseract OCR, FFmpeg, build tools)?", default=True):
        # map to package names per OS
        system = platform.system().lower()
        if system == "linux":
            pkg_list = ["tesseract-ocr","ffmpeg","build-essential","libsndfile1","libasound2-dev"]
        elif system == "darwin":
            pkg_list = ["tesseract","ffmpeg"]
        elif system == "windows":
            # winget/choco package ids vary; we give user instructions otherwise attempt common ids
            pkg_list = ["GnuWin32.Tesseract", "ffmpeg.ffmpeg"]  # winget/choco guesses
        else:
            pkg_list = []
        if pkg_list:
            install_system_packages(pkg_list)

    # 3) choose torch (GPU or CPU)
    prefer_gpu = False
    if ask_bool("Would you like to attempt installing GPU-enabled PyTorch (if you have CUDA)? (may fail)", default=False):
        prefer_gpu = True

    # 4) install python packages
    print("\nNow installing required Python packages into:", pip_path)
    base_packages = [
        "transformers>=4.30.0",
        "pyfiglet",
        "termcolor",
        "colorama",
        "pillow",
        "pydub",
        "torchaudio",
        "pytesseract",
        "SpeechRecognition",
        "pocketsphinx",
        "pyqt6",
        "huggingface_hub",
        "huggingface-hub",
        "python-dateutil",
    ]
    # Always attempt torch via dedicated function to support CPU/GPU variants
    attempt_torch_install(pip_path, prefer_gpu)

    # Install remaining packages
    pip_install(pip_path, base_packages)

    # 5) extra notes and final steps
    print("\n" + "="*60)
    print("Setup finished (or attempted). Summary / Next steps:")
    if use_venv:
        if platform.system().lower() == "windows":
            print(f"  - Activate venv: {venv_path}\\Scripts\\activate")
        else:
            print(f"  - Activate venv: source {venv_path}/bin/activate")
    else:
        print("  - You installed into your system Python environment.")

    print("  - If system packages (Tesseract, FFmpeg) failed to install, install them manually:")
    if platform.system().lower() == "linux":
        print("      sudo apt install tesseract-ocr ffmpeg")
    elif platform.system().lower() == "darwin":
        print("      brew install tesseract ffmpeg")
    elif platform.system().lower() == "windows":
        print("      Use winget or choco to install tesseract and ffmpeg, or install from official packages.")

    print("\nRun your translator script inside the venv (or system Python):")
    print("  python translator_cli.py   # for CLI")
    print("  python translator_gui.py   # for GUI (PyQt)")
    print("\nIf PyTorch or Whisper fails with CUDA errors, install the correct torch wheel matching your CUDA driver from https://pytorch.org/")
    print("\nLog of actions saved to installer_log.txt")
    with open("installer_log.txt","w",encoding="utf-8") as f:
        f.write("\n".join(LOG))

    print("\nDone. If you hit errors, read installer_log.txt and follow the printed manual instructions.")
    print("="*60)

if __name__ == "__main__":
    main()
