"""
PyInstaller Desktop Application Builder
Author: Santosh Narreddy
Builds a standalone executable for the Desktop Ear Jewellery Try-On App.
"""
import os
import sys
import subprocess

def build():
    print("[INFO] Packaging Ear Jewellery Try-On with PyInstaller...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=EarJewelleryTryOn",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--exclude-module=tensorflow",
        "--exclude-module=tensorboard",
        "--add-data=config.py:.",
        "--add-data=detectors:detectors",
        "--add-data=tracking:tracking",
        "--add-data=utils:utils",
        "--add-data=models:models",
        "app.py"
    ]
    subprocess.run(cmd, check=True)
    print("[SUCCESS] Desktop executable built in: dist/EarJewelleryTryOn")

if __name__ == '__main__':
    build()
