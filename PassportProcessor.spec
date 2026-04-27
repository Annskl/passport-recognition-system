# -*- mode: python ; coding: utf-8 -*-
import os
import cv2
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata


def safe_copy_metadata(package_name):
    try:
        return copy_metadata(package_name)
    except Exception:
        return []


def safe_collect_data_files(package_name):
    try:
        return collect_data_files(package_name)
    except Exception:
        return []


cv2_data_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')

datas = [('config.json', '.'), ('models', 'models'), (cv2_data_dir, 'cv2/data')]
datas += safe_collect_data_files('ultralytics')
datas += safe_collect_data_files('easyocr')
datas += safe_copy_metadata('torch')
datas += safe_copy_metadata('tqdm')
datas += safe_copy_metadata('regex')
datas += safe_copy_metadata('requests')
datas += safe_copy_metadata('packaging')
datas += safe_copy_metadata('filelock')
datas += safe_copy_metadata('numpy')
datas += safe_copy_metadata('tokenizers')


a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui',
        'cv2', 'ultralytics', 'easyocr', 'torch', 'torchvision',
        'pytesseract', 'pandas', 'openpyxl', 'sqlite3',
        'numpy', 'scipy', 'PIL', 'PIL.Image',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PassportProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PassportProcessor',
)
