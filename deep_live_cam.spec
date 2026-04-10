# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

# Add project root to path
project_root = os.path.dirname(os.path.abspath(SPEC))

# Data files to include (models, locales, media, etc.)
datas = [
    ('models', 'models'),
    ('locales', 'locales'),
    ('media', 'media'),
]

# Add hidden imports that might be needed
hiddenimports = [
    'tkinter_fix',
    'tkinter',
    'PIL._tkinter_finder',
    'cv2',
    'cv2_enumerate_cameras',
    'numpy',
    'onnx',
    'onnxruntime',
    'insightface',
    'tensorflow',
    'opennsfw2',
    'customtkinter',
]

# Exclude unnecessary files to reduce size
excludes = [
    'test',
    'tests',
    'pytest',
    'setup.py',
    '__pycache__',
    '.git',
    '.github',
    '.ruff_cache',
]

a = Analysis(
    ['run.py'],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DeepLiveCam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Desktop app - no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='media/icon.ico',
    version='version_info.txt',
)