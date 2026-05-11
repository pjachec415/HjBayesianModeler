# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

a = Analysis(
    ['gui.py'],
    pathex=['.'],
    binaries=[
        *collect_dynamic_libs('PySide6'),
        *collect_dynamic_libs('shiboken6'),
        *collect_dynamic_libs('pyarrow'),
        *collect_dynamic_libs('fiona'),
        *collect_dynamic_libs('pyproj'),
        *collect_dynamic_libs('shapely'),
        *collect_dynamic_libs('pyogrio'),
    ],
    datas=[
        *collect_data_files('pymc'),
        *collect_data_files('arviz'),
        *collect_data_files('geopandas'),
        *collect_data_files('pytensor'),
        *collect_data_files('PySide6'),
        *collect_data_files('PySide6_Essentials'),
        *collect_data_files('shiboken6'),
        *collect_data_files('pyarrow'),
        *collect_data_files('fiona'),
        *collect_data_files('pyproj'),
        *collect_data_files('shapely'),
        *collect_data_files('pyogrio'),
        ('splash.png', '.'),   # splash screen image
        ('HjBM.ico', '.'),     # window + taskbar icon
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'shiboken6',
        'pymc',
        'arviz',
        'geopandas',
        'libpysal',
        'libpysal.weights',
        'libpysal.weights.contiguity',
        'numpy',
        'pandas',
        'pytensor',
        'pytensor.tensor',
        'pytensor.compile',
        'statsmodels',
        'statsmodels.api',
        'statsmodels.stats.outliers_influence',
        'statsmodels.genmod.families',
        'openpyxl',
        'pyarrow',
        'pyarrow.pandas_compat',
        'fiona',
        'fiona.ogrext',
        'fiona._shim',
        'fiona.schema',
        'pyogrio',
        'pyproj',
        'pyproj.transformer',
        'shapely',
        'shapely.geometry',
        'shapely.ops',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,      # <-- move binaries into the exe
    a.datas,         # <-- move datas into the exe
    [],
    name='HjBM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='HjBM.ico'
)
