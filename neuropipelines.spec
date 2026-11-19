# https://pyinstaller.org/en/stable/spec-files.html
# https://medium.com/@arhamrumi/build-a-one-file-exe-with-pyinstaller-including-binaries-resources-760bcffe30ab

import ast
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = Path(SPECPATH)
APP_NAME = "NeuroPipelines"

ROOT_FILES = ["app.py", "_homepage.py", "_general_utils.py", "_streamlit_utils.py"]
APP_DIRS = [
    ".streamlit",
    "afni_whereami",
    "bids_conversion_pipeline",
    "connors",
    "events",
    "masks",
    "merge_data",
    "move_files",
    "nih_toolbox",
    "templates",
]

SKIP_DIRS = {"__pycache__", "api"}
SKIP_SUFFIXES = {".sb", ".md", ".pyc"}

EXTRA_IMPORTS = ["openpyxl", "tkinter.filedialog"]

COLLECT_ALL = ["streamlit", "nilearn"]
COLLECT_DATA = ["pdfminer"]

EXCLUDES = ["pyvista", "vtk", "vtkmodules", "imageio_ffmpeg", "pytest"]

def _app_files():
    files = [ROOT / name for name in ROOT_FILES]
    for folder in APP_DIRS:
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or path.suffix in SKIP_SUFFIXES:
                continue

            if SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
                continue

            files.append(path)

    return files


def _imports_used_by(py_files):
    local_names = {path.stem for path in py_files}
    local_names.update(part for path in py_files for part in path.relative_to(ROOT).parts)

    modules = set()
    for path in py_files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue

            modules.update(n for n in names if n.split(".")[0] not in local_names)

    return sorted(modules)


app_files = _app_files()

datas = [(str(path), str(path.relative_to(ROOT).parent)) for path in app_files]
binaries = []
hiddenimports = _imports_used_by([p for p in app_files if p.suffix == ".py"]) + EXTRA_IMPORTS

for package in COLLECT_ALL:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

for package in COLLECT_DATA:
    datas += collect_data_files(package)

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
