import sys
from pathlib import Path

from streamlit.web import cli

base = getattr("_MEIPASS", Path(__file__).resolve().parent)
sys.argv = ["stremlit", "run", Path(base) / "app.py", "--global.developmentMode=false"]

sys.exit(cli.main())
