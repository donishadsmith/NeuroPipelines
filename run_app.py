import sys, os

from streamlit.web import cli

base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
sys.argv = [
    "streamlit",
    "run",
    os.path.join(base, "app.py"),
    "--global.developmentMode=false",
    "--server.showEmailPrompt=false",
    "--server.fileWatcherType=none",
]

sys.exit(cli.main())
