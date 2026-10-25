import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "bids_conversion_pipeline")
)

import streamlit as st

from participants_tsv import run_pipeline
from _streamlit_utils import _select_content

st.set_page_config(layout="centered")

st.title("Participants TSV Pipeline")
st.divider()

st.markdown("""
Pipeline for creating the "participants.tsv" file or appending new participants to the "participants.tsv" file.\n

Output:

| participant_id |
|----------------|
| 101            |
| 102            |
| 103            |
| 104            |

**Notes:**\n
- If the BIDS directory has a participants TSV file, it will not be overwritten, the new subjects will be appended.\n
- If a participants TSV file exists in another directory (i.e., HPC) copy to the BIDS directory so that new subjects and demographic data can be appended.\n

**If the participants TSV file exists on the HPC transfer it to your local workstation via Globus and transfer it back to the directory after it's updated.**
""")

st.divider()
st.markdown("**Required Arguments**")

if st.button(
    "Browse for BIDS directory", help="Directory containing the BIDS-compliant dataset."
):
    folder = _select_content("directory")
    if folder:
        st.session_state.bids_dir = folder
        st.session_state.bids_subfolders = sorted(
            [
                x
                for x in Path(st.session_state.bids_dir).glob("*")
                if x.is_dir() and re.match(r"^sub-\d{2}", x.name)
            ]
        )

if st.session_state.get("bids_dir"):
    if st.session_state.bids_subfolders:
        st.success(f"BIDS directory: {st.session_state.bids_dir}")
    else:
        st.error(
            f"Not a valid BIDS directory (no subjects detected): {st.session_state.bids_dir}"
        )

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not (
        st.session_state.get("bids_dir") and st.session_state.get("bids_subfolders")
    ):
        st.error("Please select a valid BIDS directory before running.")
    else:
        with st.spinner("Processing..."):
            run_pipeline(st.session_state.get("bids_dir"))

        st.success("Participants TSV created/updated.")
