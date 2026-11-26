import logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "nih_toolbox"))

import streamlit as st

from _streamlit_utils import StreamlitLogHandler, _select_content

from organize_toolbox_data import run_pipeline

st.set_page_config(layout="centered")

st.title("NIH Toolbox Pipeline")
st.divider()
st.markdown(
    """
Pipeline for converting NIH toolbox data from long-form (where the assessments are in row form) to wide form (where the assessments are in columns)\n

Original format:

| participant_id | Name       | RawScore |
|----------------|------------|----------|
| 101            | Test1      | 0        |
| 101            | Test2      | 5        |
| 101            | Test3      | 10       |

Converted (output format):

| participant_id | Test 1 Raw Score | Test 2 Raw Score | Test 3 Raw Score |
|----------------|------------------|------------------|------------------|
| 101            | 0                | 5                | 10               |

**Note**:\n
- Use the CSV file containing the following columns: 'RawScore', 'Theta', 'SE', 'TScore', 'Computed Score',
'Uncorrected Standard Score', 'Age-Corrected Standard Score', 'National Percentile (age adjusted)', 'Fully-Corrected T-score' as columns."""
)

st.divider()

st.markdown("**Required Arguments**")
cohort = st.selectbox("Cohort", ("Kids", "Adults"), help="The cohort.")

if st.button(
    "Browse for source (NIH toolbox) file",
    help="The unorganized NIH toolbox file.",
):
    file = _select_content("file")
    if file:
        st.session_state.unorganized_nih_toolbox_file = file

if st.session_state.get("unorganized_nih_toolbox_file"):
    st.success(f"NIH Toolbox file: {st.session_state.unorganized_nih_toolbox_file}")

st.divider()
st.markdown("**Optional Arguments**")

prefix_filename = st.text_input(
    "Prefix filename",
    help=" prefix to add to the filename for the organized NIH toolbox data.",
)
if prefix_filename:
    prefix_filename = prefix_filename.strip()

if st.button(
    "Browse for output directory",
    help=(
        "Path to the output directory for the organized NIH toolbox data. "
        "If None, saves to the same directory as input data"
    ),
):
    folder = _select_content("directory")
    if folder:
        st.session_state.dst_dir = folder

if st.session_state.get("dst_dir"):
    st.success(f"Output directory: {st.session_state.dst_dir}")

if st.button(
    "Browse for pre-existing organized (NIH toolbox) file",
    help="Directory containing a pre-existing organized NIH toolbox file.",
):
    file = _select_content("file")
    if file:
        st.session_state.preexisting_nih_toolbox_file = file

if st.session_state.get("preexisting_nih_toolbox_file"):
    st.success(
        f"Pre-existing NIH Toolbox file: {st.session_state.preexisting_nih_toolbox_file}"
    )

include_assessment_dates = st.checkbox(
    "Include assessment dates",
    value=True,
    help="Includes the assessment dates from the data.",
)

kwargs = {
    "cohort": cohort.lower(),
    "unorganized_nih_toolbox_file": st.session_state.get(
        "unorganized_nih_toolbox_file"
    ),
    "dst_dir": st.session_state.get("dst_dir"),
    "prefix_filename": prefix_filename,
    "preexisting_nih_toolbox_file": st.session_state.get(
        "preexisting_nih_toolbox_file"
    ),
    "include_assessment_dates": include_assessment_dates,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not st.session_state.get("unorganized_nih_toolbox_file"):
        st.error("Please select a source (NIH toolbox file) file before running.")
    else:
        status_container = st.empty()
        with status_container.status("Running pipeline...", expanded=True) as status:
            handler = StreamlitLogHandler(status)
            logging.getLogger().addHandler(handler)

            output_path = run_pipeline(**kwargs)

            logging.getLogger().removeHandler(handler)

            if output_path:
                status.update(
                    label=f"Data saved to: {output_path}",
                    state="complete",
                    expanded=False,
                )
