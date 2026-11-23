import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "afni_whereami"))

import streamlit as st
from bidsaid._helpers import iterable_to_str
from _streamlit_utils import _select_content
from whereami import run_pipeline


def clear_dir():
    st.session_state.plot_dst_dir = None


st.title("AFNI WhereAmI Pipeline")
st.divider()

st.markdown(
    "Pipeline for looking up MNI coordinates with AFNI's ``whereami``. "
    "Uses the Freesurfer MNI2009c DK parcellation and the "
    "Brodmann atlas for MNI 2009c - Pijnenburg AFNI version."
)

st.divider()

st.markdown("**Required Arguments**")

cohort = st.selectbox(
    "Cohort",
    ("kids", "adults"),
    help="Determines the template space used for visualization purposes.",
)

col1, col2, col3 = st.columns(3)

with col1:
    X = st.number_input("X", help="X coordinate in MNI space.", format="%d", value=0)

with col2:
    Y = st.number_input("Y", help="Y coordinate in MNI space.", format="%d", value=0)

with col3:
    Z = st.number_input("Z", help="Z coordinate in MNI space.", format="%d", value=0)

st.divider()

st.markdown("**Optional Arguments**")

if st.button("Browse for output directory"):
    if folder := _select_content("directory"):
        st.session_state.plot_dst_dir = folder

if st.session_state.get("plot_dst_dir"):
    st.success(f"Output directory: {st.session_state.plot_dst_dir}")

st.button(
    "Clear output directory",
    help="Clear to not save to output directory",
    on_click=clear_dir,
)

kwargs = {
    "cohort": cohort,
    "mni_coordinate": [X, Y, Z],
    "plot_dst_dir": st.session_state.get("plot_dst_dir"),
}

if st.button("Run Pipeline", type="primary"):
    with st.spinner("Processing..."):
        (
            st.session_state.afni_output_text,
            st.session_state.afni_sphere_figure,
            st.session_state.afni_sphere_filename,
            st.session_state.sphere_radius_text,
        ) = run_pipeline(**kwargs)

if st.session_state.get("afni_output_text"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text(st.session_state.get("afni_output_text"), text_alignment="center")

    if st.session_state.get("afni_sphere_filename"):
        st.success(f"Plot saved to: {st.session_state.get('afni_sphere_filename')}")

    if st.session_state.get("afni_sphere_figure"):
        st.pyplot(st.session_state.get("afni_sphere_figure"))
        additional_text = (
            f"[{iterable_to_str([X, Y, Z])}]"
            if st.session_state.get("sphere_radius_text")
            and "focus point" in st.session_state.get("sphere_radius_text").lower()
            else f"from {iterable_to_str([X, Y, Z])}"
        )
        st.caption(
            "**Note:** The radius of the sphere mask (red) adjusts to match the distance of the closest "
            f"label from the Freesurfer MNI2009c DK parcellation (which is {st.session_state.get('sphere_radius_text')} {additional_text})."
        )
