import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "masks"))

import streamlit as st

from _streamlit_utils import _select_content
from sphere_mask import run_pipeline


def clear_dir():
    st.session_state.sphere_dst_dir = None


st.title("Sphere Mask Pipeline")
st.divider()

st.markdown(
    "Pipeline for generating a-priori sphere masks for seed-based connectivity analyses."
)

st.divider()

st.markdown("**Required Arguments**")

cohort = st.selectbox(
    "Cohort", ("Kids", "Adults"), help="Determines the template space used."
)

col1, col2, col3 = st.columns(3)

with col1:
    X = st.number_input(
        "X", help="X coordinate in MNI or Talairach.", format="%d", value=0
    )

with col2:
    Y = st.number_input(
        "Y", help="Y coordinate in MNI or Talairach.", format="%d", value=0
    )

with col3:
    Z = st.number_input(
        "Z", help="Z coordinate in MNI or Talairach.", format="%d", value=0
    )

sphere_radius = st.number_input(
    "Sphere radius",
    help="The radius of the sphere mask in mm.",
    min_value=3,
    max_value=10,
    value=5,
    format="%d",
)

original_coordinate_space = st.selectbox(
    "Original Coordinate Space",
    ("MNI", "Talairach"),
    help="The original space of the coordinate.",
)

if original_coordinate_space == "Talairach":
    transform_method = st.selectbox(
        "Talairach to MNI transform method",
        ("Lancaster", "Brett"),
        help=(
            "The method to use to transform the Talairach coordinate to MNI space. "
            "Use the Lancaster method unless the paper is pre-2010 or specifically "
            "states that the Brett transform was used."
        ),
    )
else:
    # Just to ensure its initialized
    transform_method = "Lancaster"

st.divider()


st.markdown("**Optional Arguments**")

use_black_bg = st.checkbox("Use a black background in image.")

if st.button("Browse for output directory", help="For saving plot and sphere mask"):
    if folder := _select_content("directory"):
        st.session_state.sphere_dst_dir = folder

if st.session_state.get("sphere_dst_dir"):
    st.success(f"Output directory: {st.session_state.sphere_dst_dir}")

st.button(
    "Clear output directory",
    help="Clear to not save to output directory",
    on_click=clear_dir,
)

kwargs = {
    "cohort": cohort.lower(),
    "coordinate": [X, Y, Z],
    "sphere_radius": sphere_radius,
    "original_coordinate_space": original_coordinate_space,
    "transform_method": transform_method,
    "use_black_bg": use_black_bg,
    "sphere_dst_dir": st.session_state.get("sphere_dst_dir"),
}

st.divider()

if st.button("Run Pipeline", type="primary"):
    with st.spinner("Processing..."):
        (
            st.session_state.sphere_mask_filename,
            st.session_state.sphere_plot_filename,
            st.session_state.sphere_pipeline_figure,
            st.session_state.sphere_output_text,
        ) = run_pipeline(**kwargs)

if st.session_state.get("sphere_output_text"):
    if st.session_state.get("sphere_output_text") != "Placeholder":
        st.success(st.session_state.get("sphere_output_text"))

    if st.session_state.get("sphere_mask_filename"):
        st.success(
            f"Sphere mask created at: {st.session_state.get('sphere_mask_filename')}"
        )
        st.success(
            f"Sphere plot created at: {st.session_state.get('sphere_plot_filename')}"
        )

    st.markdown("Sphere Mask (Red)")

    st.pyplot(st.session_state.get("sphere_pipeline_figure"))
