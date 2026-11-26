import logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "bids_conversion_pipeline")
)

import streamlit as st

from bids_conversion import run_pipeline
from _general_utils import _check_subjects_visits_file
from _streamlit_utils import StreamlitLogHandler, _select_content

st.set_page_config(layout="centered")

st.title("NIfTI to BIDS Pipeline")
st.divider()

st.markdown("""
Pipeline for converting a raw NIfTI dataset into a BIDS-compliant dataset.\n

### For MPH Study:
- Run 'Participants TSV Pipeline' and 'Add Dosages Pipeline' after conversion.\n

##### Subjects Visits File

Session mapping is based exclusively on the subject ID and dates and available in the subjects visits file.
The dates are standardized, then sorted, and the session IDs are based on the sorting order. Consequently,
if a date is present in the subjects visits file but not found in the subject-specific source directories,
then the date will be ignored but the session labeling will not change.
```python
date_map = {"01": "2000-01-01", "02": "2000-02-02", "03": "2000-03-03"}
# 2000-02-02 is missing in source files
new_date_map = {"01": "2000-01-01", "03": "2000-03-03"}
```

Conversely, if a date is found in the subject-specific source directories but is not in the subjects visits file,
then the date will be used as its session label.
```python
date_map = {"01": "2000-01-01", "02": "2000-03-03"}
# 2000-02-02 found in subject-specific source directories (assuming the directory is not excluded)
new_date_map = {"01": "2000-01-01", "02": "2000-03-03", "2000-02-02": "2000-02-02"}
```
The file subject visits file must contain `participant_id` and `date` columns. Use `NaN` for missing sessions or
exclude rows with no dates or unwanted dates.

| participant_id | date       |
|----------------|------------|
| 101            | 01/02/2022 |
| 101            | NaN        |
| 101            | 03/02/2022 |
| 102            | 01/02/2024 |

To include dosages, add a `dose` column:

| participant_id | date       | dose |
|----------------|------------|------|
| 101            | 01/02/2000 | 0    |
| 101            | 03/02/2000 | 10   |
| 102            | 01/02/2002 | NaN  |
""")

st.divider()
st.markdown("**Required Arguments**")

if st.button(
    "Browse for raw NIfTI directory",
    help="Directory containing the non-BIDS compliant fMRI data.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.src_dir = folder
        st.session_state.raw_subfolders = sorted(
            [
                x
                for x in Path(st.session_state.src_dir).glob("*")
                if x.is_dir() and re.match(r"^\d{5}", x.name)
            ]
        )

if st.session_state.get("src_dir"):
    if st.session_state.raw_subfolders:
        st.success(f"Raw NIfTI Directory: {st.session_state.src_dir}")
    else:
        st.error(
            f"Not a valid raw NIfTI directory (no subjects detected): {st.session_state.src_dir}"
        )

cohort = st.selectbox(
    "Cohort",
    ("Kids", "Adults"),
    help="Determines which scan protocols and tasks are available.",
)

if st.button(
    "Browse for subjects visits file",
    help=(
        "A CSV or Excel file mapping subjects to visit dates and dosages. "
        "Must contain 'participant_id', 'date', and 'dose' columns. "
        "List dates in chronological order per subject and use NaN for missing sessions. "
        "Do not include unwanted subject dates in order to skip them."
    ),
):
    file = _select_content("file")
    if file:
        st.session_state.subjects_visits_file = file
        st.session_state.is_valid_visits_file = _check_subjects_visits_file(
            file, dose_column_required=False, for_app=True, return_boolean=True
        )

if st.session_state.get("subjects_visits_file"):
    if st.session_state.is_valid_visits_file:
        st.success(f"Visits File: {st.session_state.subjects_visits_file}")
    else:
        st.error(f"Invalid visits file: {st.session_state.subjects_visits_file} ")

st.divider()
st.markdown("**Optional Arguments**")

if st.button("Browse for BIDS output directory"):
    folder = _select_content("directory")
    if folder:
        st.session_state.bids_dir = folder

if st.session_state.get("bids_dir"):
    st.success(f"BIDS output: {st.session_state.bids_dir}")

if st.button("Browse for temporary directory"):
    folder = _select_content("directory")
    if folder:
        st.session_state.temp_dir = folder

if st.session_state.get("temp_dir"):
    st.success(f"Temp: {st.session_state.temp_dir}")

if st.session_state.get("src_dir") and st.session_state.get("raw_subfolders"):
    subjects = [
        re.findall(r"\d{5}", x.name)[0]
        for x in st.session_state.get("raw_subfolders")
        if re.findall(r"\d{5}", x.name)
    ]
    subjects = sorted(list(set(subjects)))
    subjects = st.multiselect(
        "Detected subject IDs",
        subjects,
        help="Restrict conversion to specific subjects. Enter IDs without the 'sub-' prefix, separated by commas or spaces.",
    )
else:
    subjects = None

if st.session_state.get("src_dir") and st.session_state.get("raw_subfolders"):
    subfolders = st.session_state.get("raw_subfolders")
    if subjects:
        subfolders = [
            x for x in subfolders if any(sub_id in x.name for sub_id in subjects)
        ]

    exclude_src_folder_names = st.multiselect(
        "Source folders to exclude",
        subfolders,
        help="Select any source folders to skip during conversion.",
    )
else:
    exclude_src_folder_names = None

if st.session_state.get("src_dir") and st.session_state.get("raw_subfolders"):
    nifti_files = sorted(
        [
            x
            for x in Path(st.session_state.src_dir).rglob("*")
            if x.is_file() and str(x).endswith((".nii", ".nii.gz"))
        ]
    )
    if subjects:
        nifti_files = [
            x for x in nifti_files if any(sub_id in x.name for sub_id in subjects)
        ]

    exclude_nifti_filenames = st.multiselect(
        "NIfTI filenames to exclude",
        nifti_files,
        help="Select any NIfTI files to skip during conversion.",
    )
else:
    exclude_nifti_filenames = None

delete_temp_dir = st.checkbox(
    "Delete temporary directory after processing",
    value=True,
    help="If checked, the temporary directory is removed once processing is complete.",
)

create_dataset_metadata = st.checkbox(
    "Create dataset metadata",
    value=True,
    help=(
        "Create the participants TSV and dataset description JSON. "
        "Appends to an existing participants TSV if one is found. "
        "Keep unchecked when running in parallel to avoid race conditions."
    ),
)

add_sessions_tsv = st.checkbox(
    "Add sessions TSV",
    value=True,
    help="Create a sessions TSV file containing the session and scan date for each subject.",
)

kwargs = {
    "src_dir": st.session_state.get("src_dir"),
    "temp_dir": st.session_state.get("temp_dir"),
    "bids_dir": st.session_state.get("bids_dir"),
    "subjects": subjects if subjects else None,
    "exclude_src_folder_names": (
        exclude_src_folder_names if exclude_src_folder_names else None
    ),
    "exclude_nifti_filenames": (
        exclude_nifti_filenames if exclude_nifti_filenames else None
    ),
    "cohort": cohort.lower(),
    "delete_temp_dir": delete_temp_dir,
    "create_dataset_metadata": create_dataset_metadata,
    "add_sessions_tsv": add_sessions_tsv,
    "subjects_visits_file": st.session_state.get("subjects_visits_file"),
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not (st.session_state.get("src_dir") and st.session_state.get("raw_subfolders")):
        st.error("Please select a valid raw NIfTI directory before running.")
    elif not (
        st.session_state.get("subjects_visits_file")
        and st.session_state.get("is_valid_visits_file")
    ):
        st.error("Please upload a valid subjects visits file before running.")
    else:
        status_container = st.empty()
        with status_container.status("Running pipeline...", expanded=True) as status:
            handler = StreamlitLogHandler(status)
            logging.getLogger().addHandler(handler)

            bids_dir = run_pipeline(**kwargs)

            logging.getLogger().removeHandler(handler)

            status.update(
                label=f"BIDS conversion complete. Dataset located at {bids_dir}",
                state="complete",
                expanded=False,
            )
