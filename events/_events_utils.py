import argparse, logging, re, sys, textwrap
from pathlib import Path

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from _streamlit_utils import StreamlitLogHandler, _select_content
from _general_utils import _check_subjects_visits_file


def _get_cmd_args(caller):
    parser = argparse.ArgumentParser(
        description="Create BIDS-compliant event files from neurobehavioral log data."
    )
    parser.add_argument(
        "--log_dir",
        dest="log_dir",
        required=True,
        help="Directory containing the neurobehavioral log files.",
    )

    default_dir = "~/BIDS_Events" if caller == "BIDS Events" else "~/Behavioral_Data."
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=False,
        default=None,
        help=f"Output directory for the generated event files. Defaults to {default_dir}",
    )
    parser.add_argument(
        "--temp_dir",
        dest="temp_dir",
        required=False,
        default=None,
        help="Temporary working directory used during processing. Cleaned up automatically after completion.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=False,
        default="kids",
        choices=["kids", "adults"],
        help="Cohort name. Determines which tasks are available. Default: kids.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        help=(
            "Task name. kids: nback, flanker, mtle, mtlr, princess. "
            "adults: nback, flanker, simplegng, complexgng, mtle, mtlr."
        ),
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        required=False,
        default=None,
        nargs="+",
        help="One or more subject IDs (without the 'sub-' prefix) to restrict processing to.",
    )
    parser.add_argument(
        "--minimum_file_size",
        dest="minimum_file_size",
        required=False,
        default=None,
        help="Minimum file size in bytes. Files smaller than this are skipped. Uses task-specific defaults if not set.",
    )
    parser.add_argument(
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=True,
        type=str,
        help=(
            "Path to a CSV or Excel file mapping subjects to visit dates. "
            "Must contain 'participant_id' and 'date' columns. "
            "Ensure all dates use the same format. "
            "For data from unwanted dates, set to a null value (leave that cell empty) or exclude that row from the data."
        ),
    )

    if caller == "Behavioral Data":
        parser.add_argument(
            "--behavioral_data_file",
            dest="behavioral_data_file",
            required=False,
            default=None,
            help="The path to the behavioral data (if exists) to append new data to.",
        )

    parser.add_argument(
        "--delete_temp_dir",
        dest="delete_temp_dir",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Delete the temporary directory after processing. Default: True.",
    )
    parser.add_argument(
        "--exclude_filenames",
        dest="exclude_filenames",
        required=False,
        default=None,
        nargs="+",
        help="Filenames to exclude from processing (e.g., 101_nback.txt 102_flanker.xls).",
    )

    return parser


def _app(caller, pipeline):
    st.set_page_config(layout="centered")

    if caller == "BIDS Events":
        st.title("BIDS Events File Pipeline")
        note = textwrap.dedent("""
                Pipeline for creating BIDS-compliant timing files.\n

                Example output:

                Filename: "sub-101_ses-01_task-nback_run-01.tsv"
                Contents (timing is in seconds):
                | onset | duration  | trial_type | accuracy  |
                |-------|-----------|------------|-----------|
                | 0     | 1         | A          | correct   |
                | 4     | 1         | B          | correct   |
                | 6     | 1         | A          | incorrect |
                | 8     | 1         | B          | correct   |

                *onset is relative to the NIfTI image where time 0 is the first
                volume of the NIfTI image (absolute task timing is subtracted by the time of the trigger,
                assumed to be the start of the first non-dummy volume); hence an onset of 4 seconds (assuming the TR is 2)
                means that the block/event occured on the second time index (3th volume) for 0-based languages (counting
                starts at 0; e.g., Python, C). For 1-based languages (counting starts at 1; e.g., R, Matlab),
                the second time index would be the second volume, so one must be added
                for correct mapping.*\n

                **Note:**\n
                - For data from unwanted dates, set to a NULL value (leave that cell empty) or exclude that row from the subjects visits file.
            """)
    else:
        st.title("Behavioral Data Pipeline")
        note = textwrap.dedent("""
                Pipeline for creating computing task accuracy and reaction time from block and event-related tasks.\n

                Example output:

                | participant_id | session_id  | A_accuracy | A_average_reaction_time  |
                |----------------|-------------|------------|--------------------------|
                | 101            | ses-01      | 1          | 0.5                      |
                | 101            | ses-02      | 0.96       | 0.3                      |
                | 102            | ses-01      | 0.70       | 0.2                      |
                | 102            | ses-02      | 0.80       | 0.7                      |

                **Note:**\n
                - For data from unwanted dates, set to a NULL value (leave that cell empty) or exclude that row from the subjects visits file.
            """)

    st.divider()

    st.markdown(note)

    if caller == "BIDS Events":
        st.markdown(
            "\n**Use the 'Move Files Pipeline' to automatically move events TSV files to thier respective subjects directory within the BIDS directory.**"
        )

    st.divider()
    st.markdown("**Required Arguments**")

    if st.button(
        "Browse for log directory",
        help="Directory containing the log files for the specified task.",
    ):
        folder = _select_content("directory")
        if folder:
            st.session_state.log_dir = folder
            st.session_state.log_files = sorted(
                [
                    x
                    for x in Path(st.session_state.log_dir).glob("*")
                    if x.is_file() and re.findall(r"\d{5}", x.name)
                ]
            )

    if st.session_state.get("log_dir"):
        if st.session_state.log_files:
            st.success(f"Source: {st.session_state.log_dir}")
        else:
            st.error(f"Not a valid log directory: {st.session_state.get('log_dir')}")

    cohort = st.selectbox(
        "Cohort", ("Kids", "Adults"), help="Determines which tasks are available."
    )

    if cohort == "Kids":
        valid_tasks = ("nback", "princess", "flanker", "mtle", "mtlr")
    else:
        valid_tasks = ("nback", "flanker", "simplegng", "complexgng", "mtle", "mtlr")

    task = st.selectbox(
        "Task",
        valid_tasks,
        help="The neurobehavioral task to generate event files for.",
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

    if st.button("Browse for output directory"):
        folder = _select_content("directory")
        if folder:
            st.session_state.dst_dir = folder

    if st.session_state.get("dst_dir"):
        st.success(f"Output: {st.session_state.dst_dir}")

    if caller == "Behavioral Data" and st.button(
        "Browse behavioral data file",
        help="The path to the behavioral data (if exists) to append new data to.",
    ):
        file = _select_content("file")
        if file:
            st.session_state.behavioral_data_file = file

    if st.session_state.get("behavioral_data_file"):
        st.success(f"Behavioral File: {st.session_state.behavioral_data_file}")

    if st.button("Browse for temporary directory"):
        folder = _select_content("directory")
        if folder:
            st.session_state.temp_dir = folder

    if st.session_state.get("temp_dir"):
        st.success(f"Temp: {st.session_state.temp_dir}")

    if st.session_state.get("log_dir") and st.session_state.get("log_files"):
        subjects = [
            re.findall(r"\d{5}", x.name)[0]
            for x in st.session_state.get("log_files")
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

    minimum_file_size = st.number_input(
        "Minimum file size (bytes)",
        min_value=0,
        value=0,
        help="Files smaller than this are skipped. Leave at 0 to use the task-specific default.",
    )

    delete_temp_dir = st.checkbox(
        "Delete temporary directory after processing",
        value=True,
        help="If checked, the temporary directory is removed once processing is complete.",
    )

    if st.session_state.get("log_dir") and st.session_state.get("log_files"):
        files = st.session_state.get("log_files")
        if subjects:
            files = [x for x in files if any(sub_id in x.name for sub_id in subjects)]

        exclude_filenames = st.multiselect(
            "Files to exclude",
            files,
            help="Upload any log files that should be excluded from processing.",
        )
    else:
        exclude_filenames = None

    kwargs = {
        "log_dir": st.session_state.get("log_dir"),
        "dst_dir": st.session_state.get("dst_dir"),
        "temp_dir": st.session_state.get("temp_dir"),
        "delete_temp_dir": delete_temp_dir,
        "cohort": cohort.lower(),
        "task": task,
        "subjects": subjects if subjects else None,
        "subjects_visits_file": st.session_state.get("subjects_visits_file"),
        "behavioral_data_file": st.session_state.get("behavioral_data_file"),
        "minimum_file_size": minimum_file_size or None,
        "exclude_filenames": exclude_filenames if exclude_filenames else None,
    }
    if caller == "BIDS Events":
        del kwargs["behavioral_data_file"]

    st.divider()
    if st.button("Run Pipeline", type="primary"):
        if not (st.session_state.get("log_dir") and st.session_state.get("log_files")):
            st.error("Please select a valid log directory before running.")
        elif not (
            st.session_state.get("subjects_visits_file")
            and st.session_state.get("is_valid_visits_file")
        ):
            st.error("Please upload a subjects visits file before running.")
        else:
            status_container = st.empty()
            with status_container.status(
                "Running pipeline...", expanded=True
            ) as status:

                handler = StreamlitLogHandler(status)
                logging.getLogger().addHandler(handler)

                dst_dir, log_files = pipeline(**kwargs, caller=caller)

                logging.getLogger().removeHandler(handler)

                file_type_str = (
                    "Event files" if caller == "BIDS Events" else "Behavioral CSV"
                )

                if log_files:
                    status.update(
                        label=f"{file_type_str} for the {task} task ({cohort} cohort) created in: {dst_dir}",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(
                        label=f"No log files found for the {task} task in: {st.session_state.get('log_dir')}",
                        state="error",
                        expanded=False,
                    )
