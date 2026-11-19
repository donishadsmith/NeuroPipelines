import os, re, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np, pandas as pd

from bidsaid.parsers import (
    load_presentation_log,
    load_eprime_log,
    convert_edat3_to_text,
    get_presentation_log_date,
)
from bidsaid.events import (
    PresentationBlockExtractor,
    PresentationEventExtractor,
    EPrimeBlockExtractor,
    add_cue_timing,
)
from bidsaid.io import _copy_file
from bidsaid.logging import setup_logger

from _general_utils import (
    SubjectsVisitsFileError,
    _get_dataframe,
    _get_subject_visits,
    _resolve_directories,
    _standardize_dates,
    _standardize_participant_ids,
)

LGR = setup_logger(__name__)

FILE_SIZE_MINIMUM_KB = {
    "kids": {"flanker": 40, "mtle": 5, "mtlr": 5, "princess": 50, "nback": 40},
    "adults": {
        "flanker": 40,
        "simplegng": 20,
        "complexgng": 20,
        "nback": 12,
        "mtle": 8,
        "mtlr": 8,
    },
}


def _filter_log_files(log_files, subjects, exclude_filenames):
    log_files = list(log_files)
    if subjects:
        log_files = [
            log_file
            for log_file in log_files
            if any(subject in log_file.name for subject in subjects)
        ]

    if exclude_filenames:
        exclude_filenames = [Path(file).name for file in exclude_filenames]
        log_files = [
            log_file for log_file in log_files if log_file.name not in exclude_filenames
        ]

    log_files = list(log_files)
    if not log_files:
        LGR.warning(
            "No log files selected, check if ``log_dir`` or ``subjects`` are accurate."
        )

    return log_files


def _get_minimum_file_size(cohort, task, minimum_file_size):
    if not minimum_file_size:
        minimum_file_size = FILE_SIZE_MINIMUM_KB[cohort][task] * 1024

    return minimum_file_size


def _copy_event_files(log_dir, temp_dir, cohort, task, minimum_file_size):
    for event_file in log_dir.glob("*"):
        minimum_file_size = _get_minimum_file_size(cohort, task, minimum_file_size)
        if os.path.getsize(event_file) < minimum_file_size:
            LGR.warning(
                f"The following file is smaller than {minimum_file_size} bytes "
                f"and will not be copied to the temporary directory: {event_file}."
            )
            continue

        _copy_file(event_file, temp_dir / event_file.name, remove_src_file=False)


def _check_subjects_visits_file(subjects_visits_file: str | Path) -> None:
    required_colnames = ["participant_id", "date"]
    colnames = _get_dataframe(subjects_visits_file).columns
    if not all(required_colname in colnames for required_colname in required_colnames):
        raise SubjectsVisitsFileError(
            f"The following columns are required in {subjects_visits_file}: "
            f"{required_colnames}."
        )


def _get_eprime_session(
    subject_id,
    edat_file,
    subjects_visits_df,
):
    dst_path = convert_edat3_to_text(edat_file)
    curr_log_date = load_eprime_log(dst_path)["SessionDate"].tolist()[0]
    curr_log_date = str(
        pd.to_datetime(
            [curr_log_date],
        )[0]
    ).split()[0]
    dst_path.unlink()

    return map_date_to_session(
        subject_id,
        subjects_visits_df,
        curr_log_date,
    )


def _get_presentation_session(
    subject_id,
    log_file,
    subjects_visits_df,
):
    curr_log_date = str(
        pd.to_datetime(
            [get_presentation_log_date(log_file)],
        )[0]
    ).split()[0]

    return map_date_to_session(
        subject_id,
        subjects_visits_df,
        curr_log_date,
    )


def map_date_to_session(subject_id, subjects_visits_df, curr_log_date):
    date_to_session_map = _get_subject_visits(subject_id, subjects_visits_df)
    visit_session_map = {}

    if date_to_session_map:
        visit_session_map.update({v: k for k, v in date_to_session_map.items()})
        date_in_map = curr_log_date in visit_session_map
    else:
        date_in_map = False

    if not date_in_map:
        LGR.warning(
            f"Subject {subject_id} does not have the following date: {curr_log_date}. "
            "The date will be used as the session label."
        )

    return (
        (visit_session_map[curr_log_date], curr_log_date)
        if visit_session_map and date_in_map
        else (curr_log_date.replace("/", "-"), curr_log_date.replace("/", "-"))
    )


def save_df_as_tsv(event_df, dst_dir, subject_id, session_id, date, caller, task):
    date_string = f"date-{date}_" if caller == "Behavioral Data" else ""
    tsv_filename = (
        dst_dir
        / f"sub-{subject_id}_ses-{session_id}_{date_string}task-{task}_run-01_events.tsv"
    )

    event_df.to_csv(tsv_filename, sep="\t", index=False)


# See: https://www.neurobs.com/pres_docs/html/03_presentation/03_stimulus_presentation/01_trials/03_fmri_mode/index.html
# Stimulus duration is 800 ms
def _create_flanker_events_files(
    temp_dir, dst_dir, subjects, subjects_visits_df, exclude_filenames, caller
):
    excel_files = _filter_log_files(temp_dir.glob("*.xls"), subjects, exclude_filenames)
    for excel_file in excel_files:
        try:
            extractor = PresentationEventExtractor(
                excel_file,
                convert_to_seconds=["Time", "Duration"],
                trial_types=".*(left|right)$",
                scanner_event_type="Pulse",
                scanner_trigger_code="99",
            )

            events = {}
            events["onset"] = extractor.extract_onsets()
            events["duration"] = extractor.extract_durations()
            trial_types = extractor.extract_trial_types()
        except:
            LGR.exception(
                f"An event file could not be created for the following file: {excel_file}",
                exc_info=True,
            )
            continue

        # Separate trial name from arrow direction
        info = []
        for trial_type in trial_types:
            trial_name = trial_type.removesuffix("left").removesuffix("right")
            arrow_dir = trial_type.removeprefix(trial_name)
            info.append((trial_name, arrow_dir))

        trial_types, arrow_dirs = zip(*info)

        events["trial_type"] = trial_types
        events["central_arrow_direction"] = arrow_dirs
        events["response"] = extractor.extract_responses()
        events["accuracy"] = extractor.extract_accuracies(
            {
                "hit": "correct",
                "miss": "incorrect",
                "incorrect": "incorrect",
                "other": "correct",
                "false_alarm": "incorrect",
                "false alarm": "incorrect",
            }
        )
        events["reaction_time"] = extractor.extract_reaction_times()
        event_df = pd.DataFrame(events)

        # Specific accuracy case for miss
        event_df.loc[
            (event_df["trial_type"] == "nogo") & (event_df["response"] == "miss"),
            "accuracy",
        ] = "correct"

        event_df["trial_type_accuracy"] = (
            event_df["trial_type"].astype(str) + "_" + event_df["accuracy"].astype(str)
        )

        # Getting subject ID and organising files to get subject ID
        subject_id = re.search(r"(\d+).*_", excel_file.name).group(1)
        session_id, date = _get_presentation_session(
            subject_id,
            excel_file,
            subjects_visits_df,
        )

        save_df_as_tsv(
            event_df, dst_dir, subject_id, session_id, date, caller, task="flanker"
        )

    return excel_files


# Nogo trials with false alarm (indicating a button press) with an even number of
# preceeding go trials is considered correct
# Since 0 is an even number, also assuming the abscence of preceeding go trials
# means the nogo trial should be responsed to.
def _determine_complexgng_nogo_accuracy(event_df):
    nogo_indices = event_df[
        event_df["trial_type"].str.fullmatch(r".*Nogo.*")
    ].index.tolist()
    trial_responses = (
        event_df.loc[nogo_indices, "response"]
        .apply(lambda x: x.replace(" ", "_"))
        .values
    )
    is_false_alarm = np.where(np.array(trial_responses) == "false_alarm", True, False)
    n_preceeding_go_trials = np.array(
        [
            int(nogo_string.split(";")[-1])
            for nogo_string in event_df.loc[nogo_indices, "trial_type"].values
        ]
    )
    has_even_preceeding_go_trials = np.where(
        n_preceeding_go_trials % 2 == 0, True, False
    )

    return np.where(
        has_even_preceeding_go_trials & is_false_alarm
        | ~has_even_preceeding_go_trials & ~is_false_alarm,
        "correct",
        "incorrect",
    )


# https://pmc.ncbi.nlm.nih.gov/articles/PMC4517984/
# Primary difference is that the simplegng is just a regular go-no go task
# (go when green stop when red)
# The complex go-no requires a press on go and a press only on nogo trials that are
# preceeded by an even number of go trials
# Stimulus duration is 300 ms
# Based on Presentation experimental design, the nogo has the form
# 1;{stimuli_presentation_number};Nogo;;{number_of_preceeding_gos}
# Important: based on the Philips exam cards these tasks use ascending slice timing with 0 slice gap, so
# there will be cross slice excitation. In addition, these tasks use a TR of 2.5 and have 98
# volumes so collect 245 seconds of data. The Presentation tasks last for ~260 seconds, so the BIDS events
# for this file is truncated to drop trials at the end if the onset and duration exceeds the 245 second limit
def _create_gng_events_files(
    temp_dir, dst_dir, task, subjects, subjects_visits_df, exclude_filenames, caller
):
    task_name = "SimpleRepeatGNG" if task == "simplegng" else "RepeatSimpleGNG"
    log_files = _filter_log_files(
        temp_dir.glob(f"*{task_name}*.log"), subjects, exclude_filenames
    )
    for log_file in log_files:
        trial_types = ("Go", "Nogo") if task == "simplegng" else (".*Go.*", ".*Nogo.*")
        trial_column_name = "Trial_Type(str)" if task == "simplegng" else "Code"
        try:
            extractor = PresentationEventExtractor(
                log_file,
                convert_to_seconds=["Time", "Duration"],
                trial_types=trial_types,
                initial_column_headers=["Subject", "Trial"],
                scanner_event_type="Picture",
                trial_column_name=trial_column_name,
                scanner_trigger_code="scanner trigger",
            )
        except:
            LGR.exception(
                f"An event file could not be created for the following file: {log_file}",
                exc_info=True,
            )
            continue

        events = {}
        events["onset"] = extractor.extract_onsets()
        events["duration"] = extractor.extract_durations()
        events["trial_type"] = extractor.extract_trial_types()
        events["response"] = extractor.extract_responses()
        events["reaction_time"] = extractor.extract_reaction_times()

        if task == "simplegng":
            events["accuracy"] = extractor.extract_accuracies(
                {
                    "hit": "correct",
                    "miss": "incorrect",
                    "other": "correct",
                    "false_alarm": "incorrect",
                    "false alarm": "incorrect",
                }
            )
            event_df = pd.DataFrame(events)
        else:
            events["accuracy"] = extractor.extract_accuracies(
                {
                    "hit": "correct",
                    "miss": "incorrect",
                    "other": "NaN",
                    "false_alarm": "NaN",
                    "false alarm": "NaN",
                }
            )
            event_df = pd.DataFrame(events)
            event_df.loc[
                event_df["trial_type"].str.fullmatch(r".*Nogo.*"), "accuracy"
            ] = _determine_complexgng_nogo_accuracy(event_df)
            event_df.loc[
                event_df["trial_type"].str.fullmatch(r".*Nogo.*"),
                "n_preceeding_go_trials",
            ] = pd.NA

            mask = event_df["trial_type"].str.fullmatch(r".*Nogo.*")
            final_number = [
                int(re.search(r"(\d+(;)?)$", x).group(1).rstrip(";"))
                for x in events["trial_type"]
            ]
            event_df.loc[mask, "n_preceeding_go_trials"] = np.array(final_number)[mask]
            event_df.loc[
                event_df["trial_type"].str.fullmatch(r".*Go.*"), "trial_type"
            ] = "Go"
            event_df.loc[
                event_df["trial_type"].str.fullmatch(r".*Nogo.*"), "trial_type"
            ] = "Nogo"

        event_df["trial_type"] = event_df["trial_type"].apply(lambda x: x.lower())
        event_df["trial_type_accuracy"] = (
            event_df["trial_type"].astype(str) + "_" + event_df["accuracy"].astype(str)
        )

        # Drop if onset + duration exceeds 245 seconds
        event_df = event_df[(event_df["onset"] + event_df["duration"]) <= 245]

        # Getting subject ID and organising files to get subject ID
        subject_id = re.search(r"(\d+).*-", log_file.name).group(1)
        session_id, date = _get_presentation_session(
            subject_id,
            log_file,
            subjects_visits_df,
        )

        save_df_as_tsv(event_df, dst_dir, subject_id, session_id, date, caller, task)

    return log_files


def _create_nback_eprime_events_files(
    temp_dir, dst_dir, subjects, subjects_visits_df, exclude_filenames, caller
):
    edat_files = _filter_log_files(
        temp_dir.glob("*.edat3"), subjects, exclude_filenames
    )
    for edat_file in edat_files:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmpfile:
            pass

        try:
            csv_path = convert_edat3_to_text(edat_file, dst_path=tmpfile.name)

            input_df = pd.read_csv(csv_path, sep=",")
            input_df["StimDisplay.OffsetTime"] = input_df[
                "StimDisplay.OnsetTime"
            ].tolist() + input_df["StimDisplay.OnsetToOnsetTime"].to_numpy(copy=True)
            input_df["Procedure[Block]"] = input_df["Procedure[Block]"].fillna("Rest")
            input_df["Procedure[Block]"] = input_df["Procedure[Block]"].map(
                {
                    "ExpBloc": "1-back",
                    "ContBloc": "center",
                    "Exp2Bloc": "2-back",
                    "Rest": "Rest",
                }
            )
            input_df.loc[input_df.index[-1], "Procedure[Block]"] = "Quit"

            try:
                extractor = EPrimeBlockExtractor(
                    input_df,
                    onset_column_name="StimDisplay.OnsetTime",
                    procedure_column_name="Procedure[Block]",
                    trial_types=("1-back", "center", "2-back"),
                    convert_to_seconds=[
                        "StimDisplay.OnsetTime",
                        "StimDisplay.OffsetTime",
                        "StimDisplay.RTTime",
                    ],
                    rest_block_codes="Rest",
                    quit_code="Quit",
                    rest_code_pattern="variable",
                )
            except:
                LGR.exception(
                    f"An event file could not be created for the following file: {edat_file}",
                    exc_info=True,
                )
                continue

            events = {}
            # No onset column timing infomation, make assumption that as soon as scanner started
            # immediately starts at the first rest block which occures 16 seconds prior to the
            # first experimental block
            first_stim_onset_time = (
                input_df["StimDisplay.OnsetTime"].dropna(inplace=False).tolist()[0]
                / 1e3
            )
            scanner_onset_time = first_stim_onset_time - 16

            events["onset"] = extractor.extract_onsets(
                scanner_start_time=scanner_onset_time
            )
            events["duration"] = extractor.extract_durations(
                offset_column_name="StimDisplay.OffsetTime"
            )
            events["trial_type"] = extractor.extract_trial_types()
            events["accuracy"] = extractor.extract_mean_accuracies(
                subject_response_column="StimDisplay.RESP",
                correct_response_column="StimDisplay.CRESP",
                response_required_only=True,
            )
            events["reaction_time"] = extractor.extract_mean_reaction_times(
                reaction_time_column_name="StimDisplay.RTTime",
            )
            events["response_count"] = extractor.extract_response_counts(
                reaction_time_column_name="StimDisplay.RTTime",
            )
            event_df = pd.DataFrame(events)

            event_df["duration"] = event_df["duration"].apply(
                lambda x: x if not np.isnan(x) else 34.0
            )

            # Split cue from block, which is 2 seconds before each stimulus
            event_df = add_cue_timing(event_df, cue_duration=2)
            event_df["trial_type"] = event_df["trial_type"].replace(
                {
                    "1-back_cue": "nback_cue",
                    "2-back_cue": "nback_cue",
                    "center_cue": "nback_cue",
                }
            )

            subject_id = re.search(r"(\d+)-\d+", edat_file.name).group(1)
            session_id, date = _get_eprime_session(
                subject_id,
                edat_file,
                subjects_visits_df,
            )

            save_df_as_tsv(
                event_df, dst_dir, subject_id, session_id, date, caller, task="nback"
            )
        finally:
            csv_path.unlink()

    return edat_files


def _create_nback_presentation_events_files(
    temp_dir, dst_dir, subjects, subjects_visits_df, exclude_filenames, caller
):
    text_files = _filter_log_files(temp_dir.glob("*.txt"), subjects, exclude_filenames)
    for text_file in text_files:
        try:
            extractor = PresentationBlockExtractor(
                text_file,
                convert_to_seconds=["Time", "Duration"],
                trial_types=["0back", "2back"],
                scanner_event_type="Pulse",
                scanner_trigger_code="30",
            )
        except:
            LGR.exception(
                f"An event file could not be created for the following file: {text_file}",
                exc_info=True,
            )
            continue

        events = {}
        events["onset"] = extractor.extract_onsets()
        events["duration"] = extractor.extract_durations()
        events["trial_type"] = extractor.extract_trial_types()

        # Ensure "other" not counted, for mean reaction time and response counts
        # if one of the keys in the map is NaN it is ignored completely
        response_map = {
            "hit": 1,
            "miss": 0,
            "other": float("nan"),
            "incorrect": 0,
            "correct": 1,
        }
        events["accuracy"] = extractor.extract_mean_accuracies(
            response_map=response_map
        )
        events["reaction_time"] = extractor.extract_mean_reaction_times(
            response_map=response_map
        )
        events["response_count"] = extractor.extract_response_counts(
            response_map=response_map
        )

        event_df = pd.DataFrame(events)
        event_df["trial_type"] = event_df["trial_type"].replace(
            {
                "0back": "0-back",
                "2back": "2-back",
            }
        )

        subject_id = re.search(r"^(\d+).*_", text_file.name).group(1)
        session_id, date = _get_presentation_session(
            subject_id,
            text_file,
            subjects_visits_df,
        )

        save_df_as_tsv(
            event_df, dst_dir, subject_id, session_id, date, caller, task="nback"
        )

    return text_files


def _create_mtl_events_files(
    temp_dir,
    dst_dir,
    subjects,
    cohort,
    task,
    subjects_visits_df,
    exclude_filenames,
    caller,
):
    # MTLE and MTLR are separate tasks but can be processed in one function
    filename = "_PEARenc" if task == "mtle" else "_PEARret"
    task_name = "indoor" if task == "mtle" else "seen"
    suffix = "encoding" if task_name == "indoor" else "retrieval"
    excel_files = _filter_log_files(
        temp_dir.glob(f"*{filename}*.xls"), subjects, exclude_filenames
    )
    for excel_file in excel_files:
        input_df = load_presentation_log(excel_file)
        # Add quit code, some log files did not record the quit event type
        if not input_df[input_df["Event Type"] == "Quit"].empty:
            input_df.loc[input_df["Event Type"] == "Quit", "Code"] = "quit"

        try:
            extractor = PresentationBlockExtractor(
                input_df,
                convert_to_seconds=["Time", "Duration"],
                trial_types=(task_name),
                scanner_event_type="Pulse",
                scanner_trigger_code="30",
                rest_block_codes="rest",
                quit_code="quit",
                split_cue_from_block=True,
            )

            events = {}
            events["onset"] = extractor.extract_onsets()
            durations = extractor.extract_durations()
            # Fix for those that do not have the quit event type
            durations[-1] = durations[-1] if durations[-1] != 0 else 20.0
            events["duration"] = durations
            events["trial_type"] = extractor.extract_trial_types()
        except:
            LGR.exception(
                f"An event file could not be created for the following file: {excel_file}",
                exc_info=True,
            )
            continue

        # Ensure "other" not counted, for mean reaction time and response counts
        # if one of the keys in the map is NaN it is ignored completely
        response_map = {
            "hit": 1,
            "miss": 0,
            "other": float("nan"),
            "incorrect": 0,
            "correct": 1,
        }
        events["accuracy"] = extractor.extract_mean_accuracies(response_map)
        events["reaction_time"] = extractor.extract_mean_reaction_times(
            response_map=response_map
        )
        events["response_count"] = extractor.extract_response_counts(
            response_map=response_map
        )

        event_df = pd.DataFrame(events)
        event_df["trial_type"] = event_df["trial_type"].replace(
            {
                "indoor_cue": "mtle_cue",
                "seen_cue": "mtlr_cue",
            }
        )

        subject_id = re.search(r"(\d+).*_", excel_file.name).group(1)
        session_id, date = _get_presentation_session(
            subject_id,
            excel_file,
            subjects_visits_df,
        )

        if cohort == "adults":
            # Pattern seems to be the first for blocks have the aversive images and last four are neutral
            event_df.loc[event_df["trial_type"] == task_name, "trial_type"] = [
                f"aversive_{suffix}"
            ] * 4 + [f"neutral_{suffix}"] * 4
        else:
            n = len(
                event_df.loc[event_df["trial_type"] == task_name, "trial_type"].tolist()
            )
            # If 8 blocks that is the adult task that somehow made its way into the kids source data due to
            # similar IDs
            if n > 4:
                continue

            event_df.loc[event_df["trial_type"] == task_name, "trial_type"] = [
                f"neutral_{suffix}"
            ] * n

            save_df_as_tsv(
                event_df, dst_dir, subject_id, session_id, date, caller, task
            )

    return excel_files


def _create_princess_events_files(
    temp_dir, dst_dir, subjects, subjects_visits_df, exclude_filenames, caller
):
    edat_files = _filter_log_files(
        temp_dir.glob("*.edat3"), subjects, exclude_filenames
    )
    for edat_file in edat_files:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmpfile:
            pass

        try:
            csv_path = convert_edat3_to_text(edat_file, dst_path=tmpfile.name)

            # Moving cue time to onset column
            input_df = pd.read_csv(csv_path, sep=",")
            input_df.loc[
                ~input_df["indicatie.OnsetTime"].isna(), "dagnacht.OnsetTime"
            ] = input_df.loc[
                ~input_df["indicatie.OnsetTime"].isna(), "indicatie.OnsetTime"
            ]

            # Based on original paper, trials blocks should be ~52 seconds each
            # Still derive to check if timing is padded
            # Note, offset times were not recorded but paper states trials include the feedback
            input_df["feedback.OffsetTime"] = (
                input_df["feedback.OnsetTime"].to_numpy(copy=True)
                + input_df["feedback.OnsetToOnsetTime"].tolist()
            )

            huizens = []
            for huizen in input_df["huizen"].astype(str).tolist():
                huizen = str(huizen).removesuffix(".bmp")
                if huizen[-1].isdigit():
                    huizen = huizen[:-1]
                huizens.append(huizen)

            input_df["huizen"] = huizens

            dutch_to_english = {"dag": "day", "nacht": "night", "dagnacht": "daynight"}
            input_df["huizen"] = input_df["huizen"].replace(dutch_to_english)

            try:
                extractor = EPrimeBlockExtractor(
                    input_df,
                    onset_column_name="dagnacht.OnsetTime",
                    procedure_column_name="huizen",
                    trigger_column_name="eind.OnsetTime",
                    trial_types=dutch_to_english.values(),
                    convert_to_seconds=[
                        "dagnacht.OnsetTime",
                        "eind.OnsetTime",
                        "feedback.OffsetTime",
                        "dagnacht.RT",
                    ],
                    split_cue_from_block=False,
                )
            except:
                LGR.exception(
                    f"An event file could not be created for the following file: {edat_file}",
                    exc_info=True,
                )
                continue

            events = {}
            # Best guess of scanner time would be the first fixpunt which appears after
            # scanner sends trigger, two image displays appear but only second image time recorded
            # first fixpunt has a duration of ~60 seconds, the second appears three seconds
            # before cue
            scanner_start_time = (
                input_df["fixpunt.OnsetTime"].dropna().unique()[0] / 1e3 - 60
            )
            events["onset"] = extractor.extract_onsets(
                scanner_start_time=scanner_start_time
            )
            events["duration"] = extractor.extract_durations(
                offset_column_name="feedback.OffsetTime"
            )
            trial_name_dict = {
                "daynight": "switch",
                "day": "nonswitch",
                "night": "nonswitch",
            }
            events["trial_type"] = [
                trial_name_dict[trial_type]
                for trial_type in extractor.extract_trial_types()
            ]
            events["block_cue"] = extractor.extract_trial_types()
            events["accuracy"] = extractor.extract_mean_accuracies(
                subject_response_column="dagnacht.RESP",
                correct_response_column="dagnacht.CRESP",
                response_required_only=True,
            )
            events["reaction_time"] = extractor.extract_mean_reaction_times(
                reaction_time_column_name="dagnacht.RT",
            )
            events["response_count"] = extractor.extract_response_counts(
                reaction_time_column_name="dagnacht.RT",
            )

            event_df = pd.DataFrame(events)
            # For some subjects, the recorded timing at the end may be about 1 second out of bounds
            # This task doesn't have a rest block or crosshair at the end
            event_df["duration"] = 52
            event_df["onset"] = event_df["onset"].apply(lambda x: int(np.floor(x)))
            # Needs to be clipped. Timing ends at 472 or 473 for subjects but AFNI's timing_tool.py
            # results in a rounding, out-of-bounds error because the final switch block is not
            # proceeded by a rest block or several seconds of a crosshair
            event_df.loc[event_df.index[-1], "onset"] = min(
                event_df.loc[event_df.index[-1], "onset"], 471
            )

            subject_id = re.search(r"(\d+)-\d+\.edat3$", edat_file.name).group(1)
            session_id, date = _get_eprime_session(
                subject_id,
                edat_file,
                subjects_visits_df,
            )

            # Note, duration of all trials in block without cue is 48-50 seconds and duration of cue is ~3
            # seconds with above implementation: Paper says each block has 8 trials which lasts 6500 ms:
            # Paper: https://pubmed.ncbi.nlm.nih.gov/20604616/
            save_df_as_tsv(
                event_df, dst_dir, subject_id, session_id, date, caller, task="princess"
            )
        finally:
            csv_path.unlink()

    return edat_files


def _strip_entity(subjects):
    return [str(subject).removeprefix("sub-") for subject in subjects]


EVENTS_FUNC = {
    "kids": {
        "flanker": _create_flanker_events_files,
        "nback": _create_nback_eprime_events_files,
        "mtle": _create_mtl_events_files,
        "mtlr": _create_mtl_events_files,
        "princess": _create_princess_events_files,
    },
    "adults": {
        "flanker": _create_flanker_events_files,
        "nback": _create_nback_presentation_events_files,
        "simplegng": _create_gng_events_files,
        "complexgng": _create_gng_events_files,
        "mtle": _create_mtl_events_files,
        "mtlr": _create_mtl_events_files,
    },
}


def run_pipeline(
    log_dir,
    dst_dir,
    temp_dir,
    delete_temp_dir,
    task,
    cohort,
    subjects,
    minimum_file_size,
    subjects_visits_file,
    exclude_filenames,
    caller,
):
    if task not in EVENTS_FUNC[cohort]:
        raise ValueError(
            f"`task` must be one of the following: {EVENTS_FUNC[cohort].keys()}"
        )

    LGR.info("Validating subjects visits CSV...")
    _check_subjects_visits_file(subjects_visits_file)

    LGR.info("Resolving directories...")
    log_dir = Path(log_dir)
    dst_dir, temp_dir = _resolve_directories(dst_dir, temp_dir, caller)

    if subjects:
        subjects = _strip_entity(subjects)
        subjects = [re.findall(r"\d{5}", x)[0] for x in subjects]

    kwargs = {
        "temp_dir": temp_dir,
        "dst_dir": dst_dir,
        "subjects": subjects,
        "subjects_visits_df": _standardize_participant_ids(
            _standardize_dates(_get_dataframe(subjects_visits_file), sort_data=True)
        ),
        "exclude_filenames": exclude_filenames,
        "caller": caller,
    }
    if task in ["mtle", "mtlr", "simplegng", "complexgng"]:
        kwargs.update({"task": task})

    if task in ["mtle", "mtlr"]:
        kwargs.update({"cohort": cohort})

    log_files = []
    try:
        LGR.info("Copying data to temporary directory...")
        _copy_event_files(log_dir, temp_dir, cohort, task, minimum_file_size)
        LGR.info("Creating BIDS events files...")
        log_files = EVENTS_FUNC[cohort][task](**kwargs)
    finally:
        if delete_temp_dir:
            shutil.rmtree(temp_dir)

    return dst_dir, log_files
