import json, sys
from dataclasses import dataclass
from pathlib import Path

import bids, numpy as np, matplotlib.pyplot as plt
from bidsaid._helpers import iterable_to_str
from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger
from bidsaid.qc import compute_n_dummy_scans, create_censor_mask

from _denoising import get_acompcor_component_names
from _utils import VALID_TASK_NAMES, _get_dataframe, embed_img
from _report import HTMLReport

LGR = setup_logger(__name__)


@dataclass(frozen=True)
class SessionFiles:
    confounds_tsv_file: str
    confounds_json_file: str | None
    event_file: str
    mask_file: str
    nifti_file: str


@dataclass(frozen=True)
class CensorInfo:
    n_dummy_scans: int
    n_censored: int
    n_total: int
    percent_censored: float
    dummy_method: str
    mean_fd_before_censoring: float
    mean_fd_after_censoring: float


EVENT_RELATED_TASKS = ["flanker", "simplegng", "complexgng"]

# Using constant durations instead of BIDS one, which have small
# stimulus presentation delays
# cue has the same duration for all three tasks but in the
# code for clarity
CONDITION_DURATIONS = {
    "kids": {
        "flanker": 0.8,
        "nback": 32,
        "princess": 52,
        "mtle": 18,
        "mtlr": 18,
        "nback_cue": 2,
        "mtle_cue": 2,
        "mtlr_cue": 2,
    },
    "adults": {
        "flanker": 0.8,
        "nback": 30,
        "mtle": 18,
        "mtlr": 18,
        "simplegng": 0.3,
        "complexgng": 0.3,
        "mtle_cue": 2,
        "mtlr_cue": 2,
    },
}


class MissingFileError(Exception):
    pass


class InvalidRunError(Exception):
    pass


def validate_first_level_inputs(
    dst_dir, bids_dir, deriv_dir, cohort, task, subject, analysis_type
):
    report_dir = Path(dst_dir) / "reports" / "first_level"
    report_dir.mkdir(parents=True, exist_ok=True)

    if task not in VALID_TASK_NAMES[cohort]:
        LGR.warning(
            f"The task must be one of the following: {iterable_to_str(VALID_TASK_NAMES[cohort])}"
        )
        sys.exit(1)

    layout = bids.BIDSLayout(bids_dir, derivatives=deriv_dir or True)

    sessions = layout.get(
        subject=subject, task=task, target="session", return_type="id"
    )
    if not sessions:
        session = "NaN"
        report = HTMLReport(subject, session, task, analysis_type=analysis_type)
        report_path = (
            report_dir
            / f"sub-{subject}_ses-NaN_task-{task}_desc-{analysis_type}_report.html"
        )
        msg = f"No sessions for {subject} for {task}."
        LGR.warning(msg)

        report.mark_excluded(msg)
        report.create_report(report_path, "first_level.html")

        sys.exit(1)

    return report_dir, layout, sessions


def _get_required_files(layout, label, **kwargs):
    files = layout.get(return_type="file", **kwargs)
    if not files:
        raise MissingFileError(f"No {label} found for session: {kwargs.get('session')}")

    return files


def _select_by_space(files, space, label="files"):
    matches = [file for file in files if space in Path(file).name]
    if not matches:
        raise MissingFileError(
            f"No {label} matching space '{space}'. Available: "
            f"{[Path(f).name for f in files]}"
        )

    return matches[0]


def _collect_files(layout, subject, session, task, space, acompcor_strategy):
    base_kwargs = dict(subject=subject, session=session, task=task)

    confounds_tsv_file = _get_required_files(
        layout,
        "confound TSV files",
        scope="derivatives",
        desc="confounds",
        extension="tsv",
        **base_kwargs,
    )[0]

    if acompcor_strategy != "none":
        confounds_json_file = _get_required_files(
            layout,
            "confound JSON files",
            scope="derivatives",
            desc="confounds",
            extension="json",
            **base_kwargs,
        )[0]
    else:
        confounds_json_file = None

    event_file = _get_required_files(
        layout,
        "event files",
        scope="raw",
        suffix="events",
        extension="tsv",
        **base_kwargs,
    )[0]

    mask_file = _select_by_space(
        _get_required_files(
            layout,
            "mask files",
            scope="derivatives",
            suffix="mask",
            extension="nii.gz",
            **base_kwargs,
        ),
        space,
    )
    LGR.info(f"Using the following mask file: {mask_file}")

    nifti_file = _select_by_space(
        _get_required_files(
            layout,
            "nifti files",
            scope="derivatives",
            suffix="bold",
            extension="nii.gz",
            **base_kwargs,
        ),
        space,
    )
    LGR.info(f"Using the following nifti file: {nifti_file}")

    return SessionFiles(
        confounds_tsv_file=confounds_tsv_file,
        confounds_json_file=confounds_json_file,
        event_file=event_file,
        mask_file=mask_file,
        nifti_file=nifti_file,
    )


def _skip_denoising(nifti_filename, exclude_niftis_file):
    if not exclude_niftis_file or not Path(exclude_niftis_file).exists():
        return False

    excluded_niftis_prefixes = _get_dataframe(exclude_niftis_file)[
        "nifti_prefix_filename"
    ].tolist()

    return any(
        Path(nifti_filename).name.startswith(prefix)
        for prefix in excluded_niftis_prefixes
    )


def _check_run_validity(nifti_file, exclude_niftis_file):
    if _skip_denoising(nifti_file, exclude_niftis_file):
        LGR.info(
            "Denoising of the following file will be skipped due to the prefix being found in "
            f"`exclude_niftis_file` ({exclude_niftis_file}): {nifti_file}"
        )
        raise InvalidRunError(
            f"Skipped due to prefix being found in {exclude_niftis_file}"
        )


def collect_session_files(
    report_dir,
    layout,
    subject,
    session,
    task,
    space,
    acompcor_strategy,
    exclude_niftis_file,
    analysis_type,
):
    report = HTMLReport(subject, session, task, analysis_type=analysis_type)
    report_path = (
        report_dir
        / f"sub-{subject}_ses-{session}_task-{task}_desc-{analysis_type}_report.html"
    )

    try:
        session_files = _collect_files(
            layout, subject, session, task, space, acompcor_strategy
        )
        _check_run_validity(session_files.nifti_file, exclude_niftis_file)

        return report, report_path, session_files, False
    except (MissingFileError, InvalidRunError) as exc:
        LGR.warning(str(exc))
        report.mark_excluded(str(exc))
        report.create_report(report_path, "first_level.html")

        return report, report_path, None, True


def check_censoring(
    subject,
    session,
    task,
    confounds_df,
    n_dummy_scans,
    fd_threshold,
    exclusion_criteria,
):
    if n_dummy_scans == "auto":
        n_non_steady_state = compute_n_dummy_scans(confounds_df)
        LGR.info(f"There are {n_non_steady_state} non-steady state scans.")
    else:
        n_non_steady_state = n_dummy_scans

    censor_mask = create_censor_mask(
        confounds_df,
        column_name="framewise_displacement",
        n_dummy_scans=n_non_steady_state,
        threshold=fd_threshold,
    ).astype(np.int8)

    kept = censor_mask[n_non_steady_state:]
    n_censored = int(np.sum(kept == 0))
    percent_censored = n_censored / kept.size

    LGR.warning(
        f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
        f"proportion of steady state volumes removed at an fd threshold > {fd_threshold} mm: "
        f" {percent_censored}"
    )

    fd_values = confounds_df["framewise_displacement"].to_numpy(copy=True)
    fd_steady_state = fd_values[n_non_steady_state:]
    mean_fd_before_censoring = np.mean(fd_steady_state)

    fd_retained = fd_steady_state[kept.astype(bool)]
    # If NaN, guaranteed for the run to be excluded
    mean_fd_after_censoring = np.mean(fd_retained) if fd_retained.size > 0 else np.nan

    LGR.info(
        f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
        f"mean FD (excluding non-steady state volumes): {mean_fd_before_censoring:.4f} mm, "
        f"mean FD (after censoring at FD > {fd_threshold} mm): {mean_fd_after_censoring:.4f} mm"
    )

    censor_info = CensorInfo(
        n_dummy_scans=int(n_non_steady_state),
        n_censored=int(n_censored),
        n_total=int(kept.size),
        percent_censored=float(percent_censored),
        dummy_method=(
            "user-specified"
            if n_dummy_scans != "auto"
            else "number of 'non_steady_state_outlier_XX' columns in fMRIPrep confounds TSV"
        ),
        mean_fd_before_censoring=float(mean_fd_before_censoring),
        mean_fd_after_censoring=float(mean_fd_after_censoring),
    )

    if percent_censored > exclusion_criteria:
        LGR.warning(
            f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
            "run excluded because the percent censored is greater than the "
            f"exclusion criteria: {exclusion_criteria}"
        )
        raise InvalidRunError(
            f"Proportion of flagged volumes ({percent_censored:.1%}) "
            f"exceeded threshold ({exclusion_criteria:.0%})."
        )

    return censor_mask, censor_info


def plot_signal(
    signal_regressor_file,
    tr,
    plot_title,
    upsample_dt=None,
    figsize=(8, 6),
    base_filename=None,
):
    dt = upsample_dt or tr

    Y = np.loadtxt(signal_regressor_file).flatten()
    max_time = len(Y) * dt
    X = np.linspace(0, max_time, len(Y))

    plt.figure(figsize=figsize)
    plt.plot(X, Y)
    plt.xlabel(f"Time (seconds) | dt = {dt}", fontsize=15)
    plt.ylabel("Amplitude", fontsize=15)
    plt.title(plot_title)

    filename = base_filename or plot_title.replace(" ", "_").lower() + ".png"
    save_filename = signal_regressor_file.parent / filename

    LGR.info(f"Saving '{plot_title}' plot to: {save_filename}")
    plt.savefig(save_filename, dpi=120, bbox_inches="tight")
    plt.clf()

    return save_filename


def create_diagnostic_condition_plots(condition_filenames_dict, tr, fd_threshold):
    plots = []
    for cond_name, cond_vector_files in condition_filenames_dict.items():
        noncensored_plot = plot_signal(
            cond_vector_files["noncensored_binary_vector"],
            tr,
            plot_title=f"{cond_name} No Motion Censoring",
            base_filename=f"{cond_name}_desc-noncensored_binary_vector.png",
        )
        censored_plot = plot_signal(
            cond_vector_files["censored_binary_vector"],
            tr,
            plot_title=f"{cond_name} Censored (FD = {fd_threshold})",
            base_filename=f"{cond_name}_desc-censored_binary_vector.png",
        )
        plots.append(
            {
                "name": cond_name,
                "noncensored_condition_plot": embed_img(noncensored_plot),
                "censored_condition_plot": embed_img(censored_plot),
            }
        )

    return plots


def summarize_timing_conditions(timing_dir, task):
    conditions = []
    for timing_file in sorted(timing_dir.glob("*.1D")):
        data = np.loadtxt(timing_file, delimiter=" ")
        conditions.append(
            {
                "name": timing_file.stem,
                "n_events": int(data.size) if data.size > 0 else 0,
            }
        )

    event_type = "events" if task in EVENT_RELATED_TASKS else "blocks"

    return conditions, event_type


def collect_acompcor_names(
    confounds_json_file, confounds_df, acompcor_strategy, n_acompcor
):
    if acompcor_strategy == "none":
        acompcor_regressors, acompcor_regressor_names = None, None
    else:
        with open(confounds_json_file, "r") as f:
            confounds_meta = json.load(f)

        acompcor_regressor_names = get_acompcor_component_names(
            confounds_meta,
            n_acompcor,
            acompcor_strategy,
        )
        acompcor_regressors = confounds_df[acompcor_regressor_names].to_numpy(
            copy=True,
        )

    return acompcor_regressors, acompcor_regressor_names


def filter_regressor_names(
    cosine_regressor_names, motion_regressor_names, acompcor_regressor_names
):
    regressor_names_nested_list = filter(
        None,
        [
            cosine_regressor_names,
            motion_regressor_names,
            acompcor_regressor_names,
        ],
    )
    regressor_names = [
        regressor
        for regressor_list in regressor_names_nested_list
        for regressor in regressor_list
    ]

    return regressor_names


def is_timing_file_empty(timing_file):
    if not Path(timing_file).exists():
        return True

    return np.loadtxt(timing_file, delimiter=" ").size == 0


def seed_mask_name_check(seed_filename):
    if "sphere_mask" not in (seed_basename := Path(seed_filename).name):
        LGR.warning(
            f"'sphere_mask' is not in the seed mask filename: {seed_basename}\n"
            "Ignore warning if intentional."
        )


def get_sphere_radius(seed_filename):
    radius = get_entity_value(seed_filename, entity="radius")
    if radius:
        radius = str(radius).rstrip("m")

    return radius
