import argparse, re
from pathlib import Path

import numpy as np

from bidsaid.cli import none_or
from bidsaid.files import get_entity_value
from bidsaid.io import regex_glob
from bidsaid.logging import setup_logger
from bidsaid.surface import (
    fetch_fsaverage_surfaces,
    get_cortex_masks,
)
from bidsaid.qc import (
    zscore_to_reference,
    interpolate_censored_img,
    find_outlier_volumes,
)
from bidsaid.metadata import get_tr
from bidsaid.visual import SurfaceRenderer
from bidsaid.movie import (
    MovieWriter,
    compute_vmax,
    virtual_display,
    create_surface_run,
)

LGR = setup_logger(__name__)

SUB_TO_NAME = {
    "sub-01": "",
    "sub-02": "",
}


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create surface movies of denoised TASK_A data z-scored against the TASK_B baseline."
    )
    parser.add_argument(
        "--denoised_dir",
        dest="denoised_dir",
        required=True,
        help="Path to the directory containing the denoised images.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory for the movies.",
    )
    parser.add_argument(
        "--tasks",
        dest="tasks",
        nargs=2,
        default=["music", "rest"],
        metavar=("TASK_A", "TASK_B"),
        required=False,
        help=(
            "The two task names. The movie shows each TASK_A volume minus the voxel-wise TASK_B mean, "
            "divided by the voxel-wise TASK_B standard deviation (default: music rest)."
        ),
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        nargs="+",
        required=False,
        default=None,
        help="Subject IDs without the 'sub-' entity. If not specified, all subjects are used.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        required=False,
        default=None,
        type=none_or(),
        help="Template space to filter images by (e.g. 'MNI152NLin2009cAsym_res-2').",
    )
    parser.add_argument(
        "--n_interpolated_frames",
        dest="n_interpolated_frames",
        default=3,
        type=int,
        required=False,
        help="Number of interpolated frames inserted in each gap between two real volumes.",
    )
    parser.add_argument(
        "--interpolation_method",
        dest="interpolation_method",
        default="cubic",
        choices=["cubic", "pchip"],
        required=False,
        help="Interpolation method used when 'n_interpolated_frames' is greater than 0.",
    )
    parser.add_argument(
        "--fps",
        dest="fps",
        default=24,
        type=float,
        required=False,
        help=(
            "Frames per second of the movie. When '--playback_speed' is set, this is the target output frame rate (e.g. 24 or 30)."
        ),
    )
    parser.add_argument(
        "--playback_speed",
        dest="playback_speed",
        default=1.0,
        type=none_or(float),
        required=False,
        help="Playback speed relative to real time (1 = real time)",
    )
    parser.add_argument(
        "--threshold",
        dest="threshold",
        default=0.0,
        type=float,
        required=False,
        help="Values with an absolute value below this threshold are transparent.",
    )
    parser.add_argument(
        "--cmap",
        dest="cmap",
        default="cold_hot",
        required=False,
        help="the specified nilearn or matplotlib colormap",
    )
    parser.add_argument(
        "--fwhm",
        dest="fwhm",
        default=None,
        type=none_or(float),
        required=False,
        help="Spatial smoothing (mm).",
    )
    parser.add_argument(
        "--sigma",
        dest="sigma",
        default=None,
        type=none_or(float),
        required=False,
        help="Temporal smoothing.",
    )
    parser.add_argument(
        "--size",
        dest="size",
        nargs=2,
        type=int,
        default=[1000, 700],
        metavar=("WIDTH", "HEIGHT"),
        required=False,
        help="Width and height in pixels of each hemisphere panel.",
    )
    parser.add_argument(
        "--zoom",
        dest="zoom",
        default=1.7,
        type=float,
        required=False,
        help="Camera zoom of each panel",
    )
    parser.add_argument(
        "--label_font_size",
        dest="label_font_size",
        default=80,
        type=int,
        required=False,
        help="Font size of the label drawn in the upper left corner of each participant's frame.",
    )
    parser.add_argument(
        "--quality",
        dest="quality",
        default=8,
        type=int,
        required=False,
        help="Video quality passed to ffmpeg, from 0 (worst) to 10 (best, largest file).",
    )
    parser.add_argument(
        "--hemisphere_spacing",
        dest="hemisphere_spacing",
        default=180,
        type=int,
        required=False,
        help="Width in pixels of the empty space between the left and right hemisphere panels",
    )
    parser.add_argument(
        "--top_spacing",
        dest="top_spacing",
        default=0,
        type=int,
        required=False,
        help="Height in pixels of an header space above all panels.",
    )
    parser.add_argument(
        "--subject_spacing",
        dest="subject_spacing",
        default=3,
        type=int,
        required=False,
        help=" Height in pixels of the empty space between the participant rows of a movie",
    )
    parser.add_argument(
        "--outlier_threshold",
        dest="outlier_threshold",
        default=5.0,
        type=none_or(float),
        required=False,
        help=(
            "Volumes whose overall signal is this many robust standard deviations above the run's typical level "
            "(e.g. dummy scans or motion spikes) are replaced by interpolating from neighboring volumes, so the "
            "movie keeps its length and timing. Use 'none' to turn this off (default: 5.0)."
        ),
    )

    return parser


def find_denoised_files(
    denoised_dir: str | Path,
    task: str,
    subjects: list[str] | None,
    space: str | None,
) -> dict[str, Path]:
    pattern = rf"^sub-[^_]+_.*task-{re.escape(task)}_.*desc-denoised_bold\.nii\.gz$"
    denoised_files = sorted(regex_glob(denoised_dir, pattern, recursive=True))

    if subjects:
        subjects = {str(subject).removeprefix("sub-") for subject in subjects}
        denoised_files = [
            file for file in denoised_files if get_entity_value(file, "sub") in subjects
        ]

    if space:
        denoised_files = [
            file for file in denoised_files if f"space-{space}_" in file.name
        ]

    files_by_subject = {}
    for file in denoised_files:
        subject = get_entity_value(file, "sub", return_entity_prefix=True)
        files_by_subject[subject] = file

    return files_by_subject


def remove_outlier_volumes(
    zscored_img, subject_id: str, tr: float, threshold: float | None
):
    """Replace outlier volumes with values interpolated from the neighboring volumes."""
    if threshold is None:
        return zscored_img

    outliers, level = find_outlier_volumes(zscored_img, threshold)
    if not outliers.any():
        LGR.info(f"{subject_id}: no outlier volumes found.")
        return zscored_img

    if outliers.all():
        LGR.warning(
            f"{subject_id}: every volume was flagged as an outlier, leaving the data unchanged."
        )
        return zscored_img

    details = ", ".join(
        f"volume {index} (t = {index * tr:.1f} s, level {level[index]:.2f} vs typical {np.median(level):.2f})"
        for index in np.flatnonzero(outliers)
    )
    LGR.warning(f"{subject_id}: replacing outlier volumes by interpolation: {details}")

    return interpolate_censored_img(
        zscored_img, (~outliers).astype(np.uint8), method="pchip", extrapolate=True
    )


def main(
    denoised_dir,
    dst_dir,
    tasks,
    subjects,
    space,
    n_interpolated_frames,
    interpolation_method,
    fps,
    playback_speed,
    outlier_threshold,
    threshold,
    cmap,
    fwhm,
    sigma,
    size,
    zoom,
    label_font_size,
    quality,
    hemisphere_spacing,
    top_spacing,
    subject_spacing,
):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = tuple(tasks)

    surfaces = fetch_fsaverage_surfaces(density="164k")
    cortex_masks = get_cortex_masks(surfaces["medial"])

    files_by_task = {
        task: find_denoised_files(denoised_dir, task, subjects, space) for task in tasks
    }

    for task, files in files_by_task.items():
        LGR.info(f"Files for task {task}: {[file.name for file in files.values()]}")

    common_subjects = sorted(
        set.intersection(*(set(files) for files in files_by_task.values()))
    )

    for task, files in files_by_task.items():
        for subject_id in sorted(set(files).difference(common_subjects)):
            LGR.warning(
                f"{subject_id} only has a denoised file for task {task}, skipping participant."
            )

    if not common_subjects:
        LGR.critical(
            f"No participant has a denoised file for both tasks {tasks}, no movies created."
        )

        return

    surface_runs = []
    for subject_id in common_subjects:
        task_files = {task: files_by_task[task][subject_id] for task in tasks}
        tasks = list(task_files)
        LGR.info(f"Task A: {tasks[0]}; Task B: {tasks[1]}")

        zscored_img = remove_outlier_volumes(
            zscore_to_reference(task_files[tasks[0]], task_files[tasks[1]]),
            subject_id,
            get_tr(task_files[tasks[0]]),
            outlier_threshold,
        )

        surface_runs.append(
            create_surface_run(
                zscored_img,
                surfaces["pial"],
                surfaces["white"],
                cortex_masks=cortex_masks,
                fwhm=fwhm,
                temporal_sigma=sigma,
                subject_id=subject_id,
                tr=get_tr(task_files[tasks[0]]),
                name=SUB_TO_NAME.get(subject_id),
                task_name=f"{tasks[0]}_minus_{tasks[1]}",
                n_interpolated_frames=n_interpolated_frames,
                upsample_interpolation_method=interpolation_method,
            )
        )

    surface_runs = [
        surface_run._replace(
            color_range=(
                -(run_vmax := compute_vmax([surface_run], threshold=threshold)),
                run_vmax,
            )
        )
        for surface_run in surface_runs
    ]

    surface_runs = [
        run._replace(
            lh=run.lh / run.color_range[1],
            rh=run.rh / run.color_range[1],
            color_range=None,
        )
        for run in surface_runs
    ]

    with SurfaceRenderer(
        *surfaces["inflated"],
        views={"lh": ["lateral"], "rh": ["lateral"]},
        layout="row",
        sulcal_depths=surfaces["sulc"],
        size=tuple(size),
        zoom=zoom,
        cmap=cmap,
        color_range=(-1, 1),
        threshold=(0.5, 0.7),
        zero_transparent=False,
        background="black",
        label_font_size=label_font_size,
        antialiasing=True,
        hemisphere_spacing=hemisphere_spacing,
        top_spacing=top_spacing,
    ) as renderer, virtual_display():
        MovieWriter(
            surface_runs,
            renderer,
            task_order=[f"{tasks[0]}_minus_{tasks[1]}"],
            quality=quality,
            target_fps=fps,
            playback_speed=playback_speed,
            subject_spacing=subject_spacing,
            label_format="{subject_id}",
        ).write(dst_dir)


if __name__ == "__main__":
    parser = _get_cmd_args()
    args = parser.parse_args()
    main(**vars(args))
