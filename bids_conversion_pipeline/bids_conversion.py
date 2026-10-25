import re, shutil, sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nibabel.filebasedimages import ImageFileError

from bidsaid._decorators import check_nifti
from bidsaid.io import (
    _copy_file,
    load_nifti,
    compress_img,
    is_nifti_truncated,
    regex_glob,
)
from bidsaid.logging import setup_logger
from bidsaid.path_utils import is_valid_date

from _bids_conversion_utils import _strip_entity
from _general_utils import _check_subjects_visits_file, _resolve_directories
from standardize_task_names import _standardize_task_pipeline
from create_bids_dir import _generate_bids_dir_pipeline
from create_metadata import _create_json_sidecar_pipeline

LGR = setup_logger(__name__)


def _filter_subjects(
    folders: list[Path], subjects: Optional[list[str | int]]
) -> list[Path]:
    if subjects:
        return [folder for folder in folders if folder.name.split("_")[0] in subjects]
    else:
        return folders


def _filter_source_folders(
    folders: list[Path], exclude_src_folder_names: Optional[list[str]]
):
    if not exclude_src_folder_names:
        return folders

    exclude_src_folder_names = [
        Path(folder).name for folder in exclude_src_folder_names
    ]

    return [folder for folder in folders if folder.name not in exclude_src_folder_names]


def _copy_nifti_files(nifti_file: Path, temp_dir: Path) -> None:
    dst_file = temp_dir / nifti_file.parent.name / nifti_file.name
    _copy_file(
        src_file=nifti_file,
        dst_file=dst_file,
        remove_src_file=False,
    )

    try:
        load_nifti(dst_file)
    except ImageFileError:
        LGR.warning(
            "Deleting the following nifti file from the temporary directory since "
            f"Nibabel cannot work out file type: {nifti_file}",
            exc_info=True,
        )
        dst_file.unlink()
        return

    try:
        _is_raw_nifti(dst_file)
    except ValueError:
        LGR.warning(
            "Deleting the following nifti file from the temporary directory "
            f"since it is likely not a raw image: {nifti_file}",
            exc_info=True,
        )
        dst_file.unlink()
        return

    # Deal with case where files have {sub_id}{V\d+}_ instead of {sub_id}_
    filename_participant_id = re.findall(r"\d{5}", nifti_file.name)[0]
    folder_participant_id = re.findall(r"\d{5}", nifti_file.parent.name)[0]
    if filename_participant_id != folder_participant_id:
        LGR.warning(
            "Deleting the following nifti file from the temporary directory "
            "since it is nested in a source directory with a different subject "
            f"id ({folder_participant_id}): {nifti_file}"
        )
        dst_file.unlink()
        return

    if nifti_file.name.endswith(".nii"):
        if is_nifti_truncated(dst_file):
            LGR.warning(
                f"Truncated NIfTI detected: {dst_file}. Removing from temporary directory."
            )
            if dst_file.exists():
                dst_file.unlink()
        else:
            compress_img(dst_file, dst_file.parent, remove_src_file=True)


@check_nifti(nifti_param_name="nifti_file")
def _is_raw_nifti(nifti_file):
    pass


def _copy_data_to_temp_dir(
    src_dir: Path,
    temp_dir: Path,
    subjects: Optional[list[str | int]],
    exclude_src_folder_names: Optional[list[str]],
    exclude_nifti_filenames: Optional[list[str]],
) -> None:
    subject_folders = regex_glob(src_dir, pattern=r"^\d{5}.*_\d{6}$")
    subject_folders = _filter_subjects(subject_folders, subjects)
    subject_folders = _filter_source_folders(subject_folders, exclude_src_folder_names)
    if not subject_folders:
        LGR.warning(
            f"After filtering, no folders can be converted from the following directory: {src_dir}"
        )
        sys.exit(1)

    exclude_nifti_filenames = exclude_nifti_filenames or []
    exclude_nifti_filenames = [Path(file).name for file in exclude_nifti_filenames]
    for subject_folder in subject_folders:
        date_str = subject_folder.name.split("_")[-1]
        if not is_valid_date(date_str, "%y%m%d") or len(date_str) != 6:
            LGR.warning(
                f"The following folder does not have the '%y%m%d' date format: {subject_folder}. "
                "Dates are sorted and should be standardized across all folders."
            )

        # Handle edge case where analyses related files placed in same folder as
        # original nifti which has the naming {subjectID}_{scan_date}_{acqusition_number}
        nifti_files = regex_glob(
            subject_folder, pattern=r"^\d{5}([vV]\d+)?_(\d{6})?.*\.(nii|nii.gz)$"
        )

        for nifti_file in nifti_files:
            if nifti_file.name in exclude_nifti_filenames:
                continue

            _copy_nifti_files(nifti_file, temp_dir)


def run_pipeline(
    src_dir: str,
    temp_dir: str,
    bids_dir: str,
    subjects: Optional[list[str | int]],
    exclude_src_folder_names: Optional[list[str]],
    exclude_nifti_filenames: Optional[list[str]],
    cohort: Literal["kids", "adults"],
    delete_temp_dir: bool,
    create_dataset_metadata: bool,
    add_sessions_tsv: bool,
    subjects_visits_file: str,
) -> Path:
    try:
        if (cohort := cohort.lower()) not in ["kids", "adults"]:
            raise ValueError("'--cohort' must be 'kids' or 'adults'.")

        LGR.info("Validating subjects visits CSV...")
        _check_subjects_visits_file(subjects_visits_file, dose_column_required=False)

        LGR.info("Resolving directories...")
        bids_dir, temp_dir = _resolve_directories(
            bids_dir, temp_dir, caller="BIDS Dataset"
        )

        if subjects:
            subjects = _strip_entity(subjects)
            subjects = [re.findall(r"\d{5}", x)[0] for x in subjects]

        LGR.info("Copying data to temporary directory...")
        _copy_data_to_temp_dir(
            Path(src_dir),
            temp_dir,
            subjects,
            exclude_src_folder_names,
            exclude_nifti_filenames,
        )

        # Pipeline to identify un-named NIfTI images and standardize task names
        LGR.info("Standardizing task names...")
        _standardize_task_pipeline(temp_dir, cohort)

        # Pipeline to move files to BIDS directory
        LGR.info("Generating BIDS directory...")
        _generate_bids_dir_pipeline(
            temp_dir,
            bids_dir,
            cohort,
            subjects,
            create_dataset_metadata,
            add_sessions_tsv,
            delete_temp_dir,
            subjects_visits_file,
        )

        LGR.info("Creating JSON ")
        # Pipeline to create JSON sidecars for NIfTI images
        _create_json_sidecar_pipeline(bids_dir, cohort)
    finally:
        if delete_temp_dir and (isinstance(temp_dir, Path) and temp_dir.exists()):
            shutil.rmtree(temp_dir)

    return bids_dir
