import re
from pathlib import Path
from typing import Literal

from bidsaid.io import regex_glob, get_nifti_header
from bidsaid.logging import setup_logger
from bidsaid.metadata import is_3d_img, infer_task_from_img
from bidsaid.path_utils import get_file_acquisition_order, sort_by_acquisition_order

LGR = setup_logger(__name__)

_TASK_NAMES_PATTERNS = {
    "kids": "(mtl(_neu)?|n([_-])?back|princess|flanker|mprage(32)?)",
    "adults": "(mtl|n([_-])?back|(simple.*(repeat.*)?)?gng|flanker|mprage(32)?)",
}
_TASK_VOLUME_MAP = {
    "kids": {"flanker": 305, "nback": 246, "princess": 262, "mtl": 96},
    "adults": {"flanker": 305, "nback": 124, "mtl": 174, "gng": 98},
}


def _pattern_found(pattern: str, filename: str):
    match = re.search(pattern, filename.lower())

    return True if match else False


def _infer_file_identity(temp_dir: Path, cohort: Literal["kids", "adults"]) -> None:
    """
    For files with no task names, identify the file by whether its
    3D (anatomical) or not. If it is a 4D image, then infer the
    task by the number of volumes.
    """
    nifti_files = regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True)
    for nifti_file in nifti_files:
        if "nogo" in nifti_file.name:
            base_name = nifti_file.name.split("nogo")[0] + "gng"
            base_name = (
                f"{base_name}_11_1.nii.gz"
                if "simplegng" in base_name
                else f"{base_name}_12_1.nii.gz"
            )
            new_filename = nifti_file.parent / base_name
            nifti_file.rename(new_filename)
        elif not _pattern_found(_TASK_NAMES_PATTERNS[cohort], nifti_file.name):
            if is_3d_img(nifti_file):
                # Safety identity check based on voxel sizes for near isotropic mprage32
                # Note: Protocol doesn't collect fmaps
                min_thresh = 0.8
                max_thresh = 1.1
                voxel_sizes = get_nifti_header(nifti_file).get_zooms()
                if all(
                    min_thresh <= vox_size <= max_thresh for vox_size in voxel_sizes
                ):
                    desc = "mprage32"
                else:
                    LGR.warning(
                        f"Voxel sizes out of bounds ({min_thresh} mm, {max_thresh} mm). "
                        f"Actual voxels sizes are {voxel_sizes} for the following file: {nifti_file}. "
                        "Check original file in the source directory since the temp file will be deleted "
                        "to allow the conversion to continue for the remaining files."
                    )
                    nifti_file.unlink()
                    continue
            else:
                desc = infer_task_from_img(nifti_file, _TASK_VOLUME_MAP[cohort])

            if not desc:
                LGR.info(
                    f"The identity of the following file could not be identified: {nifti_file}. "
                    "Removing file form temporary directory."
                )
                nifti_file.unlink()
                continue

            # Special case since there are two mtl files for the kids dataset and the adult dataset
            # with the same number of volumes
            # Each mtl file has the acquisition number in the filename that preceeds "_1"
            # MTLE comes before the MTLR hence the acquisition number is important
            # Same for go-nogo, there are two versions with the same number of volumes for the adult data
            if desc.startswith("mtl") or desc.endswith("gng"):
                match_str = get_file_acquisition_order(nifti_file.name)
                desc += f"_{match_str}.nii.gz"

                # Get the name before the acquisition number
                prefix_filename = str(nifti_file).split(match_str)[0]
                new_filename = f"{prefix_filename}_{desc}"
            else:
                new_filename = f"{str(nifti_file).split('.nii.gz')[0]}_{desc}.nii.gz"

            new_filename = new_filename.replace("__", "_")
            nifti_file.rename(new_filename)
        elif any(x in nifti_file.name.lower() for x in ["mtle", "mtlr"]):
            new_filename = nifti_file.parent / nifti_file.name.lower()
            nifti_file.rename(new_filename)


def _standardize_task_pipeline(
    temp_dir: Path, cohort: Literal["kids", "adults"]
) -> None:
    _infer_file_identity(temp_dir, cohort)
    _differentiate_mtl_filenames(temp_dir)
    _standardize_nback_filenames(temp_dir)

    if cohort == "adults":
        _differentiate_gng_filenames(temp_dir)


def _differentiate_filenames(
    temp_dir: Path, task: str, task_order_dict: dict[int, str]
):
    for subject_folder in temp_dir.glob("*"):
        nifti_files = list(
            regex_glob(subject_folder, pattern=r"^.*\.nii\.gz$", recursive=True)
        )
        nifti_files = [
            nifti_file
            for nifti_file in nifti_files
            if _pattern_found(_get_pattern(task), nifti_file.name)
        ]
        for index, nifti_file in enumerate(sort_by_acquisition_order(nifti_files)):
            task_name = task_order_dict[index]
            new_nifti_filename = nifti_file.parent / nifti_file.name.lower().replace(
                _get_replace_name(nifti_file, task), task_name
            )
            nifti_file.rename(new_nifti_filename)


def _get_pattern(task: Literal["mtl", "gng", "nback"]):
    return {
        "mtl": "(mtl(?!e|r)(_neu)?)",
        "gng": "(((simple.*)?(repeat.*)?)?gng)",
        "nback": "(n([_-])?back)",
    }[task]


def _get_replace_name(nifti_file: Path, task: Literal["mtl", "gng", "nback"]):
    return re.search(_get_pattern(task), nifti_file.name.lower()).group(0)


def _differentiate_mtl_filenames(temp_dir: Path) -> None:
    _differentiate_filenames(
        temp_dir, task="mtl", task_order_dict={0: "mtle", 1: "mtlr"}
    )


def _differentiate_gng_filenames(temp_dir: Path) -> None:
    _differentiate_filenames(
        temp_dir, task="gng", task_order_dict={0: "simplegng", 1: "complexgng"}
    )


def _standardize_nback_filenames(temp_dir: Path) -> None:
    nifti_files = list(regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True))
    nifti_files = [
        nifti_file
        for nifti_file in nifti_files
        if _pattern_found(_get_pattern("nback"), nifti_file.name)
    ]
    for nifti_file in nifti_files:
        new_filename = nifti_file.parent / nifti_file.name.lower().replace(
            _get_replace_name(nifti_file, "nback"), "nback"
        )
        nifti_file.rename(new_filename)
