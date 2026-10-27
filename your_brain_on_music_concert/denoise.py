import argparse, json
from pathlib import Path

import bids, nibabel as nib, numpy as np, pandas as pd
from nilearn.interfaces.fmriprep import load_confounds

from bidsaid.logging import setup_logger
from bidsaid.metadata import get_tr
from bidsaid.cli import none_or
from bidsaid.qc import (
    compute_n_dummy_scans,
    create_censor_mask,
    interpolate_censored_img,
    regress_confounds,
)

LGR = setup_logger(__name__)

TRUNCATION_DURATIONS = {"music": 328}


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Denoise fMRIPrep preprocessed images."
    )
    parser.add_argument(
        "--src_dir",
        dest="src_dir",
        required=True,
        help="Path to the fMRIPrep derivatives directory.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory for the denoised images.",
    )
    parser.add_argument(
        "--subject_id",
        dest="subject_id",
        required=True,
        help="Subject ID without the 'sub-' entity.",
    )
    parser.add_argument(
        "--tasks",
        dest="tasks",
        nargs="+",
        required=False,
        default=None,
        help="Names of the tasks to denoise. If not specified, all tasks are denoised.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        required=True,
        help="Template space (e.g. 'MNI152NLin2009cAsym_res-2').",
    )
    parser.add_argument(
        "--motion",
        dest="motion",
        default="full",
        choices=["basic", "power2", "derivatives", "full"],
        required=False,
        help=(
            "Motion parameters to use: 'basic' (6), 'power2' (base + power, 12), "
            "'derivatives' (base + derivatives, 12), or 'full' (24)."
        ),
    )
    parser.add_argument(
        "--compcor",
        dest="compcor",
        default="anat_separated",
        choices=["anat_combined", "anat_separated"],
        required=False,
        help="Whether to use aCompCor components from the combined or separate WM and CSF masks.",
    )
    parser.add_argument(
        "--n_compcor",
        dest="n_compcor",
        default=5,
        type=int,
        required=False,
        help="Number of aCompCor components (per mask when using 'anat_separated').",
    )
    parser.add_argument(
        "--fd_threshold",
        dest="fd_threshold",
        default=None,
        type=none_or(float),
        required=False,
        help=(
            "Framewise displacement threshold (mm) for censoring high motion volumes. "
            "Non-steady-state volumes are always censored. Set to 'None' to only censor "
            "non-steady-state volumes."
        ),
    )

    return parser


def _get_layout(src_dir):
    return bids.BIDSLayout(src_dir, validate=False, config=["bids", "derivatives"])


def _get_nifti_files(layout, subject, task, space):
    nifti_files = layout.get(
        scope="all",
        subject=subject,
        task=task,
        desc="preproc",
        suffix="bold",
        extension="nii.gz",
        return_type="file",
    )

    return sorted(file for file in nifti_files if f"space-{space}_" in Path(file).name)


def _get_mask_file(nifti_file):
    mask_file = Path(
        str(nifti_file).replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz")
    )
    if not mask_file.exists():
        LGR.warning(
            f"No brain mask found for {Path(nifti_file).name}, denoising without a mask."
        )
        return None

    return mask_file


def _get_confounds_file(nifti_file):
    nifti_file = Path(nifti_file)
    confounds_file = nifti_file.parent / (
        nifti_file.name.split("_space-")[0] + "_desc-confounds_timeseries.tsv"
    )
    if not confounds_file.exists():
        raise FileNotFoundError(f"No confounds file found for {nifti_file.name}.")

    return confounds_file


def _create_dataset_description(dst_dir):
    dataset_description = {
        "Name": "Denoised fMRIPrep Outputs",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "denoise.py"}],
    }
    try:
        with open(Path(dst_dir) / "dataset_description.json", "x") as f:
            json.dump(dataset_description, f, indent=4)
    except FileExistsError:
        pass


def _get_n_volumes_to_keep(task, tr, n_volumes):
    if task not in TRUNCATION_DURATIONS:
        return None

    n_keep = int(round(TRUNCATION_DURATIONS[task] / tr))
    if n_keep >= n_volumes:
        return None

    LGR.info(
        f"Truncating task {task} from {n_volumes} to {n_keep} volumes "
        f"({TRUNCATION_DURATIONS[task]} seconds at TR={tr})."
    )

    return n_keep


def _get_censor_mask(nifti_file, n_volumes, fd_threshold, n_keep=None):
    confounds_tsv_df = pd.read_csv(_get_confounds_file(nifti_file), sep="\t").fillna(0)
    n_dummy_scans = compute_n_dummy_scans(confounds_tsv_df)
    censor_mask = create_censor_mask(
        confounds_tsv_df,
        column_name="framewise_displacement",
        threshold=fd_threshold,
        n_dummy_scans=n_dummy_scans,
    ).astype(np.int8)

    if censor_mask.size != n_volumes:
        raise ValueError(
            f"Censor mask has {censor_mask.size} values but {Path(nifti_file).name} "
            f"has {n_volumes} volumes."
        )

    if n_keep is not None:
        censor_mask = censor_mask[:n_keep]

    if not censor_mask.any():
        raise ValueError(f"All volumes are censored for {Path(nifti_file).name}.")

    return censor_mask, n_dummy_scans


def denoise_nifti_img(nifti_file, task, motion, compcor, n_compcor, fd_threshold):
    confounds_df, _ = load_confounds(
        str(nifti_file),
        strategy=("motion", "high_pass", "compcor"),
        motion=motion,
        compcor=compcor,
        n_compcor=n_compcor,
        demean=False,
    )

    tr = get_tr(nifti_file)
    raw_img = nib.load(nifti_file)
    n_original_volumes = raw_img.shape[-1]

    n_keep = _get_n_volumes_to_keep(task, tr, n_original_volumes)
    if n_keep is not None:
        raw_img = raw_img.slicer[..., :n_keep]
        confounds_df = confounds_df.iloc[:n_keep].reset_index(drop=True)

    n_volumes = raw_img.shape[-1]
    LGR.info(f"Using the following confounds: {list(confounds_df.columns)}")

    censor_mask, n_dummy_scans = _get_censor_mask(
        nifti_file, n_original_volumes, fd_threshold, n_keep
    )
    censored_volumes = np.flatnonzero(censor_mask == 0).tolist()
    LGR.info(
        f"Censoring {len(censored_volumes)} volumes ({n_dummy_scans} non-steady-state): "
        f"{censored_volumes}"
    )

    denoised_img = regress_confounds(
        raw_img, confounds_df, censor_mask, _get_mask_file(nifti_file)
    )

    if censored_volumes:
        denoised_img = interpolate_censored_img(
            denoised_img, censor_mask, method="cubic", extrapolate=True
        )

    hdr = raw_img.header.copy()
    hdr.set_data_dtype(np.float32)
    denoised_img = nib.Nifti1Image(
        np.asanyarray(denoised_img.dataobj, dtype=np.float32),
        raw_img.affine,
        hdr,
    )
    denoised_img.header.set_zooms(denoised_img.header.get_zooms()[:3] + (tr,))

    metadata = {
        "RepetitionTime": tr,
        "Confounds": list(confounds_df.columns),
        "FramewiseDisplacementThreshold": fd_threshold,
        "NumberOfNonSteadyStateVolumes": n_dummy_scans,
        "CensoredVolumes": censored_volumes,
        "NumberOfVolumesInterpolated": len(censored_volumes),
        "NumberOfVolumes": n_volumes,
        "NumberOfOriginalVolumes": n_original_volumes,
        "Truncated": n_keep is not None,
        "Standardize": "None",
    }

    return denoised_img, metadata


def main(
    src_dir, dst_dir, subject_id, tasks, space, motion, compcor, n_compcor, fd_threshold
):
    subject_id = str(subject_id).removeprefix("sub-")
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    _create_dataset_description(dst_dir)

    layout = _get_layout(src_dir)
    tasks = tasks or layout.get(subject=subject_id, target="task", return_type="id")
    if not tasks:
        LGR.critical(f"No tasks found for subject {subject_id}.")
        return

    for task in tasks:
        nifti_files = _get_nifti_files(layout, subject_id, task, space)
        if not nifti_files:
            LGR.critical(
                f"No NIfTI files for subject {subject_id}, task {task}, space {space}."
            )
            continue

        for nifti_file in nifti_files:
            LGR.info(f"Denoising: {nifti_file}")
            denoised_img, metadata = denoise_nifti_img(
                nifti_file, task, motion, compcor, n_compcor, fd_threshold
            )

            subject_dir = dst_dir / Path(nifti_file).relative_to(src_dir).parent
            subject_dir.mkdir(parents=True, exist_ok=True)

            denoised_nifti_file = subject_dir / Path(nifti_file).name.replace(
                "desc-preproc_bold", "desc-denoised_bold"
            )
            denoised_img.to_filename(denoised_nifti_file)

            metadata["Sources"] = [str(Path(nifti_file).relative_to(src_dir))]
            with open(str(denoised_nifti_file).replace(".nii.gz", ".json"), "w") as f:
                json.dump(metadata, f, indent=4)

            LGR.info(f"Saved denoised image to: {denoised_nifti_file}")


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
