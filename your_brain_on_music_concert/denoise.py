"""
Simple denoising of fMRIPrep outputs.

Confounds are selected with nilearn's ``load_confounds`` and regressed out with ``clean_img``:
    - Motion parameters (default: "full", 24 parameters).
    - Cosine regressors from fMRIPrep (high-pass filtering). These are always included since
      fMRIPrep computes aCompCor on high-pass filtered data.
    - aCompCor components (default: "anat_separated" with 5 components per WM and CSF mask).

Non-steady-state volumes are removed with the sample mask returned by ``load_confounds`` and
each voxel is z-scored. Denoised images are saved in the destination directory, mirroring the
fMRIPrep directory structure, with "desc-preproc_bold" replaced by "desc-denoised_bold".
"""

import argparse, json
from pathlib import Path

import bids, nibabel as nib, numpy as np
from bidsaid.logging import setup_logger
from bidsaid.metadata import get_tr
from nilearn.image import clean_img
from nilearn.interfaces.fmriprep import load_confounds

LGR = setup_logger(__name__)


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
        "--subject",
        dest="subject",
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


def denoise_nifti(nifti_file, motion, compcor, n_compcor):
    confounds_df, sample_mask = load_confounds(
        str(nifti_file),
        strategy=("motion", "high_pass", "compcor"),
        motion=motion,
        compcor=compcor,
        n_compcor=n_compcor,
        demean=True,
    )
    LGR.info(f"Using the following confounds: {list(confounds_df.columns)}")

    n_volumes = nib.load(nifti_file).shape[-1]
    n_removed_volumes = 0 if sample_mask is None else n_volumes - len(sample_mask)
    LGR.info(f"Number of non-steady-state volumes removed: {n_removed_volumes}")

    tr = get_tr(nifti_file)
    denoised_img = clean_img(
        str(nifti_file),
        confounds=confounds_df,
        detrend=False,
        standardize="zscore_sample",
        t_r=tr,
        mask_img=_get_mask_file(nifti_file),
    )

    denoised_img = nib.Nifti1Image(
        np.asanyarray(denoised_img.dataobj, dtype=np.float32),
        denoised_img.affine,
        denoised_img.header,
    )
    denoised_img.header.set_data_dtype(np.float32)
    denoised_img.header.set_zooms(denoised_img.header.get_zooms()[:3] + (tr,))
    denoised_img.header.set_xyzt_units("mm", "sec")

    metadata = {
        "RepetitionTime": tr,
        "Confounds": list(confounds_df.columns),
        "NumberOfVolumesRemoved": n_removed_volumes,
        "Standardize": "zscore_sample",
    }

    return denoised_img, metadata


def main(src_dir, dst_dir, subject, tasks, space, motion, compcor, n_compcor):
    subject = str(subject).removeprefix("sub-")
    src_dir = Path(src_dir).resolve()
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    _create_dataset_description(dst_dir)

    layout = _get_layout(src_dir)
    tasks = tasks or layout.get(subject=subject, target="task", return_type="id")
    if not tasks:
        LGR.critical(f"No tasks found for subject {subject}.")

    for task in tasks:
        nifti_files = _get_nifti_files(layout, subject, task, space)
        if not nifti_files:
            LGR.critical(
                f"No NIfTI files for subject {subject}, task {task}, space {space}."
            )
            continue

        for nifti_file in nifti_files:
            LGR.info(f"Denoising: {nifti_file}")
            denoised_img, metadata = denoise_nifti(
                nifti_file, motion, compcor, n_compcor
            )

            subject_dir = Path(dst_dir) / Path(nifti_file).relative_to(src_dir).parent
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
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    main(**vars(args))
