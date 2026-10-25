"Shared utilities"

import base64, shutil, subprocess
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd
from nilearn.image import resample_to_img

from bidsaid.logging import setup_logger
from bidsaid.metadata import needs_resampling
from bidsaid.qc import get_n_censored_volumes

LGR = setup_logger(__name__)


def delete_file(filename):
    filename = Path(filename)
    if filename.exists():
        filename.unlink()


VALID_TASK_NAMES = {
    "kids": ["nback", "mtlr", "mtle", "flanker", "princess"],
    "adults": ["nback", "mtlr", "mtle", "flanker", "simplegng", "complexgng"],
}

TASK_CONTRASTS = {
    "kids": {
        "nback": (
            "1-back_vs_center",
            "2-back_vs_center",
            "2-back_vs_1-back",
        ),
        "mtle": ("neutral_encoding",),
        "mtlr": ("neutral_retrieval",),
        "princess": ("switch_vs_nonswitch",),
        "flanker": (
            "incongruent_vs_congruent",
            "nogo_vs_neutral",
            "nogo",
            "incongruent",
            "congruent",
        ),
    },
    "adults": {
        "nback": ("2-back_vs_0-back",),
        "mtle": (
            "aversive_encoding_vs_neutral_encoding",
            "neutral_encoding",
        ),
        "mtlr": (
            "aversive_retrieval_vs_neutral_retrieval",
            "neutral_retrieval",
        ),
        "flanker": (
            "incongruent_vs_congruent",
            "nogo_vs_neutral",
            "nogo",
            "incongruent",
            "congruent",
        ),
        "simplegng": ("nogo_vs_go",),
        "complexgng": ("nogo_vs_go",),
    },
}

# For the kids data, its fully within crossover design so [10 - placebo] - [5 - placebo]) = 10 - 5
CONTRAST_CODES = {
    "kids": ("5_vs_0", "10_vs_0", "10_vs_5", "mph_vs_placebo", "mean"),
    "adults": ("mph_vs_placebo", "mean"),
}


def get_first_level_gltsym_codes(cohort, task, analysis_type, caller):
    contrasts = TASK_CONTRASTS[cohort][task]
    if analysis_type == "gPPI":
        contrasts = modify_contrast_names(contrasts)

    return (
        tuple(f"{contrast}#0_Coef" for contrast in contrasts)
        if caller == "extract_betas"
        else contrasts
    )


def get_second_level_glt_codes(cohort):
    return CONTRAST_CODES[cohort]


def modify_contrast_names(contrasts):
    modified_contrasts = []
    for contrast in contrasts:
        if "_vs_" in contrast:
            modified_contrasts.append(
                f"PPI_{contrast.split('_vs_')[0]}_vs_PPI_{contrast.split('_vs_')[1]}"
            )

        else:
            modified_contrasts.append(f"PPI_{contrast}")

    return modified_contrasts


def get_beta_names(gltsyms, add_coef_str=False, create_sub_conditions=True):
    if isinstance(gltsyms, str):
        gltsyms = [gltsyms]

    if not create_sub_conditions:
        return gltsyms

    beta_names = []
    for gltsym in gltsyms:
        names = [] if "_vs_" not in gltsym else gltsym.split("_vs_")
        names += [gltsym]

        beta_names.extend(names)

    if add_coef_str:
        beta_names = [
            f"{name}#0_Coef" if not name.endswith("#0_Coef") else name
            for name in beta_names
        ]

    return list(set(beta_names))


def get_contrast_name_from_file(filename):
    filename = Path(filename).name

    return filename.split("desc-")[-1].split("_betas")[0]


def get_contrast_entity_key(input_str):
    input_str = Path(input_str).name

    return "contrast" if "_vs_" in input_str else "condition"


def resample_seed_img(seed_img, subject_nifti_img):
    if needs_resampling(seed_img, subject_nifti_img):
        seed_img = resample_to_img(
            seed_img, subject_nifti_img, interpolation="nearest", copy_header=True
        )

    return seed_img


def get_coordinate_from_filename(filepath, replace_underscore=True):
    filepath = Path(filepath)
    possible_coordinate = ""

    markers = ("_sphere_mask_", "-sphere_mask_", "_cluster_mask_", "_individual_betas_")
    for marker in markers:
        if marker in filepath.name:
            possible_coordinate = filepath.name.split(marker)[1]
            suffix = "".join(filepath.suffixes[3:])
            possible_coordinate = possible_coordinate.removesuffix(suffix)
            if replace_underscore:
                possible_coordinate = possible_coordinate.replace("_", ",")

            break

    return possible_coordinate


def create_beta_files(
    stats_file,
    beta_dir,
    afni_img_path,
    cohort,
    task,
    analysis_type,
    out_dir=None,
    overwrite=True,
):
    first_level_gltsyms = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="extract_betas"
    )
    beta_names = get_beta_names(first_level_gltsyms, add_coef_str=True)

    for beta_name in beta_names:
        beta_file = beta_dir / stats_file.name.replace(
            "stats", beta_name.replace("#0_Coef", "_betas")
        )
        if overwrite:
            delete_file(beta_file)

        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 3dbucket "
            f"{stats_file}'[{beta_name}]' "
            f"-prefix {beta_file} "
            "-overwrite"
        )
        LGR.info(f"Extracting {beta_name} betas: {cmd}")

        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception:
            LGR.critical(f"The following command failed: {cmd}", exc_info=True)

        if out_dir and beta_file.exists():
            path = Path(out_dir) / beta_file.name
            delete_file(path)
            shutil.move(beta_file, out_dir)


def get_nontarget_dose(second_level_glt_code, cohort):
    if second_level_glt_code == "mean":
        return []

    # For kids mph_vs_placebo, all three original doses (0, 5, 10) are needed
    # since 5 and 10 are averaged into a single mph image at the second level
    if cohort == "kids" and second_level_glt_code == "mph_vs_placebo":
        return []

    doses = {"kids": {"0", "5", "10"}, "adults": {"mph", "placebo"}}

    return list(doses[cohort].difference(second_level_glt_code.split("_vs_")))


def needs_complete_cases(method):
    return False if method == "parametric" else True


def drop_dose_rows(
    data_table,
    nontarget_dose_list,
    only_complete_cases=False,
    return_removed_subjects=False,
):
    if not nontarget_dose_list and only_complete_cases is False:
        return data_table

    removed_subjects = []
    target_doses = (
        data_table.loc[
            ~data_table["dose"].astype(str).isin(nontarget_dose_list), "dose"
        ]
        .unique()
        .tolist()
    )
    target_doses = list(map(str, target_doses))
    data_table = data_table[data_table["dose"].astype(str).isin(target_doses)]
    if only_complete_cases:
        dose_counts = data_table.groupby("Subj")["dose"].nunique()
        removed_subjects = dose_counts[dose_counts < len(target_doses)].index.tolist()
        data_table = data_table[~data_table["Subj"].isin(removed_subjects)]
        total_subjects = data_table["Subj"].nunique()
        contrast_name = get_contrast_name_from_file(data_table["InputFile"].tolist()[0])

        LGR.warning(
            f"For contrast ({contrast_name}), the following subjects have been removed: {removed_subjects}. "
            f"A total of {total_subjects} unique subjects with the following doses remain: {target_doses}"
        )

    if return_removed_subjects:
        return data_table, removed_subjects
    else:
        return data_table


def get_group_labels(second_level_glt_code):
    return second_level_glt_code.split("_vs_")


def save_binary_mask(mask_img_fdata, affine, hdr, mask_filename):
    """To save as a true binary mask and prevent equality index due to floating point issues"""
    mask_img_fdata = mask_img_fdata.astype(np.int8)
    hdr.set_data_dtype(np.int8)

    mask_img = nib.nifti1.Nifti1Image(mask_img_fdata, affine, hdr)

    nib.save(mask_img, mask_filename)


# pd.read_csv(exclude_niftis_files, sep=None, engine="python") fails in cases
# where there is only one column and row
def _get_dataframe(filename):
    try:
        return pd.read_excel(filename)
    except:
        pass

    df = pd.read_csv(filename, sep=None, engine="python")
    if "nifti_prefix_filename" not in df.columns:
        # Any separator will work
        df = pd.read_csv(filename)
        if "nifti_prefix_filename" not in df.columns:
            raise Exception(
                "`exclude_niftis_file` must contain a column named 'nifti_prefix_filename'."
            )

    return df


def delete_dir(dirname):
    """
    Delete dir to prevent file pollution due to re-running pipeline. Differences in number
    of clusters could occur so not all files will be overwritten. Use only for the "get_cluster_results"
    and "extract_individual_betas" pipelines. These are were this issue occurs and were the pipeline is not
    expected to run in parallel and write to the same output directory like "second_level.py". Also
    used for the "first_level" pipeline to clean out directory due to changes in exclusion criteria
    """
    if Path(dirname).exists():
        shutil.rmtree(dirname, ignore_errors=True)


def create_condition_label_str(beta_name):
    if "_vs_" in beta_name:
        first_condition_label, second_condition_label = beta_name.split("_vs_")
        condition_label = f"{first_condition_label} - {second_condition_label}"
    else:
        condition_label = f"{beta_name} only"

    return condition_label


def embed_img(image_path):
    """
    Reads the bytes from the image, then converts to base64,
    its binary-to-text encoding that uses 64 printable characters
    to represent each 6-bit segment of a sequence of byte values
    (https://en.wikipedia.org/wiki/Base64)

    It allows an image to be embedded in an html file, which will be
    completely self-contained and won't the file path to the image.
    """
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    return f"data:image/png;base64,{b64}"


def get_beta_files(analysis_dir, task, first_level_glt_label):
    return sorted(
        list(
            Path(analysis_dir).rglob(
                f"*{task}*_desc-{first_level_glt_label}_betas.nii.gz"
            )
        )
    )


def exclude_beta_files(beta_files, exclude_niftis_file):
    if not exclude_niftis_file:
        return beta_files

    excluded_niftis_prefixes = _get_dataframe(exclude_niftis_file)[
        "nifti_prefix_filename"
    ].tolist()

    LGR.info(
        (
            "Beta image files starting with the following prefixes "
            f"will be excluded: {excluded_niftis_prefixes}"
        )
    )

    return [
        beta_file
        for beta_file in beta_files
        if not any(
            Path(beta_file).name.startswith(excluded_niftis_prefix)
            for excluded_niftis_prefix in excluded_niftis_prefixes
        )
    ]


def get_subject_beta_filenames(
    data_table,
    first_level_glt_label,
    beta_name,
    parent_path=None,
    input_file_col="InputFile",
):
    subject_beta_filenames = data_table[input_file_col].tolist()

    if first_level_glt_label == beta_name:
        return subject_beta_filenames

    subject_beta_filenames = [
        str(file).replace(f"_desc-{first_level_glt_label}", f"_desc-{beta_name}")
        for file in subject_beta_filenames
    ]

    if parent_path:
        subject_beta_filenames = [
            next(parent_path.rglob(f"*{Path(file).name}*"), None)
            for file in subject_beta_filenames
        ]
        subject_beta_filenames = [
            str(file) if file else float("NaN") for file in subject_beta_filenames
        ]

    return subject_beta_filenames


def compute_average_betas(
    data_table,
    subject_beta_filenames,
    mask_filename,
    mask_origin="cluster",
    subject_col="Subj",
):
    subjects = data_table[subject_col].tolist()
    doses = data_table["dose"].tolist()
    average_betas = np.full(data_table.shape[0], np.nan)
    mask_img = nib.load(mask_filename)

    if mask_origin == "seed":
        mask_img = resample_seed_img(mask_img, nib.load(subject_beta_filenames[0]))

    for subject, dose, subject_beta_filename in zip(
        subjects, doses, subject_beta_filenames
    ):
        subject_mask = (data_table[subject_col] == subject) & (
            data_table["dose"] == dose
        )
        if pd.isna(subject_beta_filename):
            average_betas[subject_mask] = float("NaN")
            continue

        subject_beta_filename = Path(subject_beta_filename)
        beta_img = nib.load(subject_beta_filename)
        beta_img_fdata = beta_img.get_fdata()
        average_beta = beta_img_fdata[mask_img.get_fdata() == 1].mean()
        average_betas[subject_mask] = average_beta

    return average_betas


def get_individual_interpretations(
    data_table, beta_name, mask_origin, analysis_type, remove_PPI_prefix=False
):
    if remove_PPI_prefix:
        beta_name = beta_name.replace("PPI_", "")

    betas = data_table[
        f"{analysis_type.upper()}_Individual_{mask_origin.capitalize()}_Beta"
    ].to_numpy(copy=True)
    if "_vs_" in beta_name:
        first_condition_label, second_condition_label = beta_name.split("_vs_")
        interpretations = np.where(
            np.isnan(betas) | (betas == 0),
            "NaN",
            np.where(
                betas > 0,
                f"{first_condition_label} > {second_condition_label}",
                f"{second_condition_label} > {first_condition_label}",
            ),
        )
    else:
        descriptions = (
            ("activation", "deactivation")
            if "PPI_" not in beta_name
            else (
                "increased connectivity with seed roi",
                "decreased connectivity with seed roi",
            )
        )
        interpretations = np.where(
            np.isnan(betas) | (betas == 0),
            "NaN",
            np.where(betas > 0, descriptions[0], descriptions[1]),
        )

    interpretations[interpretations == "NaN"] = np.nan

    return interpretations.tolist()


def get_qc_info(subject_beta_file):
    all_censored_file_name = (
        Path(subject_beta_file).name.split("desc-")[0] + "desc-all_censored_volumes.1D"
    )
    parent_path = Path(subject_beta_file).parent
    if parent_path.name == "betas":
        parent_path = parent_path.parent

    all_censored_file = parent_path / all_censored_file_name
    high_motion_file = all_censored_file.parent / all_censored_file.name.replace(
        "all_censored_volumes", "high_motion_outliers_only"
    )
    fd_before_file = all_censored_file.parent / all_censored_file.name.replace(
        "all_censored_volumes", "fd_before_censoring"
    )
    fd_after_file = all_censored_file.parent / all_censored_file.name.replace(
        "all_censored_volumes", "fd_after_censoring"
    )

    info = {}
    if all_censored_file.exists() and high_motion_file.exists():
        n_high_motion = get_n_censored_volumes(high_motion_file)
        n_dummy_scans = get_n_censored_volumes(all_censored_file) - n_high_motion
        info["n_censored_volumes"] = n_high_motion
        info["n_dummy_scans"] = n_dummy_scans
    else:
        info["n_censored_volumes"] = np.nan
        info["n_dummy_scans"] = np.nan

    if fd_before_file.exists() and fd_after_file.exists():
        info["mean_fd_before_censoring"] = float(np.mean(np.loadtxt(fd_before_file)))
        info["mean_fd_after_censoring"] = float(np.mean(np.loadtxt(fd_after_file)))
    else:
        info["mean_fd_before_censoring"] = np.nan
        info["mean_fd_after_censoring"] = np.nan

    return info


def standardize_doses(data_table, cohort):
    data_table = data_table.dropna(subset=["dose"])

    if cohort != "kids":
        return data_table

    data_table["dose"] = data_table["dose"].astype(int).astype(str)

    return data_table


def validate_optional_path(path):
    return Path(path) if path and Path(path).exists() else None


def save_tabular_data(
    data_table,
    output_dir,
    output_csv_name,
    first_level_glt_label,
    beta_name,
    add_condition_entity_key,
    save_excel_version,
):
    data_filename = output_dir / output_csv_name
    data_filename.parent.mkdir(parents=True, exist_ok=True)

    if add_condition_entity_key:
        data_filename = data_filename.parent / data_filename.name.replace(
            first_level_glt_label, f"{first_level_glt_label}_condition-{beta_name}"
        )

    data_table.to_csv(data_filename, sep=",", index=None)
    if save_excel_version:
        data_table.to_excel(str(data_filename).replace(".csv", ".xlsx"), index=False)


def format_beta_df(analysis_type, beta_coefficient_df, method=None):
    if analysis_type == "gPPI":
        beta_coefficient_df["GPPI_Units_of_Beta_Coefficient"] = "unitless"

    if any(col.startswith("GLM") for col in beta_coefficient_df.columns):
        beta_coefficient_df["GLM_Units_of_Beta_Coefficient"] = (
            "percent (percent signal change)"
        )

    for drop_col in ["InputFile"]:
        if drop_col in beta_coefficient_df.columns:
            beta_coefficient_df = beta_coefficient_df.drop(columns=[drop_col])

    beta_coefficient_df["Analysis_Type"] = (
        f"{method} {analysis_type}" if method else analysis_type
    )

    beta_coefficient_df["Subj"] = [
        x.removeprefix("sub-") for x in beta_coefficient_df["Subj"].tolist()
    ]

    beta_coefficient_df.columns = [
        col.replace(" ", "_") for col in beta_coefficient_df.columns
    ]

    return beta_coefficient_df
