import argparse, shutil, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path

import bids, nibabel as nib, numpy as np, pandas as pd
from nilearn.masking import intersect_masks
from nilearn.image import resample_to_img
from pandas.api.types import is_object_dtype, is_string_dtype

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger
from bidsaid.io import compress_image
from bidsaid.cli import none_or

from _denoising import remove_collinear_columns
from _report import HTMLReport
from _second_level_utils import (
    estimate_noise_smoothness,
    perform_cluster_simulation,
    threshold_palm_output,
)
from _utils import (
    drop_dose_rows,
    exclude_beta_files,
    get_beta_files,
    get_qc_info,
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    get_second_level_glt_codes,
    get_nontarget_dose,
    get_group_labels,
    save_binary_mask,
    standardize_doses,
    needs_complete_cases,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Perform second level analysis. Cool paper: https://onlinelibrary.wiley.com/doi/10.1002/hbm.70437"
    )
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="Path to BIDS directory."
    )
    parser.add_argument(
        "--deriv_dir",
        dest="deriv_dir",
        required=False,
        default=None,
        help="Root of the derivatives directory.",
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help=(
            "Path to directory containing the extracted beta coefficients. "
            "Files are grabbed recursively, only the naming of the coefficients "
            "need to be BIDS compliant."
        ),
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination directory for analysis.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        required=True,
        help="Template space (i.e. 'MNIPediatricAsym_cohort-1_res-2')",
    )
    parser.add_argument(
        "--gm_probseg_img_path",
        dest="gm_probseg_img_path",
        default=None,
        required=False,
        type=none_or(),
        help=(
            "The probability mask for gray matter voxels. Should approximately be in same space as the template."
            "Used to exclude non-gray matter voxels in the group mask and exclude false activations from "
            "white matter and ventricles. http://jpeelle.net/mri/misc/creating_explicit_mask.html"
        ),
    )
    parser.add_argument(
        "--gm_mask_threshold",
        dest="gm_mask_threshold",
        default=0.20,
        type=float,
        required=False,
        help="The probability for gray matter voxels in the mask. See: https://pmc.ncbi.nlm.nih.gov/articles/PMC3812339/",
    )
    parser.add_argument(
        "--apriori_img_path",
        dest="apriori_img_path",
        default=None,
        required=False,
        type=none_or(),
        help=(
            "Reduce the search space to the mask. If ``gm_probseg_img_path`` is supplied, the intersected group "
            "mask is restricted to the gray matter mask then restricted to the apriori mask. Note that "
            "cluster-based inference relies on spatially contiguous voxels, it is possible for clusters with the highest "
            "voxel statistical value to be in an apriori region and spread to non-apriori regions. This can result in the cluster  "
            "being deemed insignificant since the cluster extent can be reduced significantly. Recommend the apriori mask to be "
            "large regions (e.g., networks). Tradeoff is increased sensitivity to clusters in mask via the "
            "small volume correction approach - 3dclustsim will perform simulations in the mask and output a smaller k threshold "
            "required for significance while still controlling for false positives "
            "(https://www.biologicalpsychiatryjournal.com/article/S0006-3223(25)01251-X/fulltext). Mask should approximately be in "
            "the same space as template."
        ),
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--first_level_glt_label",
        dest="first_level_glt_label",
        required=False,
        default=None,
        help=(
            "Name of a single valid first level gltlabel for the given task. "
            "Used to run several labels of a task in parallel if there are many. "
            "Particularly useful is using nonparametric."
        ),
    )
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        choices=["glm", "gPPI"],
        required=True,
        help="The type of analysis performed (glm or gPPI).",
    )
    parser.add_argument(
        "--group_mask_threshold",
        dest="group_mask_threshold",
        default=1.0,
        required=False,
        type=float,
        help="Value between 0 to 1 denoting the level of intersection for the masks.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=False,
        default=None,
        help=("Path to Apptainer image of Afni with R. Required if using parametric."),
    )
    parser.add_argument(
        "--fsl_img_path",
        dest="fsl_img_path",
        required=False,
        default=None,
        help=(
            "Path to apptainer image of FSL with Palm using Octave. "
            "Required if method is nonparametric, palm is not in path, "
            "matlab not module and fsl not module"
        ),
    )
    parser.add_argument(
        "--categorical_covariates",
        dest="categorical_covariates",
        default=[],
        required=False,
        nargs="*",
        help=(
            "Categorical covariates to include in the second level model (e.g., sex race ethnicity). "
            "For nonparametric only, these covariates will be dummy-coded and mean-centered to prevent "
            "E[Y] = B0 + B1E[Xc] = E[Y] = B0 + B1(0), which allows the intercept to be the mean of Y instead "
            "of the mean of the reference group coded 0. "
            "Should be a single string with variables separated by space."
        ),
    )
    parser.add_argument(
        "--numerical_covariates",
        dest="numerical_covariates",
        default=["n_censored_volumes"],
        required=False,
        nargs="*",
        help=(
            "Numerical covariates to include in the second level model (e.g., n_censored_volumes age). "
            "These will be mean-centered for both parametric and nonparametric methods. "
            "Should be a single string with variables separated by space."
        ),
    )
    parser.add_argument(
        "--method",
        dest="method",
        default="parametric",
        choices=["parametric", "nonparametric"],
        required=False,
        help=(
            "Whether to use 3dlmer (parametric) or Palm (nonparametric). "
            "Nonparametric doesn't assume the error distribution of the data and better controls false positives."
            "Refer to: https://www.pnas.org/doi/abs/10.1073/pnas.1602413113?utm_source "
            "and https://onlinelibrary.wiley.com/doi/10.1002/hbm.23115. If parametric is used then the acf method method should "
            "Nonparametric is stochastic be used on the residuals to estimate smoothness and determine the appropriate cluster "
            "size via simulations. However, Palm sets seed to 0 by default for reproducibility. Though different results "
            "are expected if using Octave: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)UserGuide.html"
        ),
    )
    parser.add_argument(
        "--n_permutations",
        dest="n_permutations",
        default="auto",
        required=False,
        help=(
            "If method is nonparametric, the number of permutations to pass to Palm. "
            "For 'auto' the permutation is computed by doing 2^N since each contrast be "
            "either positive or negative in the two sample paired case or the single sample test. "
            "If the max permutation exceeds 10,000, then the permutation is set to 10,000. "
            "Lowest p-value -log10(1/1e4) in that case."
        ),
    )
    parser.add_argument(
        "--tfce_H",
        dest="tfce_H",
        default=2,
        required=False,
        type=float,
        help=(
            "The height power. Higher values weigh signal intensity more. "
            "See https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf"
        ),
    )
    parser.add_argument(
        "--tfce_E",
        dest="tfce_E",
        default=0.5,
        required=False,
        type=float,
        help=(
            "The extent power. Higher values weigh cluster extent more. "
            "See https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf"
        ),
    )
    parser.add_argument(
        "--tfce_C",
        dest="tfce_C",
        default=6,
        required=False,
        help=(
            "The connectivity to use for the nonparametric approach. "
            "See https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf"
        ),
    )
    parser.add_argument(
        "--nonparametric_cluster_correction_p",
        dest="nonparametric_cluster_correction_p",
        default=0.05,
        type=float,
        required=False,
        help=(
            "Significance threshold for cluster significance for nonparametric. "
            "This script uses the threshold free cluster enhancement approach which "
            "eliminates the need to select an arbritrary threshold for the voxels (cluster-forming threshold)"
        ),
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=1,
        type=int,
        required=False,
        help="Number of cores to use for 3dlmer when method is parametric.",
    )
    parser.add_argument(
        "--exclude_niftis_file",
        dest="exclude_niftis_file",
        default=None,
        required=False,
        type=none_or(),
        help=(
            "Path to a file containing prefixes of the filename of the NIfTI images to exclude. "
            "Can list the fill name of the file (no parent directories) to exlude that specific file "
            "or can include the prefix (i.e., 'sub-101_task-nback_ses-01_space-MNI' or 'sub-101') to exclude all files starting "
            "with that prefix. Should contain a single column named 'nifti_prefix_filename' "
        ),
    )

    return parser


@dataclass
class DataContainer:
    grouping_columns: set[str] = field(
        default_factory=lambda: set(
            ["Subj", "session_id", "InputFile", "dose", "dose_mg"]
        ),
    )
    categorical_covariates: list[str] = field(default_factory=list)
    numerical_covariates: list[str] = field(default_factory=list)

    def __post_init__(self):
        conflicting_categorical = set(self.categorical_covariates).intersection(
            self.grouping_columns
        )
        conflicting_numerical = set(self.numerical_covariates).intersection(
            self.grouping_columns
        )
        if conflicting_categorical or conflicting_numerical:
            all_conflicting = conflicting_categorical | conflicting_numerical
            LGR.warning(
                f"The following variables are grouping columns and cannot be "
                f"included as covariates: {all_conflicting}. "
                "They will be removed as covariates."
            )
            self.categorical_covariates = list(
                set(self.categorical_covariates).difference(self.grouping_columns)
            )
            self.numerical_covariates = list(
                set(self.numerical_covariates).difference(self.grouping_columns)
            )

    @property
    def included_covariates(self) -> list[str]:
        return self.categorical_covariates + self.numerical_covariates

    @property
    def columns_to_keep(self) -> list[str]:
        return list(self.grouping_columns) + self.included_covariates

    @staticmethod
    def get_glt_codes(cohort: str) -> str:
        glt_codes = {
            "kids": (
                "-gltCode 5_vs_0 'dose : 1*5 -1*0' ",
                "-gltCode 10_vs_0 'dose : 1*10 -1*0' ",
                "-gltCode 10_vs_5 'dose : 1*10 -1*5' ",
                "-gltCode mph_vs_placebo 'dose : 0.5*10 +0.5*5 -1*0' ",
                "-gltCode mean 'dose : {mean_code}' ",
            ),
            "adults": (
                "-gltCode mph_vs_placebo 'dose : 1*mph -1*placebo' ",
                "-gltCode mean 'dose : {mean_code}' ",
            ),
        }

        return glt_codes[cohort]

    @property
    def afni_regressor_columns(self) -> list[str]:
        return ["dose"] + self.included_covariates


def get_subjects(beta_files):
    return sorted([get_entity_value(file.name, "sub") for file in beta_files])


def drop_constant_columns(data_table):
    constant_columns = [
        col for col in data_table.columns if data_table[col].nunique() == 1
    ]

    LGR.info(f"Dropping the following constant columns: {constant_columns}")

    return data_table.drop(columns=constant_columns), constant_columns


def needs_within_subject_averaging(cohort, second_level_glt_code):
    if second_level_glt_code == "mean" or (
        cohort == "kids" and second_level_glt_code == "mph_vs_placebo"
    ):
        return True
    else:
        return False


def sort_data_table(data_table, subject_only=False):
    sort_cols = ["Subj"] if subject_only else ["Subj", "dose"]

    return data_table.sort_values(sort_cols).reset_index(drop=True)


def recode_doses(glt_data_table, cohort, second_level_glt_code):
    if not (cohort == "kids" and second_level_glt_code == "mph_vs_placebo"):
        return glt_data_table

    glt_data_table = glt_data_table.copy()
    glt_data_table["dose"] = glt_data_table["dose"].astype(str)
    glt_data_table["dose"] = glt_data_table["dose"].apply(
        lambda x: "placebo" if x == "0" else "mph"
    )

    LGR.info(
        f"Recoded kids dose column for {second_level_glt_code}: "
        f"5 to mph, 10 to mph, and 0 to placebo "
        f"Dose value counts after recoding:\n{glt_data_table['dose'].value_counts().to_string()}"
    )

    return glt_data_table


def collapse_within_subject_doses(glt_data_table, datacontainer, second_level_glt_code):
    groupby_keys = ["Subj"] if second_level_glt_code == "mean" else ["Subj", "dose"]

    glt_data_table, categorical_cols, numerical_cols = set_column_types(
        glt_data_table, datacontainer
    )
    glt_data_table = filter_glt_data_table(
        glt_data_table, categorical_cols, numerical_cols
    )

    agg_rules = {col: "mean" for col in numerical_cols}
    agg_rules.update({col: "first" for col in categorical_cols})

    if agg_rules:
        collapsed_data_table = (
            glt_data_table.groupby(groupby_keys, sort=True).agg(agg_rules).reset_index()
        )
    else:
        collapsed_data_table = (
            glt_data_table[groupby_keys]
            .drop_duplicates()
            .sort_values(groupby_keys)
            .reset_index(drop=True)
        )

    collapsed_data_table = collapsed_data_table.sort_values(groupby_keys).reset_index(
        drop=True
    )

    LGR.info(
        f"Collapsed within-subject doses using groupby {groupby_keys}: "
        f"{len(glt_data_table)} rows became {len(collapsed_data_table)} rows"
    )

    return collapsed_data_table


def replace_whitespace_with_underscores(data_table):
    for column in data_table.columns:
        if is_object_dtype(data_table[column]) or is_string_dtype(data_table[column]):
            data_table[column] = data_table[column].str.replace(" ", "_")

    return data_table


def order_columns_names(columns):
    return (
        ["Subj", "dose"]
        + [name for name in columns if name not in ["Subj", "dose", "InputFile"]]
        + ["InputFile"]
    )


def create_data_table(bids_dir, datacontainer, subject_list, beta_files, cohort):
    bids_dir = Path(bids_dir)
    participants_df = pd.read_csv(bids_dir / "participants.tsv", sep="\t")

    session_files = sorted(list(bids_dir.rglob("sub-*_sessions.tsv")))
    sessions_dfs = []
    for session_file in session_files:
        if (sub_id := get_entity_value(session_file, "sub")) not in subject_list:
            continue

        df = pd.read_csv(session_file, sep="\t")
        df["participant_id"] = f"sub-{sub_id}"

        subject_beta_files = [str(file) for file in beta_files if sub_id in str(file)]
        for subject_beta_file in subject_beta_files:
            ses_id = get_entity_value(
                subject_beta_file, "ses", return_entity_prefix=True
            )
            df.loc[df["session_id"] == ses_id, "InputFile"] = subject_beta_file

            qc_info = get_qc_info(subject_beta_file)
            for key, value in qc_info.items():
                df.loc[df["session_id"] == ses_id, key] = value

        sessions_dfs.append(df)

    all_sessions = pd.concat(sessions_dfs, ignore_index=True)
    data_table = all_sessions.merge(participants_df, on="participant_id")
    # AFNI 26 requires first column to be named "Subj"
    data_table = data_table.rename(columns={"participant_id": "Subj"})
    column_names = order_columns_names(data_table.columns)
    data_table = data_table.loc[:, column_names]
    data_table = data_table.dropna(how="all", axis=1)
    data_table = standardize_doses(data_table, cohort)

    for numerical_var in set(datacontainer.numerical_covariates).intersection(
        data_table.columns
    ):
        data_table[numerical_var] = data_table[numerical_var].astype(float)

    data_table = replace_whitespace_with_underscores(data_table)

    data_table, constant_columns = drop_constant_columns(data_table)

    important_columns = (
        ["Subj", "dose"] + datacontainer.included_covariates + ["InputFile"]
    )
    # Only drop na rows when na is in important columns
    filtered_important_columns = list(
        set(important_columns).intersection(data_table.columns)
    )
    filtered_important_columns = order_columns_names(filtered_important_columns)
    data_table = data_table.dropna(subset=filtered_important_columns, axis=0)

    return sort_data_table(data_table), constant_columns, filtered_important_columns


def get_sample_sizes(data_table):
    dose_sample_sizes = data_table["dose"].value_counts().items()
    dose_sample_sizes_string = ", ".join(
        f"{name} (N={size})" for name, size in dose_sample_sizes
    )

    subject_dose_sample_sizes = dict(
        data_table["Subj"].value_counts().value_counts()
    ).items()
    subject_dose_sample_sizes_string = ", ".join(
        f"{num_doses} doses available (N={size})"
        for num_doses, size in subject_dose_sample_sizes
    )
    subject_dose_sample_sizes_string = subject_dose_sample_sizes_string.replace(
        "1 doses", "1 dose"
    )

    return dose_sample_sizes_string, subject_dose_sample_sizes_string


def create_group_mask(
    dst_dir,
    layout,
    task,
    space,
    group_mask_threshold,
    filtered_beta_files,
    gm_probseg_img_path,
    gm_mask_threshold,
    apriori_img_path,
    method,
    entity_key,
    first_level_glt_label,
    second_level_glt_code=None,
):
    if second_level_glt_code:
        group_mask_filename = (
            dst_dir
            / "group_masks"
            / f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}_desc-{method}_group_mask.nii.gz"
        )
    else:
        group_mask_filename = (
            dst_dir
            / "group_masks"
            / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-{method}_group_mask.nii.gz"
        )

    group_mask_filename.parent.mkdir(parents=True, exist_ok=True)

    subject_mask_files = []
    for filtered_beta_file in filtered_beta_files:
        sub_id = get_entity_value(filtered_beta_file, "sub")
        ses_id = get_entity_value(filtered_beta_file, "ses")

        mask_files = layout.get(
            scope="derivatives",
            subject=sub_id,
            session=ses_id,
            task=task,
            suffix="mask",
            extension="nii.gz",
            return_type="file",
        )

        mask_files = [mask_file for mask_file in mask_files if space in str(mask_file)]
        subject_mask_files.extend(mask_files)

    group_mask = intersect_masks(subject_mask_files, threshold=group_mask_threshold)
    group_mask = reduce_search_space(
        group_mask,
        gm_probseg_img_path,
        mask_type="gray_matter",
        gm_mask_threshold=gm_mask_threshold,
    )
    group_mask = reduce_search_space(group_mask, apriori_img_path, mask_type="apriori")

    save_binary_mask(
        group_mask.get_fdata(),
        group_mask.affine,
        group_mask.header.copy(),
        group_mask_filename,
    )

    nib.save(group_mask, group_mask_filename)

    return group_mask_filename


def reduce_search_space(group_mask, mask_img_path, mask_type, gm_mask_threshold=0.20):
    if not mask_img_path:
        return group_mask

    mask_img = nib.load(mask_img_path)
    resmapled_mask_img = resample_to_img(mask_img, group_mask, interpolation="nearest")

    if mask_type == "gray_matter":
        LGR.info(
            f"Thresholding gray metter probability mask using {Path(mask_img_path).name},  "
            f"any voxel with a probability greater than {gm_mask_threshold} "
            "will be retained in the group mask."
        )
        binarized_mask_image = resmapled_mask_img.get_fdata() > gm_mask_threshold
    else:
        binarized_mask_image = resmapled_mask_img.get_fdata() != 0

    group_mask_data = group_mask.get_fdata() * binarized_mask_image

    return nib.nifti1.Nifti1Image(group_mask_data, group_mask.affine, group_mask.header)


def get_glt_codes_str(data_table, datacontainer, cohort):
    glt_str = ""
    available_doses = sorted(data_table["dose"].astype(str).unique())
    for glt_code in datacontainer.get_glt_codes(cohort):
        level_str = glt_code.removeprefix("-gltCode").lstrip().split(" ")[0]
        if level_str == "mean":
            value = round(1 / len(available_doses), 4)
            dose_list = [f"{x}" for x in available_doses]
            mean_code = f"{value}*" + f" +{value}*".join(dose_list)
            glt_str += glt_code.format(mean_code=mean_code)
        elif "_vs_" not in glt_code:
            # In future if ever asked to look at single group
            glt_str += glt_code if level_str in available_doses else ""
        elif level_str == "mph_vs_placebo":
            required = {"mph", "placebo"} if cohort == "adults" else {"0", "5", "10"}
            if required.issubset(set(available_doses)):
                glt_str += glt_code
        else:
            dose_list = level_str.split("_vs_")
            if all(dose in available_doses for dose in dose_list):
                glt_str += glt_code

    return glt_str


def get_model_str(datacontainer):
    model_str = "+".join(datacontainer.afni_regressor_columns)
    model_str += "+(1|Subj)"

    LGR.info(f"The following model will be used: {model_str}")

    return model_str


def get_centering_str(datacontainer):
    numerical_vars = datacontainer.numerical_covariates
    if not numerical_vars:
        return ""

    qvars_str = "'" + ",".join(numerical_vars) + "'"
    centers_str = "'" + ",".join(["0"] * len(numerical_vars)) + "'"
    centering_str = f"-qVars {qvars_str} -qVarCenters {centers_str}"

    LGR.info(f"The following centering string will be used: {centering_str}")

    return centering_str


def generate_matrices_filenames(
    output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
):
    matrices_filenames_dict = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = (
        f"task-{task}_{entity_key}-{first_level_glt_label}"
        f"_gltcode-{second_level_glt_code}"
    )

    if "_vs_" in second_level_glt_code:
        matrices_filenames_dict["eb_file"] = (
            output_dir / f"{prefix}_desc-exchangeability_blocks.csv"
        )
    else:
        matrices_filenames_dict["eb_file"] = None

    matrices_filenames_dict["design_matrix_file"] = (
        output_dir / f"{prefix}_desc-design_matrix.csv"
    )
    matrices_filenames_dict["header_file"] = (
        output_dir / f"{prefix}_desc-header_names.txt"
    )
    matrices_filenames_dict["contrast_matrix_file"] = (
        output_dir / f"{prefix}_desc-contrast_matrix.csv"
    )

    return matrices_filenames_dict


def set_column_types(glt_data_table, datacontainer):
    categorical_cols = list(
        set(datacontainer.categorical_covariates).intersection(glt_data_table.columns)
    )
    numerical_cols = list(
        set(datacontainer.numerical_covariates).intersection(glt_data_table.columns)
    )

    for col in categorical_cols:
        glt_data_table[col] = glt_data_table[col].astype(str)

    for col in numerical_cols:
        glt_data_table[col] = glt_data_table[col].astype(float)

    return glt_data_table, categorical_cols, numerical_cols


def filter_glt_data_table(glt_data_table, categorical_cols, numerical_cols):
    cols_to_keep = ["Subj", "dose"] + categorical_cols + numerical_cols

    return glt_data_table[cols_to_keep]


def create_design_matrix(
    glt_data_table,
    datacontainer,
    include_intercept,
    average_within_subjects=False,
    second_level_glt_code=None,
):
    glt_data_table, categorical_cols, numerical_cols = set_column_types(
        glt_data_table, datacontainer
    )

    if average_within_subjects:
        glt_data_table = collapse_within_subject_doses(
            glt_data_table, datacontainer, second_level_glt_code
        )
    else:
        glt_data_table = filter_glt_data_table(
            glt_data_table, categorical_cols, numerical_cols
        )

    design_components = {}
    mean_center = lambda arr: arr - arr.mean()
    for col in numerical_cols:
        design_components.update({col: mean_center(glt_data_table[col].to_numpy())})

    # All categorical variables are constant across sessions and have been dropped
    # previously for dose pairisons to avoid rank deficiency
    # Categorical cols need to be mean centered so that no group can be coded as 0
    # If one categorical variable is binary (0 and 1), then
    # for 0 E[Y] = B0 + B1E[Xc] = E[Y] = B0 + B1(0), which forces the intercept to be
    # the reference group
    for col in categorical_cols:
        dummies = pd.get_dummies(
            glt_data_table[col], prefix=col, drop_first=True, dtype=int
        )
        design_components.update(
            {col: mean_center(dummies[col].to_numpy()) for col in dummies.columns}
        )

    design_matrix = pd.DataFrame(design_components)
    collinear_column_names = []
    if design_matrix.shape[1] > 1:
        regressor_positions = {
            index: col for index, col in enumerate(design_matrix.columns)
        }
        design_arr, regressor_positions, collinear_column_names = (
            remove_collinear_columns(design_matrix.to_numpy(), regressor_positions)
        )
        design_matrix = pd.DataFrame(
            design_arr, columns=list(regressor_positions.values())
        )

    if include_intercept:
        # Using row shape of original table for case where design matrix is empty
        intercept = [1] * (glt_data_table.shape[0])
        design_matrix.insert(0, "intercept", intercept)
    else:
        first_label, _ = get_group_labels(second_level_glt_code)
        dose_codes = np.where(glt_data_table["dose"].astype(str) == first_label, 1, -1)
        design_matrix.insert(0, "dose", dose_codes)
        subject_regressors = pd.get_dummies(
            glt_data_table["Subj"], prefix="", prefix_sep="", dtype=int
        )
        subject_regressors.columns = [
            f"{col}_intercept" for col in subject_regressors.columns
        ]
        design_matrix = pd.concat(
            [
                design_matrix.reset_index(drop=True),
                subject_regressors.reset_index(drop=True),
            ],
            axis=1,
        )

    residual_dof = design_matrix.shape[0] - design_matrix.shape[1]
    if residual_dof <= 0:
        LGR.warning(
            f"Insufficient degrees of freedom: {design_matrix.shape[0]} observations, "
            f"{design_matrix.shape[1]} regressors, residual DOF = {residual_dof}. "
            "Consider removing covariates."
        )

    return design_matrix, collinear_column_names, residual_dof


def create_contrast_matrix(design_matrix, contrast_matrix_filename):
    vector_pos = np.zeros(design_matrix.shape[1])
    vector_pos[0] = 1
    # Zeroes will show up as negative, that is fine -0 == 0 returns True
    vector_neg = vector_pos * -1

    np.savetxt(
        contrast_matrix_filename,
        np.array([vector_pos, vector_neg]),
        delimiter=",",
        fmt="%.4f",
    )


def create_header_file(design_matrix, header_file):
    with open(header_file, "w") as f:
        f.write(f"{','.join(design_matrix.columns.tolist())}")


def write_contrast_direction_file(
    output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
):
    prefix = f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}"
    contrast_direction_file = output_dir / f"{prefix}_desc-contrast_direction.txt"

    if "_vs_" in second_level_glt_code:
        first_label, second_label = get_group_labels(second_level_glt_code)
        positive_label = f"{first_label} > {second_label}"
        negative_label = f"{second_label} > {first_label}"
    else:
        positive_label = "above mean of 0"
        negative_label = "below mean of 0"

    with open(contrast_direction_file, "w") as f:
        f.write("# Positive direction contrast:\n")
        f.write(f"c1: {positive_label}\n")
        f.write("\n# Negative direction contrast:\n")
        f.write(f"c2: {negative_label}\n")


def create_mean_matrices(
    glt_data_table,
    datacontainer,
    output_dir,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    average_within_subjects,
):
    """
    Takes the data table and creates matrices for PALM specifically for means
    (e.g., 5, mean, etc).

    # PALM GUIDE: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)UserGuide.html

    According to the PALM guide, FSL design matrices need to be used.
    This is the guide for FSL design matrices:
    https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/glm.html?h=design
    """
    matrices_filenames_dict = generate_matrices_filenames(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    design_matrix, collinear_column_names, residual_dof = create_design_matrix(
        glt_data_table,
        datacontainer,
        include_intercept=True,
        average_within_subjects=average_within_subjects,
        second_level_glt_code=second_level_glt_code,
    )
    design_matrix.to_csv(
        matrices_filenames_dict["design_matrix_file"],
        sep=",",
        header=False,
        index=False,
    )
    create_header_file(design_matrix, matrices_filenames_dict["header_file"])

    create_contrast_matrix(
        design_matrix, matrices_filenames_dict["contrast_matrix_file"]
    )
    write_contrast_direction_file(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    report_info = {
        "constant_column_names": [],
        "collinear_column_names": collinear_column_names,
        "residual_dof": residual_dof,
    }

    return matrices_filenames_dict, report_info


def create_comparison_matrices(
    glt_data_table,
    datacontainer,
    output_dir,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    average_within_subjects,
):
    """
    Takes the data table and creates matrices for PALM specifically for comparisons
    (e.g., 5_vs_0).

    # PALM GUIDE: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)UserGuide.html

    According to the PALM guide, FSL design matrices need to be used.
    This is the guide for FSL design matrices:
    https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/glm.html?h=design
    """
    matrices_filenames_dict = generate_matrices_filenames(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    if average_within_subjects:
        glt_data_table = collapse_within_subject_doses(
            glt_data_table, datacontainer, second_level_glt_code
        )

    glt_data_table, constant_columns_names = drop_within_subject_constant_regressors(
        datacontainer, glt_data_table
    )

    eb_data = glt_data_table["Subj"].factorize()[0] + 1
    np.savetxt(matrices_filenames_dict["eb_file"], eb_data, delimiter=",", fmt="%d")

    design_matrix, collinear_column_names, residual_dof = create_design_matrix(
        glt_data_table,
        datacontainer,
        include_intercept=False,
        average_within_subjects=False,
        second_level_glt_code=second_level_glt_code,
    )
    design_matrix.to_csv(
        matrices_filenames_dict["design_matrix_file"],
        sep=",",
        header=False,
        index=False,
    )
    create_header_file(design_matrix, matrices_filenames_dict["header_file"])

    create_contrast_matrix(
        design_matrix, matrices_filenames_dict["contrast_matrix_file"]
    )
    write_contrast_direction_file(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    report_info = {
        "constant_column_names": constant_columns_names,
        "collinear_column_names": collinear_column_names,
        "residual_dof": residual_dof,
    }

    return matrices_filenames_dict, report_info


def set_permutations(true_max_permutation):
    max_permutation = min(true_max_permutation, 10000)

    LGR.info(f"Setting number of permutations to: {max_permutation}")

    return max_permutation


def compute_n_permutation(glt_data_table):
    n_subjects = len(glt_data_table["Subj"].unique())
    true_max_permutation = 2**n_subjects
    LGR.info(f"Maximum permutations possible = {true_max_permutation}")

    return set_permutations(true_max_permutation), true_max_permutation


def create_mean_image(
    subject_temp_merged_filename,
    subject_mean_image_filename,
    fsl_merge_call,
    fsl_maths_call,
    subject_files,
):
    subprocess.run(
        f"{fsl_merge_call} -t {subject_temp_merged_filename} {' '.join(subject_files)}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"{fsl_maths_call} {subject_temp_merged_filename} -Tmean {subject_mean_image_filename}",
        shell=True,
        check=True,
    )

    subject_temp_merged_filename.unlink()

    return subject_mean_image_filename


def concatenate_all_images(fsl_merge_call, concatenated_filename, files_to_merge):
    cmd = f"{fsl_merge_call} -t {concatenated_filename} {' '.join(files_to_merge)}"
    LGR.info(f"Concatenating images: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def delete_images(image_filenames):
    for image_filename in image_filenames:
        Path(image_filename).unlink()


def create_concatenated_image(
    output_dir,
    glt_data_table,
    fsl_img_path,
    use_native_fsl,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
):
    prefix = (
        f"task-{task}_{entity_key}-{first_level_glt_label}"
        f"_gltcode-{second_level_glt_code}"
    )

    concatenated_filename = output_dir / f"{prefix}_desc-group_concatenated.nii.gz"
    volume_order_filename = output_dir / f"{prefix}_desc-volume_order.tsv"

    if concatenated_filename.exists():
        concatenated_filename.unlink()

    fsl_merge_call = (
        "fslmerge"
        if use_native_fsl
        else f"apptainer exec -B /projects:/projects {fsl_img_path} fslmerge"
    )

    fsl_maths_call = (
        "fslmaths"
        if use_native_fsl
        else f"apptainer exec -B /projects:/projects {fsl_img_path} fslmaths"
    )

    has_duplicate_dose_rows = (
        "dose" in glt_data_table.columns
        and glt_data_table.groupby(["Subj", "dose"]).size().max() > 1
    )

    volume_records = []
    if second_level_glt_code == "mean":
        LGR.info("Averaging all doses within subjects for the mean contrast.")

        sorted_table = sort_data_table(
            glt_data_table,
            subject_only=True,
        )

        subject_mean_images = []

        for subject, group in sorted_table.groupby("Subj", sort=True):
            subject_files = group["InputFile"].tolist()

            subject_temp_merged = (
                concatenated_filename.parent / f"temp_{prefix}_{subject}_merged.nii.gz"
            )
            subject_mean_img = (
                concatenated_filename.parent / f"temp_{prefix}_{subject}_mean.nii.gz"
            )

            subject_mean_img = create_mean_image(
                subject_temp_merged,
                subject_mean_img,
                fsl_merge_call,
                fsl_maths_call,
                subject_files,
            )

            subject_mean_images.append(str(subject_mean_img))

            volume_records.append(
                {
                    "Subj": subject,
                    "dose": "mean",
                    "averaged": True,
                    "n_source_images": len(subject_files),
                    "source_images": "|".join(map(str, subject_files)),
                }
            )

        concatenate_all_images(
            fsl_merge_call,
            concatenated_filename,
            subject_mean_images,
        )

        delete_images(subject_mean_images)

    elif has_duplicate_dose_rows:
        LGR.info(
            "Detected duplicate Subj + dose rows. Averaging images within "
            f"Subj + dose groups for {second_level_glt_code}."
        )

        sorted_table = sort_data_table(glt_data_table)

        images_to_concat = []
        temp_files = []

        for (subject, dose), group in sorted_table.groupby(
            ["Subj", "dose"],
            sort=True,
        ):
            group_files = group["InputFile"].tolist()

            if len(group_files) > 1:
                subject_temp_merged = (
                    concatenated_filename.parent
                    / f"temp_{prefix}_{subject}_{dose}_merged.nii.gz"
                )
                subject_mean_img = (
                    concatenated_filename.parent
                    / f"temp_{prefix}_{subject}_{dose}_mean.nii.gz"
                )

                subject_mean_img = create_mean_image(
                    subject_temp_merged,
                    subject_mean_img,
                    fsl_merge_call,
                    fsl_maths_call,
                    group_files,
                )

                images_to_concat.append(str(subject_mean_img))
                temp_files.append(subject_mean_img)

                averaged = True

            else:
                images_to_concat.append(group_files[0])
                averaged = False

            volume_records.append(
                {
                    "Subj": subject,
                    "dose": dose,
                    "averaged": averaged,
                    "n_source_images": len(group_files),
                    "source_images": "|".join(map(str, group_files)),
                }
            )

        concatenate_all_images(
            fsl_merge_call,
            concatenated_filename,
            images_to_concat,
        )

        delete_images(temp_files)

    else:
        sorted_table = sort_data_table(glt_data_table)

        files_to_merge = sorted_table["InputFile"].tolist()

        for _, row in sorted_table.iterrows():
            volume_records.append(
                {
                    "Subj": row["Subj"],
                    "dose": row["dose"],
                    "averaged": False,
                    "n_source_images": 1,
                    "source_images": str(row["InputFile"]),
                }
            )

        concatenate_all_images(
            fsl_merge_call,
            concatenated_filename,
            files_to_merge,
        )

    volume_order = pd.DataFrame(volume_records)

    volume_order.insert(
        0,
        "volume_index",
        np.arange(1, len(volume_order) + 1),
    )

    volume_order.to_csv(
        volume_order_filename,
        sep="\t",
        index=False,
    )

    LGR.info(f"Saved PALM volume ordering to: {volume_order_filename}")

    return concatenated_filename


def drop_within_subject_constant_regressors(datacontainer, glt_data_table):
    remaining_columns = [
        col
        for col in glt_data_table.columns
        if col not in datacontainer.grouping_columns
    ]
    is_constant = lambda col: glt_data_table.groupby("Subj")[col].nunique().max() <= 1

    constant_columns = []
    for column in remaining_columns:
        if is_constant(column):
            constant_columns.append(column)

    if constant_columns:
        LGR.info(
            f"Removing the following columns that are constant within subjects: {constant_columns}"
        )

        for col in constant_columns:
            if col in glt_data_table.columns:
                glt_data_table = glt_data_table.drop(col, axis=1)

    return glt_data_table, constant_columns


def perform_palm(
    concatenated_filename,
    group_mask_filename,
    design_matrix_file,
    eb_file,
    contrast_matrix_file,
    fsl_img_path,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    n_permutations,
    tfce_H,
    tfce_E,
    tfce_C,
    use_native_palm,
):
    output_prefix = (
        concatenated_filename.parent
        / f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}_desc-nonparametric"
    )

    # The corrcon flag is used to control false positive rates since the positive and negative
    # maps are combined programatically later, which doubles the fpr (i.e., 2 × 0.05 = 0.10)
    # Note that Palm also has a two-tailed; however based on source code:
    #
    # https://github.com/andersonwinkler/PALM/blob/d937704489cc8ae8a7dafb273e421536f8b1e1a5/palm_core.m#L1603-L1605
    # if opts.twotail && ~ opts.missingdata && ( ~ opts.accel.lowrank || p > plm.nJ{m}(c))
    #      G{y}{m}{c} = abs(G{y}{m}{c});
    #   end
    #
    # The abs() is applied before tfce is computed:
    # https://github.com/andersonwinkler/PALM/blob/d937704489cc8ae8a7dafb273e421536f8b1e1a5/palm_core.m#L1674-L1675
    # if opts.tfce.uni.do
    #   tfce{y}{m}{c} = tfcefunc(G{y}{m}{c},y,opts,plm);
    #
    # tfce only integrates over positive thresholds:
    # https://github.com/andersonwinkler/PALM/blob/d937704489cc8ae8a7dafb273e421536f8b1e1a5/palm_tfce.m#L68-L75
    # for h = dh:dh:max(D(:));
    #    CC = bwconncomp(D>=h,opts.tfce.conn);
    #    integ = cellfun(@numel,CC.PixelIdxList).^opts.tfce.E * h^opts.tfce.H;
    #    for c = 1:CC.NumObjects,
    #        tfcestat(CC.PixelIdxList{c}) = ...
    #        tfcestat(CC.PixelIdxList{c}) + integ(c);
    #    end
    # end
    #
    # It appears that with -twotail, abs() makes all voxels positive before tfce, and
    # this would cause opposite effect directions (e.g., A > B and B > A) to be merged
    # into a single cluster which artificially inflates tfce. The intended behavior
    # for a true bidirectional map where each cluster only represents one direction is to do two separate
    # one-tailed contrasts with -corrcon, which corrects fwer across both contrasts, and merge the results
    # into a single bidirectional t-stat map.
    palm_flags = (
        "-noniiclass "
        f"-i {concatenated_filename} "
        f"-m {group_mask_filename} "
        f"-d {design_matrix_file} "
        f"-t {contrast_matrix_file} "
        f"-n {n_permutations} "
        "-T "
        f"-tfce_H {tfce_H} "
        f"-tfce_E {tfce_E} "
        f"-tfce_C {tfce_C} "
        "-corrcon "
        "-logp "
        "-savedof "
        f"-o {output_prefix}"
    )

    if "_vs_" in second_level_glt_code and eb_file:
        palm_flags += f" -eb {eb_file} -ee -within"
    else:
        palm_flags += " -ise"

    if use_native_palm:
        palm_dir = Path(shutil.which("palm")).parent
        cmd = f"matlab -nodisplay -nosplash -r \"addpath('{palm_dir}'); palm {palm_flags}; exit;\""
    else:
        cmd = (
            f"apptainer exec -B /projects:/projects {fsl_img_path} "
            f"octave --eval 'palm {palm_flags}'"
        )

    LGR.info(f"Running PALM for {second_level_glt_code}: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return str(output_prefix), cmd


def perform_3dlmer(
    task,
    entity_key,
    first_level_glt_label,
    dst_dir,
    data_table_filename,
    group_mask_filename,
    afni_img_path,
    model_str,
    center_str,
    glt_str,
    n_cores,
):
    output_filename = (
        dst_dir
        / "second_level"
        / "parametric"
        / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-parametric_stats.nii.gz"
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    if output_filename.exists():
        output_filename.unlink()

    residual_filename = Path(str(output_filename).replace("_stats", "_residuals"))
    if residual_filename.exists():
        residual_filename.unlink()

    residual_log = Path(str(residual_filename).replace(".nii.gz", "_log.txt"))
    if residual_log.exists():
        residual_log.unlink()

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dLMEr "
        f"-mask {group_mask_filename} "
        f"-model '{model_str}' "
        f"-jobs {n_cores} "
        f"{center_str} "
        "-dbgArgs "
        f"{glt_str} "
        f"-prefix {output_filename} "
        f"-resid {residual_filename} "
        f"-dataTable @{data_table_filename}"
    )

    LGR.info(f"Running 3dLMEr: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return residual_filename, cmd


def main(
    bids_dir,
    deriv_dir,
    analysis_dir,
    dst_dir,
    cohort,
    task,
    first_level_glt_label,
    analysis_type,
    space,
    gm_probseg_img_path,
    gm_mask_threshold,
    group_mask_threshold,
    apriori_img_path,
    afni_img_path,
    fsl_img_path,
    method,
    n_permutations,
    tfce_H,
    tfce_E,
    tfce_C,
    nonparametric_cluster_correction_p,
    n_cores,
    exclude_niftis_file,
    categorical_covariates,
    numerical_covariates,
):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir) if deriv_dir else None
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    layout = bids.BIDSLayout(bids_dir, derivatives=deriv_dir or None)

    LGR.info(f"TASK: {task}, METHOD: {method}")

    report_dir = Path(analysis_dir) / "reports" / "second_level"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = HTMLReport(
        subject=None,
        session=None,
        task=task,
        analysis_type=analysis_type,
        method=method,
    )
    report.add_context(
        cohort=cohort,
        space=space,
        group_mask_threshold=group_mask_threshold,
        gm_probseg_img_path=gm_probseg_img_path or None,
        gm_mask_threshold=gm_mask_threshold,
        apriori_img_path=apriori_img_path or None,
    )
    if method == "nonparametric":
        report.add_context(
            tfce_H=tfce_H,
            tfce_E=tfce_E,
            tfce_C=tfce_C,
            nonparametric_cluster_correction_p=nonparametric_cluster_correction_p,
        )

    if first_level_glt_label:
        first_level_glt_labels = (first_level_glt_label,)
    else:
        first_level_glt_labels = get_first_level_gltsym_codes(
            cohort, task, analysis_type, caller="second_level"
        )

    datacontainer = DataContainer(
        categorical_covariates=categorical_covariates,
        numerical_covariates=numerical_covariates,
    )
    LGR.info(
        f"Included covariates: {datacontainer.included_covariates} "
        f"(categorical: {categorical_covariates}, numerical: {numerical_covariates})"
    )
    report.add_context(
        categorical_covariates=datacontainer.categorical_covariates,
        numerical_covariates=datacontainer.numerical_covariates,
    )

    original_categorical_covariates = set(datacontainer.categorical_covariates)
    original_numerical_covariates = set(datacontainer.numerical_covariates)

    for first_level_glt_label in first_level_glt_labels:
        datacontainer.categorical_covariates = list(original_categorical_covariates)
        datacontainer.numerical_covariates = list(original_numerical_covariates)

        entity_key = get_contrast_entity_key(first_level_glt_label)
        LGR.info(f"FIRST LEVEL GLTLABEL: {first_level_glt_label}")

        beta_files = get_beta_files(analysis_dir, task, first_level_glt_label)
        all_subjects = set(get_subjects(beta_files))
        beta_files = exclude_beta_files(
            beta_files,
            exclude_niftis_file,
        )
        if not beta_files:
            LGR.warning(f"No beta files found for {first_level_glt_label}")
            continue

        subject_list = set(get_subjects(beta_files))
        data_table, drop_constant_column_names, important_columns = create_data_table(
            bids_dir, datacontainer, subject_list, beta_files, cohort
        )
        retained_subjects = set(
            [sub.removeprefix("sub-") for sub in data_table["Subj"].unique()]
        )
        retained_beta_files = data_table["InputFile"].tolist()
        LGR.info(
            f"Found {len(retained_beta_files)} files from {len(retained_subjects)} subjects"
        )

        excluded_subjects = sorted(all_subjects - retained_subjects)

        data_table_filename = (
            dst_dir
            / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-data_table.txt"
        )
        data_table.to_csv(data_table_filename, sep="\t", index=False, encoding="utf-8")

        all_data_table_columns = data_table.columns.tolist()

        # Create a filtered version of data table
        data_table = data_table[important_columns]
        data_table_filename = (
            dst_dir
            / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-data_table_filtered_columns.txt"
        )
        data_table.to_csv(data_table_filename, sep="\t", index=False, encoding="utf-8")

        missing_covariates = list(
            set(datacontainer.included_covariates).difference(important_columns)
        )
        if missing_covariates:
            datacontainer.categorical_covariates = list(
                set(datacontainer.categorical_covariates).difference(missing_covariates)
            )
            datacontainer.numerical_covariates = list(
                set(datacontainer.numerical_covariates).difference(missing_covariates)
            )
            LGR.info(
                f"The following covariates were dropped due to being constant or NaN: {missing_covariates}. "
                f"Remaining included covariates: {datacontainer.included_covariates}"
            )

        dose_sample_sizes, subject_dose_sample_sizes = get_sample_sizes(data_table)

        report.add_context(
            first_level_glt_label=f"{first_level_glt_label} ({first_level_glt_label.replace('_vs_', ' > ')})",
            n_beta_files=len(retained_beta_files),
            n_subjects=len(retained_subjects),
            excluded_subjects=excluded_subjects,
            dropped_columns=set(drop_constant_column_names + missing_covariates),
            important_columns=important_columns,
            all_data_table_columns=all_data_table_columns,
            filtered_data_table_columns=important_columns,
            dose_sample_sizes=dose_sample_sizes,
            subject_dose_sample_sizes=subject_dose_sample_sizes,
        )

        if method == "parametric":
            if not afni_img_path:
                LGR.warning("`afni_img_path` is required when method is parametric.")
                sys.exit(1)

            LGR.warning(
                f"Using {len(data_table['InputFile'].tolist())} files "
                f"from {len(data_table['Subj'].unique())} subjects for analysis "
            )
            LGR.info(f"Creating group mask with threshold: {group_mask_threshold}")
            group_mask_filename = create_group_mask(
                dst_dir,
                layout,
                task,
                space,
                group_mask_threshold,
                data_table["InputFile"].tolist(),
                gm_probseg_img_path,
                gm_mask_threshold,
                apriori_img_path,
                method,
                entity_key,
                first_level_glt_label,
            )

            glt_str = get_glt_codes_str(data_table, datacontainer, cohort)
            model_str = get_model_str(datacontainer)
            center_str = get_centering_str(datacontainer)

            residual_filename, lmer_cmd = perform_3dlmer(
                task,
                entity_key,
                first_level_glt_label,
                dst_dir,
                data_table_filename,
                group_mask_filename,
                afni_img_path,
                model_str,
                center_str,
                glt_str,
                n_cores,
            )

            report.add_context(
                model_str=model_str.replace("  ", " "),
                centering_str=center_str.replace("  ", " "),
                glt_str=glt_str.replace("  ", " "),
                lmer_cmd=lmer_cmd.replace("  ", " "),
            )

            acf_parameters_filename = estimate_noise_smoothness(
                dst_dir,
                afni_img_path,
                group_mask_filename,
                residual_filename,
                first_level_glt_label,
            )
            perform_cluster_simulation(
                afni_img_path,
                group_mask_filename,
                acf_parameters_filename,
                first_level_glt_label,
            )

            acf_lines = acf_parameters_filename.read_text().strip().splitlines()
            acf_parameters = acf_lines[1].split()[:4]
            report.add_context(acf_parameters=acf_parameters)
        else:
            # Nonparametric (PALM)
            use_native_palm = shutil.which("palm") is not None
            use_native_fsl = shutil.which("fslmerge") is not None

            if not fsl_img_path and not (use_native_palm or use_native_fsl):
                LGR.warning(
                    "`fsl_img_path` is required when method is nonparametric "
                    "and palm and `fslmerge` are not in path."
                )
                sys.exit(1)

            output_dir = dst_dir / "second_level" / "nonparametric"
            output_dir.mkdir(parents=True, exist_ok=True)
            second_level_contrasts = []
            for second_level_glt_code in get_second_level_glt_codes(cohort):
                LGR.info(f"Processing the following glt code: {second_level_glt_code}")

                glt_data_table, removed_subjects = drop_dose_rows(
                    data_table,
                    get_nontarget_dose(second_level_glt_code, cohort),
                    only_complete_cases=needs_complete_cases(method),
                    return_removed_subjects=True,
                )

                glt_data_table = recode_doses(
                    glt_data_table, cohort, second_level_glt_code
                )

                glt_data_table, constant_columns = drop_constant_columns(glt_data_table)
                glt_data_table = sort_data_table(glt_data_table)
                n_subjects = len(glt_data_table["Subj"].unique())

                if average_within_subjects := needs_within_subject_averaging(
                    cohort, second_level_glt_code
                ):
                    if second_level_glt_code == "mean":
                        # All doses averaged into one image per subject
                        n_files = n_subjects
                    else:
                        n_dose_levels = glt_data_table["dose"].nunique()
                        n_files = n_subjects * n_dose_levels
                else:
                    n_files = len(glt_data_table["InputFile"].tolist())

                LGR.warning(
                    f"Using {n_files} files "
                    f"from {n_subjects} subjects for analysis "
                    f"using {second_level_glt_code}"
                )

                LGR.info(f"Creating group mask with threshold: {group_mask_threshold}")
                group_mask_filename = create_group_mask(
                    dst_dir,
                    layout,
                    task,
                    space,
                    group_mask_threshold,
                    glt_data_table["InputFile"].tolist(),
                    gm_probseg_img_path,
                    gm_mask_threshold,
                    apriori_img_path,
                    method,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                )

                vs_in_code = "_vs_" in second_level_glt_code
                matrix_creation_func = (
                    create_comparison_matrices if vs_in_code else create_mean_matrices
                )
                matrices_output_dict, matrix_report_info = matrix_creation_func(
                    glt_data_table,
                    datacontainer,
                    output_dir,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                    average_within_subjects,
                )
                max_permutations, true_max_permutations = (
                    compute_n_permutation(glt_data_table)
                    if n_permutations == "auto"
                    else (n_permutations, n_permutations)
                )

                concatenated_filename = create_concatenated_image(
                    output_dir,
                    glt_data_table,
                    fsl_img_path,
                    use_native_fsl,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                )

                output_prefix, palm_cmd = perform_palm(
                    concatenated_filename,
                    group_mask_filename,
                    matrices_output_dict["design_matrix_file"],
                    matrices_output_dict["eb_file"],
                    matrices_output_dict["contrast_matrix_file"],
                    fsl_img_path,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                    max_permutations,
                    tfce_H,
                    tfce_E,
                    tfce_C,
                    use_native_palm,
                )
                for img_path in Path(output_prefix).parent.glob(
                    f"*{first_level_glt_label}*.nii"
                ):
                    if img_path.exists():
                        compress_image(
                            img_path,
                            dst_dir=Path(output_prefix).parent,
                            remove_src_file=True,
                        )

                threshold_palm_output(
                    output_prefix,
                    second_level_glt_code,
                    nonparametric_cluster_correction_p,
                )

                header_names_filename = str(
                    matrices_output_dict["design_matrix_file"]
                ).replace("-design_matrix.csv", "-header_names.txt")
                with open(header_names_filename, "r") as f:
                    design_columns = f.read().split(",")

                matrix_report_info["constant_column_names"] = list(
                    set(matrix_report_info["constant_column_names"] + constant_columns)
                )
                contrast_info = {
                    "glt_code": second_level_glt_code,
                    "n_files": n_files,
                    "n_subjects": n_subjects,
                    "dose_list": glt_data_table["dose"].unique().tolist(),
                    "max_permutations": max_permutations,
                    "true_max_permutations": true_max_permutations,
                    "is_comparison": vs_in_code,
                    "design_columns": design_columns,
                    "removed_subjects": removed_subjects,
                    "dropped_within_subject_columns": list(
                        matrix_report_info.get("constant_column_names", [])
                    ),
                    "dropped_collinear_columns": list(
                        matrix_report_info.get("collinear_column_names", [])
                    ),
                    "residual_dof": matrix_report_info.get("residual_dof"),
                    "palm_cmd": palm_cmd.replace("  ", " "),
                    "averaged_within_subjects": average_within_subjects,
                }
                if vs_in_code:
                    labels = get_group_labels(second_level_glt_code)
                    contrast_info["positive_label"] = f"{labels[0]} > {labels[1]}"
                    contrast_info["negative_label"] = f"{labels[1]} > {labels[0]}"
                else:
                    contrast_info["positive_label"] = "above mean of 0"
                    contrast_info["negative_label"] = "below mean of 0"

                second_level_contrasts.append(contrast_info)

            report.add_context(second_level_contrasts=second_level_contrasts)

        report_path = (
            report_dir
            / f"task-{task}_contrast-{first_level_glt_label}_desc-{method}_{analysis_type}_report.html"
        )
        report.create_report(report_path, "second_level.html")


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
