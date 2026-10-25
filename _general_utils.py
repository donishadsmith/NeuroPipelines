import tempfile, re
from pathlib import Path
from typing import Optional

import nibabel as nib, numpy as np, pandas as pd
from nilearn.masking import _unmask_3d
from nilearn.maskers import nifti_spheres_masker

from bidsaid.path_utils import is_valid_date
from bidsaid.logging import setup_logger
from bidsaid.parsers import _is_float

LGR = setup_logger(__name__)

SEPARATORS = (",", "\t", ";", "|", " ")

COHORT_MAP = {
    "kids": {
        "template_mask_path": Path(__file__).parent
        / "templates/kids/tpl-MNIPediatricAsym_cohort-1_res-2_desc-brain_mask.nii.gz",
        "template_img_path": Path(__file__).parent
        / "templates/kids/tpl-MNIPediatricAsym_cohort-1_res-1_desc-brain_T1w.nii.gz",
    },
    "adults": {
        "template_mask_path": Path(__file__).parent
        / "templates/adults/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
        "template_img_path": Path(__file__).parent
        / "templates/adults/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz",
    },
}


def _convert_to_bool(arg: bool | str) -> bool:
    if str(arg).lower() == "true":
        return True
    elif str(arg).lower() == "false":
        return False
    else:
        raise ValueError("For booleans, only 'True' and 'False' are valid.")


def _resolve_directories(bids_dir, temp_dir, caller):
    base_folder = {
        "BIDS Dataset": "BIDS_Dataset",
        "BIDS Events": "BIDS_Events",
        "Behavioral Data": "Behavioral_Data",
    }
    if not bids_dir:
        bids_dir = Path().home() / base_folder[caller]
    else:
        bids_dir = Path(bids_dir)

    bids_dir.mkdir(parents=True, exist_ok=True)

    use_tempfile = bool(temp_dir)
    temp_dir = temp_dir or tempfile.TemporaryDirectory().name
    temp_dir = Path(temp_dir)
    if temp_dir.exists() and not use_tempfile:
        raise FileExistsError(
            "The temporary directory exists; either choose another name or "
            f"delete the following directory: {temp_dir}"
        )

    temp_dir.mkdir(parents=True, exist_ok=True)

    LGR.info(f"Output Directory: {bids_dir}; Temporary Directory: {temp_dir}")

    return bids_dir, temp_dir


def _extract_subjects_visits_data(
    participant_id: str,
    subjects_visits_df: pd.DataFrame,
    column_name: int,
    scan_date: Optional[str] = None,
    date_column_name: str = "date",
):
    mask = subjects_visits_df["participant_id"] == str(participant_id)
    if scan_date:
        mask &= subjects_visits_df[date_column_name] == str(scan_date)

    subjects_visits_df[mask].loc[:, column_name].astype(str).tolist()

    return subjects_visits_df[mask].loc[:, column_name].astype(str).tolist()


def _get_subject_visits(
    participant_id: str,
    subjects_visits_df: pd.DataFrame,
) -> dict[str, str]:
    visit_dates = _extract_subjects_visits_data(
        participant_id,
        subjects_visits_df,
        column_name="date",
    )
    if not visit_dates:
        LGR.warning(f"Subject {participant_id} has no visit dates.")
        return None

    return {
        f"0{session_id}": date for session_id, date in enumerate(visit_dates, start=1)
    }


def _standardize_dates(
    df: pd.DataFrame,
    date_column_name="date",
    sort_data=True,
    participant_column_name="participant_id",
) -> list[str]:
    df.columns = [col.strip() for col in df.columns]

    if participant_column_name in df.columns:
        df[[participant_column_name, date_column_name]] = df[
            [participant_column_name, date_column_name]
        ].astype(str)
    else:
        df[date_column_name] = df[date_column_name].astype(str)

    null_dates = df[date_column_name].isna()
    if any(null_dates.tolist()):
        LGR.info("Dropping Null/NaN dates")

    df = df[~null_dates]
    df[date_column_name] = [
        (
            str(pd.to_datetime([date_str], format=r"%y%m%d")[0]).split()[0]
            if is_valid_date(date_str, r"%y%m%d")
            else str(pd.to_datetime([date_str], errors="coerce")[0]).split()[0]
        )
        for date_str in df[date_column_name]
    ]

    invalid_dates = df[date_column_name] == "NaT"
    if any(invalid_dates.tolist()):
        LGR.info(
            f"Dropping the following invalid dates: {df.loc[invalid_dates, date_column_name]}"
        )

    df = df[~invalid_dates]

    if sort_data:
        if participant_column_name in df.columns:
            df = df.sort_values(
                by=[participant_column_name, date_column_name],
                ascending=True,
                key=lambda col: (
                    pd.to_datetime(col) if col.name == date_column_name else col
                ),
            )
        else:
            df = df.sort_values(
                by=[date_column_name],
                ascending=True,
                key=lambda col: pd.to_datetime(col),
            )

    return df


def _standardize_participant_ids(
    df: pd.DataFrame, participant_id_column="participant_id"
):
    df[participant_id_column] = (
        df[participant_id_column]
        .apply(lambda x: str(x))
        .apply(lambda x: re.findall(r"\d{5}", x)[0])
    )

    return df


class SubjectsVisitsFileError(Exception):
    """Exception for issues with the subjects sessions file."""

    pass


def _check_subjects_visits_file(
    subjects_visits_file: str | Path,
    dose_column_required: bool,
    return_df: bool = False,
    for_app: bool = False,
    return_boolean: bool = False,
) -> None | bool | pd.DataFrame:
    required_colnames = ["participant_id", "date"]

    if str(subjects_visits_file).endswith(".xlsx") or str(
        subjects_visits_file
    ).endswith(".xls"):
        subjects_visits_df = pd.read_excel(subjects_visits_file)
    else:
        subjects_visits_df = pd.read_csv(
            subjects_visits_file, sep=None, engine="python"
        )

    subjects_visits_df.columns = [col.strip() for col in subjects_visits_df.columns]

    if not all(
        required_colname in subjects_visits_df.columns.tolist()
        for required_colname in required_colnames
    ):
        if not return_boolean:
            raise SubjectsVisitsFileError(
                f"The following columns are required in {subjects_visits_file}: "
                f"{required_colnames}."
            )
        valid_file = False
    else:
        valid_file = True

    if dose_column_required:
        if "dose" not in subjects_visits_df.columns:
            if not return_boolean:
                raise SubjectsVisitsFileError(
                    f"A 'dose' column is required in {subjects_visits_file}"
                )
            valid_file = False
        else:
            valid_file = True

    if len(subjects_visits_df.columns) > 2 and "dose" not in subjects_visits_df.columns:
        LGR.warning(
            f"More than two columns detected in {subjects_visits_file} but 'dose' "
            "column is missing. To include 'dose' values in the session TSV files, this column "
            "must be included."
        )

    if for_app:
        return valid_file

    subjects_visits_df["participant_id"] = subjects_visits_df["participant_id"].astype(
        str
    )

    return subjects_visits_df if return_df else None


def _get_dataframe(subjects_visits_file: str | Path) -> pd.DataFrame | None:
    if not subjects_visits_file:
        return None

    if str(subjects_visits_file).endswith(".xlsx") or str(
        subjects_visits_file
    ).endswith(".xls"):
        return pd.read_excel(subjects_visits_file)

    with open(subjects_visits_file, "r", encoding="utf-8-sig") as f:
        header = f.readline()

    if not any(sep in header for sep in SEPARATORS):
        return pd.read_csv(subjects_visits_file, encoding="utf-8-sig")

    return pd.read_csv(
        subjects_visits_file, sep=None, engine="python", encoding="utf-8-sig"
    )


def _check_coordinate(coordinate):
    if not all([_is_float(x) for x in coordinate]):
        coordinate = list(map(str, coordinate))
        raise ValueError(f"Invalid MNI coordinate: {','.join(coordinate)}")


def get_template_imgs(cohort):
    return (
        COHORT_MAP[cohort]["template_mask_path"],
        COHORT_MAP[cohort]["template_img_path"],
    )


def _create_sphere_mask(coordinate, sphere_radius, template_mask):
    # https://neurostars.org/t/create-a-10mm-sphere-roi-mask-around-a-given-coordinate/28853/3
    _, A = nifti_spheres_masker.apply_mask_and_get_affinity(
        seeds=[tuple(coordinate)],
        niimg=None,
        radius=sphere_radius,
        allow_overlap=False,
        mask_img=template_mask,
    )

    sphere_mask = _unmask_3d(
        X=A.toarray().flatten(), mask=template_mask.get_fdata().astype(bool)
    ).astype(np.int8)

    hdr = template_mask.header.copy()
    hdr.set_data_dtype(np.int8)

    return nib.nifti1.Nifti1Image(sphere_mask, template_mask.affine, hdr)
