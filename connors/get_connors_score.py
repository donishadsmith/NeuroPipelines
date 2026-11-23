"""Extract data from Conners 4."""

import re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from typing import Literal
from datetime import datetime

import pandas as pd, numpy as np
from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger

from _general_utils import guess_delimiter

LGR = setup_logger(__name__)

CSV_COLUMN_NAMES = [
    "Subject",
    "Session",
    "Assessment Date",
    "Rater",
    "INA/EDF T score",
    "INA/EDF %",
    "HYP T score",
    "HYP %",
    "IMP T score",
    "IMP %",
    "EMDYS T score",
    "EMDYS %",
    "DEP T score",
    "DEP %",
    "ANX T score",
    "ANX %",
    "SCHOOL T score",
    "SCHOOL %",
    "PEER T score",
    "PEER %",
    "FAMILY T score",
    "FAMILY %",
    "ADHD-I T score",
    "ADHD-I %",
    "ADHD-HI T score",
    "ADHD-HI %",
    "ADHD-TOT T score",
    "ADHD-TOT %",
    "ODD T score",
    "ODD %",
    "CD T score",
    "CD %",
    "Prob score %",
]
UNIQUE_DATA_FIELD_NAMES = [
    "Inattention/Executive Dysfunction",
    "Hyperactivity",
    "Impulsivity",
    "Emotional Dysregulation",
    "Depressed Mood",
    "Anxious Thoughts",
    "Schoolwork",
    "Peer Interactions",
    "Family Life",
    "ADHD Inattentive Symptoms",
    "ADHD Hyperactive/Impulsive Symptoms",
    "Total ADHD Symptoms",
    "Oppositional Defiant Disorder Symptoms",
    "Conduct Disorder Symptoms",
    "ADHD Index",
]


def get_files(target_dir: Path, pattern: str) -> list[str]:
    """Gets files with a specific extension."""
    return list(target_dir.glob(pattern))


def get_non_session_column_index() -> int:
    for index, element in enumerate(CSV_COLUMN_NAMES):
        if any(x in element for x in ["T score", "%"]):
            return index


def initialization_sessions_dict() -> dict[str, None]:
    return {key: None for key in CSV_COLUMN_NAMES[: get_non_session_column_index()]}


def extract_pdf_text(
    pdf_file: str, page_number: int, use_pdfplumber: bool = False
) -> list[str] | list[list[str]]:
    """
    Extract text from a single page of a PDF and removes lines with only whitespace.
    when using pypdf or a list of list for pdfplumber
    """
    import pdfplumber
    from pypdf import PdfReader

    if use_pdfplumber:
        with pdfplumber.open(pdf_file) as pdf:
            page = pdf.pages[page_number]
            table = page.extract_table()

        return table
    else:
        reader = PdfReader(pdf_file)
        single_page = reader.pages[page_number].extract_text()
        stripped_page_list = [
            line.strip(" ")
            for line in [
                line for line in single_page.splitlines() if not line.isspace()
            ]
        ]
        return stripped_page_list


def determine_rater(single_page_list: list[str]) -> Literal["parent", "child"]:
    """
    Determines if the rater is a child or parent.

    Uses the first page (index 0) of the Conners 4 PDF
    to determine if the rater is a child or parent.
    For Conners 4, if the rater is a parent, there
    is a "Parent's/Guardian's" field that is not
    present when the child is the rater.

    Parameters
    ----------
    single_page_list: :obj:`list[str]`
        A single PDF page represented as a list where each element
        is a line of of text as a string.

    Returns
    -------
    Literal["parent", "child"]:
        A string denoting if the rater is a parent or child
    """
    has_parent_guardian_str = any(
        [line.startswith("Parent") for line in single_page_list]
    )

    return "parent" if has_parent_guardian_str else "child"


def standardize_date(date: str, date_format: str = "%B %d, %Y") -> str:
    "Standardizes the administration date."
    return datetime.strptime(date, date_format).date()


def standardize_pdf_filenames(pdf_dir: Path) -> None:
    """
    Cleans PDF filenames.

    Standardizes the PDF filename from a random assortment of letters to
    sub-{sub_id}_date-{administation_date}_rater-{file_id}.
    Outputs these filenames in a subdirectory named "reformatted_filenames"
    under the directory specified by ``pdf_dir``. Dynamically creates additional
    "child" and "parent" subdirectories under "reformatted_filenames" if any PDF is
    detected as the rater being the child or parent.

    Parameters
    ----------
    pdf_dir: :obj:`str`
        Directory containing the unformatted Conners 4 PDF files

    Returns
    -------
    None
    """
    target_dir = pdf_dir / "reformatted_filenames"
    target_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = get_files(pdf_dir, "*.pdf")
    for pdf_file in pdf_files:
        stripped_page_list = extract_pdf_text(pdf_file, 0)
        rater = determine_rater(stripped_page_list)

        sub_id = [line for line in stripped_page_list if line.startswith("Name/ID")][0]
        sub_id = sub_id.split(" ")[1]

        administration_date = [
            line
            for line in stripped_page_list
            if line.startswith("Administration Date")
        ]
        administration_date = (
            administration_date[0].removeprefix("Administration Date:").strip(" ")
        )
        administration_date = standardize_date(administration_date)

        new_filename = f"sub-{sub_id}_date-{administration_date}_rater-{rater}"
        output_dir = target_dir / rater
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file: Path = output_dir / pdf_file.name
        shutil.copyfile(pdf_file, output_file)

        new_filename = output_file.with_name(f"{new_filename}.pdf")
        if new_filename.exists():
            new_filename.unlink()

        output_file.rename(new_filename)


def get_subject_ids(reformatted_pdf_files: list[str]) -> list[str]:
    """
    Gets subject IDs from the reformatted pdf files.

    Parameters
    ----------
    reformatted_pdf_files: :obj:`list[str]`
        Absolute path of PDF files with reformatted names in the
        form "sub-[subject_id]_date-[administration_date]_rater-[rater].pdf"

    Returns
    -------
        list[str]
            List of subject IDs.
    """
    all_subjects_list = [
        (get_entity_value(file.name, "sub"), get_entity_value(file.name, "rater"))
        for file in reformatted_pdf_files
    ]

    return zip(*all_subjects_list)


def get_sessions(subject_id_list: list[str]) -> list[str]:
    """
    Returns a list of the sessions in the form "v1", "v2",
    "v3", etc for each subject id.
    """
    session_list = []
    unique_ids = list(dict.fromkeys(subject_id_list).keys())
    for unique_id in unique_ids:
        visit_num_list = [val + 1 for val in range(subject_id_list.count(unique_id))]
        session_list.extend([f"ses-0{val}" for val in visit_num_list])

    return session_list


def get_sn_visit(subject_id_list: list[str], visit_list: list[str]) -> list[str]:
    """
    Gets the subject id and visit. Assumes ``subject_id_list`` and
    ``visit_list`` are the same length.
    """
    sn_visit_list = []
    subject_session_list = list(zip(subject_id_list, visit_list))
    for subject, session in subject_session_list:
        sn_visit_list.append(f"{subject}{session}")

    return sn_visit_list


def get_assessment_dates(reformatted_pdf_files: list[str]) -> list[str]:
    return [
        pd.to_datetime(str(file).split("date-")[-1].split("_rater")[0])
        for file in reformatted_pdf_files
    ]


def sort_files_and_dates(reformatted_pdf_files: list[str], assessment_dates: list[str]):
    sorted_tuples = sorted(
        list(zip(assessment_dates, reformatted_pdf_files)),
        key=lambda x: pd.to_datetime(x[0]),
    )
    assessment_dates, sorted_reformatted_pdf_files = zip(*sorted_tuples)

    return list(assessment_dates), list(sorted_reformatted_pdf_files)


def create_unique_column_dict() -> dict[str, None]:
    """
    Converts a list of column names to dictionary containing the unique
    starting names. For instance, "ANX T score" and "ANX %" are reduced
    to "ANX". This is done to later map these unique name to the
    Conners data field that contains both of these values
    (when converted to text, each line of Conners data field contains raw score,
    t score, percentile, etc).
    """
    unique_column_names_dict = {}
    for name in CSV_COLUMN_NAMES[get_non_session_column_index() :]:
        reduced_name = name.split(" ")[0]
        if reduced_name not in unique_column_names_dict:
            unique_column_names_dict[reduced_name] = None

    return unique_column_names_dict


def create_column_names_dict() -> dict[str, list]:
    """
    Converts a list of column names to dictionary of each name mapped to
    an empty list.
    """
    column_names_dict = {}
    for name in CSV_COLUMN_NAMES[get_non_session_column_index() :]:
        column_names_dict[name] = []

    return column_names_dict


def clean_lines(table: list[list[str]]) -> list[list[str]]:
    "Replace newlines with space and remove asterisks from words."
    new_table = []
    for line in table:
        line = [
            word.replace("\n", " ").strip("*").strip("^") if word else word
            for word in line
        ]
        new_table.append(line)

    return new_table


def create_table_column_map(table: list[list[str]]) -> dict[str, int]:
    for index, line in enumerate(table):
        # Should be the second list in table but searching for safety
        for word in line:
            match = re.search("Raw Score", word) if word else None
            if match:
                break
        else:
            continue

        break

    # Index position -> column name
    column_names = dict(enumerate(table[index]))
    filtered_column_names = dict()
    for key in column_names:
        if column_names[key]:
            # Column name -> index position
            filtered_column_names.update({column_names[key]: key})

    return filtered_column_names


def get_target_list(table: list[list[str]], target_name: str) -> list[str, None]:
    for line in table:
        for word in line:
            if word and target_name == word:
                return line


def get_score(
    table_column_map: dict[str, int], line: list[str] | None, score_type: str
) -> str:
    connors_column_field_map = {"T score": "T-score", "%": "90% CI"}
    connors_field_name = connors_column_field_map[score_type]
    connors_field_index = table_column_map[connors_field_name]

    return line[connors_field_index]


def extract_conners_datafields(
    reformatted_pdf_files: str, conners_data_fields_dict: dict[str, str]
) -> dict[str, list[float]]:
    """Extracts the ADHD data from the pdf files."""
    column_names_dict = create_column_names_dict()

    for pdf_file in reformatted_pdf_files:
        table = clean_lines(
            extract_pdf_text(pdf_file=pdf_file, page_number=3, use_pdfplumber=True)
        )
        table_column_map = create_table_column_map(table)

        for key in conners_data_fields_dict.keys():
            data_field_name = conners_data_fields_dict[key]
            if key != "Prob":
                for score_type in ["T score", "%"]:
                    line = get_target_list(table, data_field_name)
                    column_names_dict[f"{key} {score_type}"].append(
                        get_score(
                            table_column_map,
                            line,
                            score_type,
                        )
                    )
            else:
                adhd_index_list = [
                    x for x in get_target_list(table, "ADHD Index") if x and "%" in x
                ]

                if not adhd_index_list:
                    adhd_index_list = [
                        x
                        for x in get_target_list(table, "ADHD Index")
                        if x and "?" in x
                    ]

                column_names_dict["Prob score %"].append(adhd_index_list[0])

    return column_names_dict


def clean_df(df: pd.DataFrame, include_assessment_dates: bool) -> pd.DataFrame:
    date_in_df = "Assessment Date" in df.columns
    if date_in_df:
        if df["Assessment Date"].isna().any() or not include_assessment_dates:
            df = df.drop(columns=["Assessment Date"], axis=1)

    return df


def run_pipeline(
    pdf_dir: Path, csv_file_path: str, include_assessment_dates: bool
) -> None:
    """Main function to reformat filenames and extract Conners data."""
    pdf_dir = Path(pdf_dir)

    LGR.info("Standardizing PDF filenames...")
    standardize_pdf_filenames(pdf_dir)

    reformatted_pdf_files = get_files(
        pdf_dir / "reformatted_filenames" / "child", "*sub-*.pdf"
    )
    reformatted_pdf_files += get_files(
        pdf_dir / "reformatted_filenames" / "parent", "*sub-*.pdf"
    )

    if not reformatted_pdf_files:
        return None

    data_fields_dict = initialization_sessions_dict()
    assessment_dates = get_assessment_dates(reformatted_pdf_files)
    data_fields_dict["Assessment Date"], reformatted_pdf_files = sort_files_and_dates(
        reformatted_pdf_files, assessment_dates
    )

    data_fields_dict["Subject"], data_fields_dict["Rater"] = get_subject_ids(
        reformatted_pdf_files
    )
    data_fields_dict["Session"] = get_sessions(data_fields_dict["Subject"])

    conners_data_fields_dict = dict(
        zip(create_unique_column_dict(), UNIQUE_DATA_FIELD_NAMES)
    )

    LGR.info("Extracting Connors data...")
    extracted_conners_data_dict = data_fields_dict | extract_conners_datafields(
        reformatted_pdf_files, conners_data_fields_dict
    )

    df = pd.DataFrame(extracted_conners_data_dict)
    df["Subject"] = df["Subject"].astype(str)
    filename = (
        pdf_dir / "reformatted_filenames" / "conners_data.csv"
        if csv_file_path is None
        else csv_file_path
    )
    if csv_file_path and Path(csv_file_path).exists():
        LGR.info(f"Appending new data to {Path(filename).name}...")
        if csv_file_path.endswith(".xlsx") or csv_file_path.endswith(".xls"):
            original_df = pd.read_excel(filename)
        else:
            sep = guess_delimiter(filename)
            original_df = pd.read_csv(
                filename, sep=None, engine="python", encoding="utf-8-sig", iterator=True
            )

        original_df.columns = [col.replace("\ufeff", "") for col in original_df.columns]
        original_df["Subject"] = original_df["Subject"].astype(str)

        df = pd.concat([original_df, df], axis=0, ignore_index=True)
        df = df.drop_duplicates()
        df = df.replace("?", np.nan)

        df = clean_df(df, include_assessment_dates)
        if csv_file_path.endswith(".xlsx"):
            df.to_excel(filename)
        else:
            # Fix for dash encoding issue
            df.to_csv(filename, sep=sep, encoding="utf-8-sig", index=False)
    else:
        df = clean_df(df, include_assessment_dates)
        LGR.info(f"Creating {Path(filename).name}...")
        # Fix for dash encoding issue
        df.to_csv(filename, sep=",", encoding="utf-8-sig", index=False)

    return filename
