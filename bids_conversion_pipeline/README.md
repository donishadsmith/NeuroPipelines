# NIfTI to BIDS

Pipeline for converting non-BIDS data to BIDS format. Assumes source data has folder names in the form `{subjectID}_{scan_date}` (also compatible with `{subjectID}_{other_content}_{scan_date}` as long as the subject ID is first and the scan date is last). Also assumes one run per task. Tailored for specific tasks and scan protocols.

**Notes:**
- Ensure scan dates are standardized across all source folders.
- If converting on Windows, you may need to convert line endings from [DOS to UNIX](https://blog.bachi.net/?p=1715): `sed -i "s/\r//" filename`

**What the pipeline does:**
1. Collects `.nii` and `.nii.gz` files and compresses any uncompressed NIfTI images.
2. Identifies each NIfTI image by checking dimensionality (3D vs 4D) and matching volume counts to known tasks.
3. Converts data to BIDS format and creates minimal JSON sidecars.
4. Optionally creates the participants TSV, dataset description JSON, and sessions TSV files. If a participants TSV already exists, new subjects are appended.

**Execution order:**
1. `bids_conversion_cli.py`: main conversion
2. `participants_tsv_cli.py`: create or update participants TSV
3. `participants_demographics_cli.py`: add or update participants TSV with demographics data
4. `add_dosages_cli.py`: add dosages to sessions TSV

## Command Line Interface (CLI)

### Basic Usage

```bash
python cli/bids_conversion_cli.py \
    --src_dir /path/to/src/dir \
    --bids_dir /path/to/bids/dir \
    --subjects_visits_file /path/to/visits.csv \
    --create_dataset_metadata True \
    --add_sessions_tsv True
```

### Restricting to Specific Subjects

```bash
python cli/bids_conversion_cli.py \
    --src_dir /path/to/src/dir \
    --bids_dir /path/to/bids/dir \
    --subjects_visits_file /path/to/visits.csv \
    --create_dataset_metadata True \
    --add_sessions_tsv True \
    --subjects 100 101
```

### Full Help

```bash
python cli/bids_conversion_cli.py --help
```

### Running on HPC with SLURM

```bash
sbatch --array=0-1 main.sb 100 101
sbatch participants_tsv.sb
```

## Subjects Visits File
Session mapping is based exclusively on the subject ID and dates and available in the subjects visits file. The dates are standardized, then sorted, and the session IDs
are based on the sorting order. Consequently, if a date is present in the subjects visits file but not found in the subject-specific source directories,
then the date will be ignored but the session labeling will not change.
```python
date_map = {"01": "2000-01-01", "02": "2000-02-02", "03": "2000-03-03"}
# 2000-02-02 is missing in source files
new_date_map = {"01": "2000-01-01", "03": "2000-03-03"}
```
Conversely, if a date is found in the subject-specific source directories but is not in the subjects visits file,
then the date will be used as its session label.
```python
date_map = {"01": "2000-01-01", "02": "2000-03-03"}
# 2000-02-02 found in subject-specific source directories (assuming the directory is not excluded)
new_date_map = {"01": "2000-01-01", "02": "2000-03-03", "2000-02-02": "2000-02-02"}
```
The file subject visits file must contain `participant_id` and `date` columns. Use `NaN` for missing sessions or
exclude rows with no dates or unwanted dates.

| participant_id | date       |
|----------------|------------|
| 101            | 01/02/2022 |
| 101            | NaN        |
| 101            | 03/02/2022 |
| 102            | 01/02/2024 |

To include dosages, add a `dose` column:

| participant_id | date       | dose |
|----------------|------------|------|
| 101            | 01/02/2000 | 0    |
| 101            | 03/02/2000 | 10   |
| 102            | 01/02/2002 | NaN  |

## Streamlit Graphical User Interface (GUI)

For a point-and-click interface, launch the Streamlit app:

```bash
streamlit run bids_app.py
```

The app provides pages for each pipeline step: BIDS Conversion, Participants TSV, and Add Dosages.
