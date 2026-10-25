"""
gPPI workflow

Papers:
1) https://pmc.ncbi.nlm.nih.gov/articles/PMC4632075/
2) https://pmc.ncbi.nlm.nih.gov/articles/PMC3376181/

Forums:
1) https://discuss.afni.nimh.nih.gov/t/gppi-analysis-and-upsampling/172
2) https://web.archive.org/web/20241103095511/https://afni.nimh.nih.gov/CD-CorrAna
   (archived website)
3) https://discuss.afni.nimh.nih.gov/t/how-to-do-gppi-to-event-related-fmri/457/4

1) Collect confounds and create censor file
2) PSC scaling of NIfTI image, compute mean for censored files
3) Resample mask to NIfTI (if needed) then extract timeseries
4) Tranpose the seed timeseries to a column vector
5) Denoise seed timeseries. Do same denoising with seed and image. Ensure regular OLS is used just
to orthogonalize to the nuisance regressors for the seed, no prewhitening should be done to prevent
temporal autocorrelation in the residuals. Note that smoothing is not done prior
to extracting the seed, the timeseries is already averaged which helps with spatial noise
reduction. More importantly, smoothing blur signal outside of the voxels of interest into
the will result in your seed timeseries containing signal from voxels outside of your mask

For each condition in task (6-10):
6) Upsample seed timeseries (https://www.nature.com/articles/s42003-024-07088-3) and task regressor to 0.1
   (TR_orig/ TR_sub is equal to number of points added between each TR or
   the duration / TR_sub is equal to the number of points added after each onset
   time). Improves the convolution procedure.
   Resources:
        1) https://discuss.afni.nimh.nih.gov/t/gppi-analysis-and-upsampling/172
        2) https://www.nature.com/articles/s42003-024-07088-3
7) The task regressor should then be mean centered so that the subsequent interaction term
   is not highly correlated with the main effect of the seed timeseries and result in
   spurious results that attribute correlation with the seed timeseries to the interaction
   term. Great paper about this:
   https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.989/133601/Common-pitfalls-during-model-specification-in
8) Deconvolve seed timeseries to get the neural signal that will later
   interact with the task regressor and this interaction will be convolved. The seed timeseries
   is not directly multiplied by the convolved task timing because the seed timeseries itself (assuming it
   is affected by the task) is already the observed product of the neural signal and task, so
   direct interaction results in (neural signal of seed * hrf) x (task * hrf) != (neural signal of seed x Task) * hrf.
   Though finding the best estimate of the neural signal based on observed BOLD = Ideal Hrf * neural signal of seed
   is a noisy, imperfect operation which is compounded by the fact that the observed BOLD timeseries is itself
   the true BOLD + noise, which denoising can help but never fully fix.
9) Create PPI term PPI = ([neural signal * binary_condition_vector] * hrf)(t).
   Use GAM for event-related tasks, and a simulated BLOCK function for block-design tasks.
   Basically use same hrf funcion used for task in GLM.
10) Downsample the PPI term back down to the true TR grid

After:
11) For NIfTI image, smooth, then use 3ddeconvolve. Ensure to model everything
   from nuisance regressors, all main effect conditions (convolved), the
   denoised seed signal, and the PPI interaction terms (already convolved
   in previous step). Create contrasts of the interaction terms (+ means
   greater connectivity for A than B and - means reduced connectivity for
   A relative to B)
12) Use 3dremlfit to account for temporal autocorrelation
13) Extract PPI interaction contrasts betas for downstream analyses

# Interpretation:
- Positive beta coefficients for the PPI regressor means greater connectivity between
  the seed region and the brain region during a specific condition
- Negative beta coefficients for the PPI regressor means reduced connectivity between
  the seed region and the brain region during a specific condition


Denoising papers:
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC7978116/

- Aggressive denoising strategies can also remove task-signal, strategy should depend on
clinical population, type of analysis being done (activation vs connectivity, where in some cases
connectivity may require more aggressive denoising to ensure that correlation is not due to
shared nuisance variance), characteristics of data (numerous high-motion participants or
mostly low-motion participants), and whether strategies such as strict scrubbing (FD < 0.2) will
remove a significant amount of frames resulting in either suboptimal estimated beta coefficients
or too little retainerd participants. There is no optimal denoising strategy for all datasets.
"""

import argparse, subprocess
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd

from bidsaid.logging import setup_logger
from bidsaid.metadata import get_tr, get_n_volumes
from bidsaid.cli import none_or, n_dummy_type, str_to_bool

from _denoising import (
    get_cosine_regressors,
    get_motion_regressors,
    percent_signal_change,
    perform_spatial_smoothing,
)
from _gen_afni_files import (
    create_censor_file,
    create_fd_file,
    create_binary_condition,
    create_timing_files,
    create_nuisance_regressor_file,
)
from _first_level_utils import (
    CONDITION_DURATIONS,
    EVENT_RELATED_TASKS,
    InvalidRunError,
    check_censoring,
    collect_acompcor_names,
    collect_session_files,
    create_diagnostic_condition_plots,
    filter_regressor_names,
    get_sphere_radius,
    is_timing_file_empty,
    plot_signal,
    seed_mask_name_check,
    summarize_timing_conditions,
    validate_first_level_inputs,
)
from _models import (
    create_design_matrix,
    get_task_deconvolve_adults_cmd,
    get_task_deconvolve_kids_cmd,
    perform_first_level,
)
from _utils import (
    create_beta_files,
    embed_img,
    get_beta_names,
    get_coordinate_from_filename,
    get_first_level_gltsym_codes,
    resample_seed_img,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Perform first level gPPI (task-based functional connectivty) for a task."
    )
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="Path to BIDS directory."
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory.",
    )
    parser.add_argument(
        "--deriv_dir",
        dest="deriv_dir",
        required=False,
        default=None,
        help="Root of the derivatives directory.",
    )
    parser.add_argument(
        "--seed_mask_path",
        dest="seed_mask_path",
        required=True,
        help="The mask of the seed region.",
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
        "--subject",
        dest="subject",
        required=True,
        help="Subject ID without the 'sub-' entity.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--filter_correct_trials",
        dest="filter_correct_trials",
        required=False,
        default=False,
        type=str_to_bool,
        help="Filter correct trials for event-related tasks.",
    )
    parser.add_argument(
        "--n_motion_parameters",
        dest="n_motion_parameters",
        default=6,
        type=int,
        choices=[6, 12, 18, 24],
        required=False,
        help=(
            "Number of motion parameters to use: 6 (base trans + rot), "
            "12 (base + derivatives), 18 (base + derivatives + power), "
            "24 (base + derivatives + power + derivative power). "
        ),
    )
    parser.add_argument(
        "--fd_threshold",
        dest="fd_threshold",
        default=0.5,
        type=float,
        required=False,
        help="Framewise displacement threshold.",
    )
    parser.add_argument(
        "--exclusion_criteria",
        dest="exclusion_criteria",
        default=0.20,
        type=float,
        required=False,
        help=(
            "Threshold to exclude entire run if more than the specified "
            "percentage of steady-state volumes are removed."
        ),
    )
    parser.add_argument(
        "--n_dummy_scans",
        dest="n_dummy_scans",
        default="auto",
        type=n_dummy_type,
        required=False,
        help=(
            "Number of dummy scans to remove. If 'auto' computes number of dummy scans "
            "by the numnber of 'non_steady_state_outlier_XX' columns."
        ),
    )
    parser.add_argument(
        "--n_acompcor",
        dest="n_acompcor",
        default=5,
        type=int,
        required=False,
        help="Number of aCompCor components.",
    )
    parser.add_argument(
        "--acompcor_strategy",
        dest="acompcor_strategy",
        default="separate",
        choices=["combined", "separate", "none"],
        type=str,
        required=False,
        help="Whether to use 'combined' aCompCor, 'separate' components, or 'none'.",
    )
    parser.add_argument(
        "--fwhm",
        dest="fwhm",
        default=6,
        type=int,
        required=False,
        help="Spatial blurring.",
    )
    parser.add_argument(
        "--upsample_dt",
        dest="upsample_dt",
        default=0.1,
        type=none_or(float),
        required=False,
        help=(
            "Time resolution to upsample seed timeseries (and condition times) to prior "
            "to deconvolution. Set to 'none' to skip upsampling"
        ),
    )
    parser.add_argument(
        "--pad_seconds",
        dest="pad_seconds",
        default=30.0,
        type=float,
        required=False,
        help=(
            "Time in seconds to determine the padding to add to both ends (pad_seconds/upsample_dt) "
            "to minimize boundary spikes prior to deconvolution. The padding is dropped immediatelly afterwards "
            "so the final deconvolved timeseries includes no padding."
        ),
    )
    parser.add_argument(
        "--faltung_penalty_syntax",
        dest="faltung_penalty_syntax",
        default="012 0",
        required=False,
        type=str,
        help=(
            "Deconvolution penalty syntax to pass to the FALTUNG parameter in 3dTfitter "
            "(fset fpre pen fac). See: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTfitter.html"
        ),
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


def extract_seed_timeseries(
    subject_dir,
    subject_nifti_file,
    seed_mask_path,
    afni_img_path,
):
    LGR.info(f"Using the following seed mask file: {seed_mask_path}")

    possible_coordinate = get_coordinate_from_filename(
        seed_mask_path,
        replace_underscore=False,
    )
    if possible_coordinate:
        seed_name = f"seed_{possible_coordinate}"
    else:
        seed_name = "seed"

    seed_timeseries_file = subject_dir / "seed" / f"{seed_name}_desc-timeseries.1D"
    seed_timeseries_file.parent.mkdir(parents=True, exist_ok=True)

    seed_img = resample_seed_img(nib.load(seed_mask_path), nib.load(subject_nifti_file))

    resampled_seed_file = subject_dir / f"resampled_{seed_mask_path.name}"
    nib.save(seed_img, resampled_seed_file)

    # Note: output is a column vector
    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dmaskave "
        f"-mask {resampled_seed_file} "
        f"-q {subject_nifti_file} > {seed_timeseries_file}"
    )

    LGR.info(f"Extracting seed: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    resampled_seed_file.unlink()

    return seed_timeseries_file


def denoise_seed_timeseries(
    seed_timeseries_file,
    nuisance_regressors_file,
    censor_file,
    afni_img_path,
    cosine_regressor_names,
):
    polort = 0 if cosine_regressor_names else "A"
    LGR.info(f"Using polort {polort} for 3dTproject.")

    denoised_seed_timeseries_file = (
        seed_timeseries_file.parent
        / seed_timeseries_file.name.replace("_desc-timeseries", "_desc-denoised")
    )

    # Note: Some Afni functions only accept rows and require \', using \\' to
    # make the backslash literal
    # Check the diagnostic plots to ensure there are not several long continuous gaps of
    # censored volumes since that run may need to be discarded. Appears to be
    # no best approach for dealing with high-motion prior to deconvolution
    # everything has a tradeoff. The framewise displacement threshold of 0.5
    # combined with outlier threshold of 0.20 should capture a good amount of
    # unusable runs with severe motion
    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dTproject "
        f"-input {seed_timeseries_file}\\' "
        f"-ort {nuisance_regressors_file} "
        f"-polort {polort} "
        f"-censor {censor_file} "
        "-cenmode NTRP "
        f"-prefix {denoised_seed_timeseries_file}"
    )

    LGR.info(
        f"Denoising seed (same nuisance regressors used for seed and BOLD/NIfTI image): {cmd}"
    )
    subprocess.run(cmd, shell=True, check=True)

    return denoised_seed_timeseries_file


def get_cue_name(timing_dir, cohort, task, condition_filenames):
    if task in ("mtle", "mtlr") or (task == "nback" and cohort == "kids"):
        return condition_filenames + [timing_dir / f"{task}_cue.1D"]

    return condition_filenames


def resample_data(target_file, tr, afni_img_path, upsample_dt, method):
    if method == "upsample":
        resampled_filename = target_file.parent / target_file.name.replace(
            "_desc-denoised",
            "_desc-upsampled",
        )

        # New length of interpolated timseries is (tr / upsample_dt) * n_original_volumes
        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 1dUpsample {int(tr / upsample_dt)} "
            f"{str(target_file)}\\' > {resampled_filename}"
        )

        LGR.info(f"Upsampling seed timeseries from {tr} to {upsample_dt}: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    else:
        # original TR divided by sub_TR, starts at the first tr (0) and takes every
        # (tr / upsample_dt) point
        resampled_filename = (
            target_file.parent
            / f"PPI_{target_file.name.replace('_desc-PPI_upsampled.1D', '.1D')}"
        )
        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 1dcat "
            f"'{target_file}{{0..$({int(tr / upsample_dt)})}}' > {resampled_filename}"
        )

        LGR.info(
            f"Downsampling the PPI regressor back to the original {tr} s grid: {cmd}"
        )
        subprocess.run(cmd, shell=True, check=True)

    return resampled_filename


def deconvolve_seed_timeseries(
    seed_timeseries_file,
    dt,
    pad_seconds,
    faltung_penalty_syntax,
    afni_img_path,
    task,
    input_desc="upsampled",
):
    gamma_file_name = seed_timeseries_file.parent / "GammaHR.1D"
    deconvolved_seed_timeseries_file = (
        seed_timeseries_file.parent
        / seed_timeseries_file.name.replace(
            f"_desc-{input_desc}",
            "_desc-deconvolved",
        )
    )

    padded_deconvolved_seed_timeseries_file = (
        deconvolved_seed_timeseries_file.parent
        / deconvolved_seed_timeseries_file.name.replace(
            "_desc-deconvolved",
            "_desc-deconvolved_padded",
        )
    )

    # Use some padding for smooth ramp up at ends
    pad_length = int(pad_seconds / dt)
    padded_seed_timeseries_file = (
        seed_timeseries_file.parent
        / seed_timeseries_file.name.replace(
            f"_desc-{input_desc}",
            f"_desc-{input_desc}_padded",
        )
    )
    padded_arr = np.pad(
        np.loadtxt(seed_timeseries_file),
        pad_width=pad_length,
        mode="reflect",
    )
    np.savetxt(
        padded_seed_timeseries_file,
        padded_arr.reshape(-1, 1),
        fmt="%f",
    )

    # https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html
    # https://doi.org/10.1002/hbm.26047
    # Creating 30 second hrf
    hrf_model = "GAM" if task in EVENT_RELATED_TASKS else f"BLOCK({dt},1)"

    hrf_cmd = (
        f"3dDeconvolve -nodata {int(30 / dt)} {dt} -polort -1 "
        f"-num_stimts 1 -stim_times 1 '1D: 0' '{hrf_model}' "
        f"-x1D {gamma_file_name}_tmp -x1D_stop -quiet && "
        f"1dcat {gamma_file_name}_tmp > {gamma_file_name}"
    )

    # Perform deconvolution to estimate the neural response given the seed timeseries
    # and an hrf response function, while also adding a penalty for better/smoother estimation
    cmd = (
        f'apptainer exec -B /projects:/projects {afni_img_path} bash -c "{hrf_cmd} && '
        f"3dTfitter -RHS {padded_seed_timeseries_file} "
        f'-FALTUNG {gamma_file_name} {padded_deconvolved_seed_timeseries_file} {faltung_penalty_syntax}"'
    )

    LGR.info(f"Deconvolving seed timeseries (dt={dt}): {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    deconvolved_arr = np.loadtxt(padded_deconvolved_seed_timeseries_file)[
        pad_length:-pad_length
    ]
    np.savetxt(
        deconvolved_seed_timeseries_file,
        deconvolved_arr.reshape(-1, 1),
        fmt="%f",
    )

    padded_seed_timeseries_file.unlink()
    padded_deconvolved_seed_timeseries_file.unlink()
    Path(f"{gamma_file_name}_tmp").unlink()

    return deconvolved_seed_timeseries_file, cmd


def mean_center_condition_vector(condition_regressor_file, save_filename):
    condition_vector = np.loadtxt(condition_regressor_file)
    condition_vector -= condition_vector.mean()
    np.savetxt(
        save_filename,
        condition_vector.reshape(-1, 1),
        fmt="%f",
    )


def upsample_condition_regressor(
    timing_file,
    cohort,
    task,
    tr,
    n_volumes,
    upsample_dt,
    afni_img_path,
):
    condition_name = timing_file.name.removesuffix(".1D")

    upsampled_condition_regressor_file = (
        timing_file.parent / "upsampled" / f"{condition_name}_desc-upsampled.1D"
    )
    upsampled_condition_regressor_file.parent.mkdir(parents=True, exist_ok=True)

    duration = (
        CONDITION_DURATIONS[cohort][task]
        if not condition_name.endswith("_cue")
        else CONDITION_DURATIONS[cohort][condition_name]
    )

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} timing_tool.py "
        f"-timing {timing_file} "
        f"-tr {upsample_dt} "
        f"-stim_dur {duration} "
        f"-run_len {tr * n_volumes} "
        f"-timing_to_1D {upsampled_condition_regressor_file}"
    )

    LGR.info(f"Upsampling condition {condition_name} to {upsample_dt} s: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    # Now mean center the task regressor
    mean_center_condition_vector(
        upsampled_condition_regressor_file, upsampled_condition_regressor_file
    )

    return upsampled_condition_regressor_file


def create_convolved_ppi_term(
    ppi_dir,
    deconvolved_seed_timeseries_file,
    condition_regressor_file,
    afni_img_path,
    dt,
    condition_desc="upsampled",
):
    neural_interaction_file = (
        deconvolved_seed_timeseries_file.parent
        / condition_regressor_file.name.replace(
            f"_desc-{condition_desc}",
            "_desc-neural_interaction",
        )
    )
    ppi_regressor_file = ppi_dir / condition_regressor_file.name.replace(
        f"_desc-{condition_desc}",
        f"_desc-PPI_{condition_desc}",
    )

    numout = np.loadtxt(deconvolved_seed_timeseries_file).size
    gamma_file_name = deconvolved_seed_timeseries_file.parent / "GammaHR.1D"

    # PPI = ([neural signal * binary_condition_vector] * hrf)(t)
    convolution_cmd = f"waver -FILE {dt} {gamma_file_name} -peak 1 -input {neural_interaction_file} -numout {numout} > {ppi_regressor_file}"

    # Create the interaction, which simply zeroes the parts when the condition is not active
    # Then reconvolve the interaction term to get the estimated HRF, ensure no extended tail due to convolution
    # So regressor can be properly downsampled
    cmd = (
        f'apptainer exec -B /projects:/projects {afni_img_path} bash -c "1deval '
        f"-a {deconvolved_seed_timeseries_file} -b {condition_regressor_file} "
        f"-expr 'a*b' > {neural_interaction_file} && "
        f'{convolution_cmd}"'
    )

    LGR.info(f"Reconvolving PPI regressor: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return ppi_regressor_file, cmd


def main(
    bids_dir,
    afni_img_path,
    dst_dir,
    deriv_dir,
    seed_mask_path,
    cohort,
    subject,
    space,
    task,
    filter_correct_trials,
    n_motion_parameters,
    fd_threshold,
    exclusion_criteria,
    n_dummy_scans,
    n_acompcor,
    acompcor_strategy,
    fwhm,
    upsample_dt,
    pad_seconds,
    faltung_penalty_syntax,
    exclude_niftis_file,
):
    seed_mask_name_check(seed_mask_path)

    report_dir, layout, sessions = validate_first_level_inputs(
        dst_dir,
        bids_dir,
        deriv_dir,
        cohort,
        task,
        subject,
        analysis_type="gPPI",
    )

    for session in sessions:
        report, report_path, session_files, skip_iteration = collect_session_files(
            report_dir,
            layout,
            subject,
            session,
            task,
            space,
            acompcor_strategy,
            exclude_niftis_file,
            analysis_type="gPPI",
        )
        if skip_iteration:
            continue

        confounds_tsv_file = session_files.confounds_tsv_file
        confounds_json_file = session_files.confounds_json_file
        event_file = session_files.event_file
        mask_file = session_files.mask_file
        nifti_file = session_files.nifti_file

        subject_dir = (
            Path(dst_dir) / f"sub-{subject}" / f"ses-{session}" / "func" / task
        )
        subject_dir.mkdir(parents=True, exist_ok=True)

        confounds_df = pd.read_csv(confounds_tsv_file, sep="\t").fillna(0)

        try:
            censor_mask, censor_info = check_censoring(
                subject,
                session,
                task,
                confounds_df,
                n_dummy_scans,
                fd_threshold,
                exclusion_criteria,
            )
        except InvalidRunError as exc:
            report.mark_excluded(str(exc))
            report.create_report(report_path, "first_level.html")
            continue

        report.add_context(
            fd_threshold=fd_threshold,
            exclusion_criteria=exclusion_criteria,
            n_censored_volumes=censor_info.n_censored,
            n_total_volumes=censor_info.n_total,
            percent_censored=censor_info.percent_censored,
            dummy_method=censor_info.dummy_method,
            n_dummy_scans=censor_info.n_dummy_scans,
            mean_fd_before_censoring=censor_info.mean_fd_before_censoring,
            mean_fd_after_censoring=censor_info.mean_fd_after_censoring,
        )

        censor_file = create_censor_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            censor_mask,
        )

        high_motion_only_mask = censor_mask.copy()
        high_motion_only_mask[: censor_info.n_dummy_scans] = 1
        create_censor_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            high_motion_only_mask,
            desc="high_motion_outliers_only",
        )

        create_fd_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            confounds_df,
            censor_mask,
            censor_info.n_dummy_scans,
        )

        cosine_regressors, cosine_regressor_names = get_cosine_regressors(confounds_df)

        motion_regressors, motion_regressor_names = get_motion_regressors(
            confounds_df,
            n_motion_parameters,
        )

        acompcor_regressors, acompcor_regressor_names = collect_acompcor_names(
            confounds_json_file, confounds_df, acompcor_strategy, n_acompcor
        )

        report.add_context(
            n_motion_parameters=n_motion_parameters,
            motion_regressor_names=motion_regressor_names,
            acompcor_strategy=acompcor_strategy,
            n_acompcor=n_acompcor,
            acompcor_component_names=acompcor_regressor_names or [],
            cosine_parameter_names=cosine_regressor_names,
            fwhm=fwhm,
            filter_correct_trials=filter_correct_trials,
        )

        regressor_names = filter_regressor_names(
            cosine_regressor_names, motion_regressor_names, acompcor_regressor_names
        )
        nuisance_regressors_file, report_info = create_nuisance_regressor_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            censor_mask,
            regressor_names,
            cosine_regressors,
            motion_regressors,
            acompcor_regressors,
            analysis_type="gPPI",
        )

        report.add_context(
            dropped_regressors=(
                report_info.collinear_regressor_names
                + report_info.constant_column_names
            )
        )

        timing_dir = create_timing_files(
            subject_dir,
            event_file,
            task=task,
            filter_correct_trials=filter_correct_trials,
            append_task_name=False,
        )

        tr = get_tr(nifti_file)
        n_volumes = get_n_volumes(nifti_file)

        condition_filenames_dict = create_binary_condition(
            afni_img_path,
            timing_dir,
            cohort,
            task,
            tr,
            n_volumes,
            censor_file,
        )

        diagnostic_condition_plots = create_diagnostic_condition_plots(
            condition_filenames_dict,
            tr,
            fd_threshold,
        )
        report.add_context(diagnostic_condition_plots=diagnostic_condition_plots)

        timing_conditions, event_type = summarize_timing_conditions(timing_dir, task)
        report.add_context(timing_conditions=timing_conditions, event_type=event_type)

        percent_change_nifti_file = percent_signal_change(
            subject_dir,
            afni_img_path,
            nifti_file,
            mask_file,
            censor_file,
        )

        # gPPI preparation
        seed_mask_path = Path(seed_mask_path)

        do_upsample = upsample_dt is not None
        effective_dt = upsample_dt if do_upsample else tr

        hrf_model_type = (
            "GAM" if task in EVENT_RELATED_TASKS else f"BLOCK({effective_dt}, 1)"
        )
        hrf_model_desc = (
            "A standard Gamma (GAM) function was used to model the impulse response for this event-related task."
            if task in EVENT_RELATED_TASKS
            else f"A custom {effective_dt}s duration BLOCK function was simulated via 3dDeconvolve to model the impulse response for this block-design task."
        )

        report.add_context(
            seed_mask_path=str(seed_mask_path),
            seed_coordinate=get_coordinate_from_filename(seed_mask_path),
            seed_radius=get_sphere_radius(seed_mask_path),
            do_upsample=do_upsample,
            upsample_dt=upsample_dt,
            pad_seconds=pad_seconds,
            pad_length=int(pad_seconds / effective_dt),
            faltung_penalty_syntax=faltung_penalty_syntax,
            tr=tr,
            hrf_model_type=hrf_model_type,
            hrf_model_desc=hrf_model_desc,
        )

        ppi_dir = timing_dir / "ppi"
        ppi_dir.mkdir(parents=True, exist_ok=True)

        seed_timeseries_file = extract_seed_timeseries(
            subject_dir,
            percent_change_nifti_file,
            seed_mask_path,
            afni_img_path,
        )
        seed_timeseries_plot_filename = plot_signal(
            seed_timeseries_file,
            tr,
            "Seed Timeseries",
        )

        denoised_seed_timeseries_file = denoise_seed_timeseries(
            seed_timeseries_file,
            nuisance_regressors_file,
            censor_file,
            afni_img_path,
            cosine_regressor_names,
        )
        denoised_seed_timeseries_plot_filename = plot_signal(
            denoised_seed_timeseries_file,
            tr,
            "Denoised Seed Timeseries",
        )

        if do_upsample:
            seed_input_for_deconvolution = resample_data(
                denoised_seed_timeseries_file,
                tr,
                afni_img_path,
                upsample_dt,
                method="upsample",
            )
            upsampled_seed_timeseries_plot_filename = plot_signal(
                seed_input_for_deconvolution,
                tr,
                "Upsampled Seed Timeseries",
                upsample_dt,
            )
            seed_input_desc = "upsampled"
        else:
            seed_input_for_deconvolution = denoised_seed_timeseries_file
            upsampled_seed_timeseries_plot_filename = None
            seed_input_desc = "denoised"

        deconvolved_seed_timeseries_file, deconvolve_seed_cmd = (
            deconvolve_seed_timeseries(
                seed_input_for_deconvolution,
                effective_dt,
                pad_seconds,
                faltung_penalty_syntax,
                afni_img_path,
                task,
                input_desc=seed_input_desc,
            )
        )
        deconvolved_seed_timeseries_plot_filename = plot_signal(
            deconvolved_seed_timeseries_file,
            tr,
            "Deconvolved Seed Timeseries",
            upsample_dt,
        )

        report.add_context(
            deconvolve_seed_cmd=deconvolve_seed_cmd,
            seed_timeseries_plot=embed_img(seed_timeseries_plot_filename),
            denoised_seed_timeseries_plot=embed_img(
                denoised_seed_timeseries_plot_filename,
            ),
            upsampled_seed_timeseries_plot=(
                embed_img(upsampled_seed_timeseries_plot_filename)
                if upsampled_seed_timeseries_plot_filename
                else None
            ),
            deconvolved_seed_timeseries_plot=embed_img(
                deconvolved_seed_timeseries_plot_filename,
            ),
        )

        first_level_gltsym_codes = get_first_level_gltsym_codes(
            cohort,
            task,
            analysis_type="glm",  # This is intentional to get the non-gPPI condition names
            caller="gPPI",
        )
        condition_filenames = [
            timing_dir / f"{condition}.1D"
            for condition in get_beta_names(first_level_gltsym_codes)
            if "_vs_" not in condition
        ]
        condition_filenames = get_cue_name(
            timing_dir,
            cohort,
            task,
            condition_filenames,
        )

        errors_timing_file = timing_dir / "errors.1D"
        if (
            not is_timing_file_empty(errors_timing_file)
            and errors_timing_file not in condition_filenames
        ):
            condition_filenames = condition_filenames + [errors_timing_file]

        condition_names = []
        condition_plots = []
        for condition_filename in condition_filenames:
            if is_timing_file_empty(condition_filename):
                continue

            cond_name = condition_filename.name.removesuffix(".1D")
            condition_names.append(cond_name)

            if do_upsample:
                condition_regressor_file = upsample_condition_regressor(
                    condition_filename,
                    cohort,
                    task,
                    tr,
                    n_volumes,
                    upsample_dt,
                    afni_img_path,
                )
                condition_regressor_plot_filename = plot_signal(
                    condition_regressor_file,
                    tr,
                    f"{cond_name.capitalize()} Upsampled Condition Regressor",
                    upsample_dt,
                )
                condition_desc = "upsampled"
            else:
                binary_vector_file = condition_filenames_dict[cond_name][
                    "noncensored_binary_vector"
                ]
                condition_dir = timing_dir / "condition_regressors"
                condition_dir.mkdir(parents=True, exist_ok=True)
                condition_regressor_file = (
                    condition_dir / f"{cond_name}_desc-centered.1D"
                )
                mean_center_condition_vector(
                    binary_vector_file, condition_regressor_file
                )
                condition_regressor_plot_filename = plot_signal(
                    condition_regressor_file,
                    tr,
                    f"{cond_name.capitalize()} Mean-Centered Condition Regressor",
                )
                condition_desc = "centered"

            ppi_regressor_file, ppi_cmd = create_convolved_ppi_term(
                ppi_dir,
                deconvolved_seed_timeseries_file,
                condition_regressor_file,
                afni_img_path,
                effective_dt,
                condition_desc=condition_desc,
            )

            if do_upsample:
                ppi_plot_filename = plot_signal(
                    ppi_regressor_file,
                    tr,
                    f"{cond_name.capitalize()} Upsampled PPI Timeseries",
                    upsample_dt,
                )
                final_ppi_regressor_file = resample_data(
                    ppi_regressor_file,
                    tr,
                    afni_img_path,
                    upsample_dt,
                    method="downsample",
                )
                final_ppi_plot_filename = plot_signal(
                    final_ppi_regressor_file,
                    tr,
                    f"{cond_name.capitalize()} Downsampled PPI Timeseries",
                )
            else:
                ppi_plot_filename = None
                final_ppi_regressor_file = ppi_dir / f"PPI_{cond_name}.1D"
                ppi_regressor_file.rename(final_ppi_regressor_file)
                final_ppi_plot_filename = plot_signal(
                    final_ppi_regressor_file,
                    tr,
                    f"{cond_name.capitalize()} PPI Timeseries",
                )

            condition_plots.append(
                {
                    "name": cond_name,
                    "do_upsample": do_upsample,
                    "condition_regressor_plot": embed_img(
                        condition_regressor_plot_filename,
                    ),
                    "ppi_cmd": ppi_cmd.replace("  ", " "),
                    "ppi_plot": (
                        embed_img(ppi_plot_filename) if ppi_plot_filename else None
                    ),
                    "final_ppi_plot": embed_img(final_ppi_plot_filename),
                }
            )

        report.add_context(
            condition_names=condition_names,
            condition_plots=condition_plots,
        )

        smoothed_nifti_file = perform_spatial_smoothing(
            subject_dir.parent,
            afni_img_path,
            percent_change_nifti_file,
            mask_file,
            fwhm,
        )

        get_task_deconvolve_cmd = {
            "kids": get_task_deconvolve_kids_cmd,
            "adults": get_task_deconvolve_adults_cmd,
        }

        deconvolve_cmd = get_task_deconvolve_cmd[cohort](
            task,
            timing_dir,
            nuisance_regressors_file,
            analysis_type="gPPI",
            seed_timeseries_file=seed_timeseries_file,
            ppi_dir=ppi_dir,
        )

        report.add_context(
            deconvolve_cmd=f"{deconvolve_cmd['num_stimts']} {deconvolve_cmd['args']}".replace(
                "  ", " "
            ),
        )

        design_matrix_file = create_design_matrix(
            subject_dir.parent,
            afni_img_path,
            smoothed_nifti_file,
            mask_file,
            censor_file,
            deconvolve_cmd,
            cosine_regressor_names,
        )

        stats_file_relm = perform_first_level(
            subject_dir,
            afni_img_path,
            design_matrix_file,
            smoothed_nifti_file,
            mask_file,
        )

        betas_dir = stats_file_relm.parent / "betas"
        betas_dir.mkdir(parents=True, exist_ok=True)

        create_beta_files(
            stats_file_relm,
            betas_dir,
            afni_img_path,
            cohort,
            task,
            analysis_type="gPPI",
        )

        report.create_report(report_path, "first_level.html")


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
