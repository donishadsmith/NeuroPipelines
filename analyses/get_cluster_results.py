import argparse, itertools, subprocess, sys
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd
from nilearn.image import threshold_img
from nilearn.plotting import plot_stat_map, plot_roi
from nilearn.reporting import get_clusters_table
from matplotlib.colors import ListedColormap
from scipy.stats import norm

from bidsaid.logging import setup_logger
from bidsaid.parsers import _is_float
from bidsaid.masking import create_sphere_mask

from _report import HTMLReport
from _utils import (
    create_condition_label_str,
    delete_dir,
    get_contrast_entity_key,
    get_coordinate_from_filename,
    get_first_level_gltsym_codes,
    get_group_labels,
    get_second_level_glt_codes,
    resample_seed_img,
    save_binary_mask,
)

LGR = setup_logger(__name__)

# For nonparametric approach which is already thresholded
ZERO_STAT_THRESHOLD = 0
ZERO_CLUSTER_SIZE = 0


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description=(
            "If parametric, apply NN1 2-sided cluster correction to data "
            "and identify significant clusters. If nonparametric, just identifies "
            " significant clusters."
        )
    )
    parser.add_argument(
        "--second_level_report_dir",
        dest="second_level_report_dir",
        required=True,
        help=("Path to the report directory containing the second level html."),
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help=(
            "Root path to directory containing second level stats "
            "and cluster correction tables."
        ),
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=False,
        default=None,
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        required=True,
        choices=["glm", "gPPI"],
        help="The type of analysis performed (glm or gPPI).",
    )
    parser.add_argument(
        "--seed_mask_path",
        dest="seed_mask_path",
        required=False,
        default=None,
        help=(
            "Path to the seed mask used as the seed for the gPPI. "
            "Used only when ``analysis_type`` is gPPI. "
        ),
    )
    parser.add_argument(
        "--method",
        dest="method",
        required=False,
        default="parametric",
        choices=["parametric", "nonparametric"],
        help="Whether parametric (3dlmer) or nonparametric (Palm) was used.",
    )
    parser.add_argument(
        "--parametric_connectivity",
        dest="parametric_connectivity",
        required=False,
        default="NN1",
        help="Connectivity to use for parametric. Will always use 2-sided/bisided version.",
    )
    parser.add_argument(
        "--parametric_voxel_correction_p",
        dest="parametric_voxel_correction_p",
        required=False,
        default=0.001,
        type=float,
        help=("P-value for voxel correction. Only used for the parametric approach."),
    )
    parser.add_argument(
        "--parametric_cluster_correction_p",
        dest="parametric_cluster_correction_p",
        required=False,
        default=0.05,
        type=float,
        help="P-value for cluster correction. Only used for the parametric approach.",
    )
    parser.add_argument(
        "--template_mask_path",
        dest="template_mask_path",
        required=False,
        default=None,
        help="Path to a template brain mask image to use for creating sphere masks.",
    )
    parser.add_argument(
        "--template_img_path",
        dest="template_img_path",
        required=False,
        default=None,
        help="Path to a T1w template image to use for plotting.",
    )
    parser.add_argument(
        "--sphere_radius",
        dest="sphere_radius",
        required=False,
        type=float,
        default=5.0,
        help=(
            "The radius of the sphere for the MNI coordinate. If `analysis_type` is 'glm', "
            "seed masks are only created for the mean `second_level_glt_code`, which "
            "are the maps denoting activation and deactivation for all subjects. "
            "This is done so that seeds are not deliberately biased towards a specific group."
        ),
    )

    return parser


def p_to_z(p_value, two_sided=True):
    return norm.ppf(1 - p_value / (2 if two_sided else 1))


def get_zscore_map_and_mask(
    analysis_dir,
    afni_img_path,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
):
    stats_filename = next(
        analysis_dir.rglob(
            f"task-{task}_{entity_key}-{first_level_glt_label}_desc-parametric_stats.nii.gz"
        )
    )
    group_mask_filename = next(
        analysis_dir.rglob(
            f"task-{task}_{entity_key}-{first_level_glt_label}_desc-parametric_group_mask.nii.gz"
        )
    )

    zcore_map_filename = str(stats_filename).replace(
        "_desc-parametric_stats",
        f"_gltcode-{second_level_glt_code}_desc-parametric_z_map",
    )

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dbucket "
        f"{stats_filename}'[{second_level_glt_code} Z]' "
        f"-prefix {zcore_map_filename} "
        "-overwrite"
    )
    LGR.info(f"Extracting {second_level_glt_code} z score map: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception:
        LGR.warning(f"The following command failed: {cmd}", exc_info=True)

    return Path(zcore_map_filename), group_mask_filename


def get_cluster_correction_table(
    analysis_dir, task, entity_key, first_level_glt_label, parametric_connectivity
):
    cluster_correction_filename = next(
        analysis_dir.rglob(
            f"task-{task}_{entity_key}-{first_level_glt_label}"
            f"_desc-cluster_correction.{parametric_connectivity}_bisided.1D"
        )
    )
    cluster_correction_table = pd.DataFrame(
        np.loadtxt(cluster_correction_filename),
        columns=["pthr", ".10000", ".05000", ".02000", ".01000"],
    )

    return cluster_correction_table.astype(float)


def get_cluster_size(
    cluster_correction_table,
    parametric_voxel_correction_p,
    parametric_cluster_correction_p,
    verbose=True,
):
    cluster_p_values = list(map(float, cluster_correction_table.columns[1:]))
    cluster_p_values_arr = np.array(cluster_p_values)
    clust_p_index = (
        np.where((cluster_p_values_arr == parametric_cluster_correction_p) == True)[0][
            0
        ]
        + 1
    )
    cluster_p_str = cluster_correction_table.columns[clust_p_index]

    cluster_size = int(
        np.ceil(
            cluster_correction_table.loc[
                cluster_correction_table["pthr"] == parametric_voxel_correction_p,
                cluster_p_str,
            ].to_numpy(copy=True)[0]
        )
    )

    if verbose:
        LGR.info(
            "Cluster size for bisided using cluster-forming threshold "
            f"{parametric_voxel_correction_p} and cluster correction {parametric_cluster_correction_p}: {cluster_size}"
        )

    return cluster_size


def identify_clusters(
    dst_dir,
    thresholded_img,
    analysis_type,
    seed_mask_path,
    method,
    stat_threshold,
    cluster_size,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
):
    # https://github.com/nilearn/nilearn/blob/b33c40c9cd2995811863acf67bcb5594320ea7ec/nilearn/reporting/get_clusters_table.py#L505-L506
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    # Nilearn uses NN1: bin_struct = generate_binary_structure(rank=3, connectivity=1)
    clusters_table, labels_map_list = get_clusters_table(
        thresholded_img,
        stat_threshold=(
            p_to_z(stat_threshold) if method == "parametric" else stat_threshold
        ),
        cluster_threshold=cluster_size,
        two_sided=True,
        return_label_maps=True,
    )

    cluster_table_filename = (
        dst_dir
        / "cluster_results"
        / method
        / (
            f"task-{task}_{entity_key}-{first_level_glt_label}"
            f"_gltcode-{second_level_glt_code}_desc-{method}_cluster_results.csv"
        )
    )
    cluster_table_filename.parent.mkdir(parents=True, exist_ok=True)

    if not clusters_table.empty:
        n_rows = clusters_table.shape[0]
        clusters_table["Statistic Name"] = (
            ["Z-score"] * n_rows if method == "parametric" else ["T-score"] * n_rows
        )

        clusters_table["Cluster ID"] = clusters_table["Cluster ID"].astype(str)

        peaks = clusters_table["Peak Stat"].astype(float).to_numpy(copy=True)
        mask_primary = clusters_table["Cluster ID"].str.isdigit()
        mask_pos = mask_primary & (peaks > 0)
        mask_neg = mask_primary & (peaks < 0)

        if second_level_glt_code == "mean":
            interpretation = (
                "activation" if analysis_type == "glm" else "connectivity with seed ROI"
            )
            clusters_table.loc[mask_pos, "Interpretation"] = (
                f"mean {interpretation} across doses > 0"
            )
            clusters_table.loc[mask_neg, "Interpretation"] = (
                f"mean {interpretation} across doses < 0"
            )
        else:
            first_label, second_label = get_group_labels(second_level_glt_code)

            suffix = " mg MPH" if _is_float(first_label) else ""
            end_str = (
                "; greater activation"
                if analysis_type == "glm"
                else "; greater connectivity"
            )

            clusters_table.loc[mask_pos, "Interpretation"] = (
                f"{first_label}{suffix} > {second_label}{suffix}{end_str}"
            )
            clusters_table.loc[mask_neg, "Interpretation"] = (
                f"{second_label}{suffix} > {first_label}{suffix}{end_str}"
            )

        possible_coordinate = (
            get_coordinate_from_filename(seed_mask_path)
            if analysis_type == "gPPI"
            else ""
        )
        if possible_coordinate:
            clusters_table["Seed MNI Coordinate"] = possible_coordinate

        clusters_table["Condition Label"] = create_condition_label_str(
            first_level_glt_label
        )

        clusters_table["Analysis Method"] = f"{method} {analysis_type}"

        clusters_table.to_csv(cluster_table_filename, sep=",", index=False)

        # Get label map
        label_base_dir = dst_dir / "cluster_masks" / method
        label_base_dir.mkdir(parents=True, exist_ok=True)
        for label_map in labels_map_list:
            label_ids = np.unique(label_map.get_fdata()[label_map.get_fdata() != 0])
            if label_ids.shape[0] == 0:
                continue

            label_map_fdata = label_map.get_fdata()
            thresholded_img_fdata = thresholded_img.get_fdata()

            voxel_data = thresholded_img_fdata[label_map_fdata != 0].ravel()
            voxel_stats = voxel_data[voxel_data != 0]
            peak_voxel_val = voxel_stats[np.argmax(np.abs(voxel_stats))]
            tail = "positive" if peak_voxel_val > 0 else "negative"

            peak_stats = clusters_table["Peak Stat"].to_numpy(copy=True)
            if tail == "positive":
                peak_stats_indices = clusters_table.loc[
                    peak_stats > 0, "Peak Stat"
                ].index.tolist()
            else:
                peak_stats_indices = clusters_table.loc[
                    peak_stats < 0, "Peak Stat"
                ].index.tolist()

            cluster_ids = clusters_table.loc[peak_stats_indices, "Cluster ID"].to_numpy(
                copy=True
            )
            cluster_ids = [label for label in cluster_ids if label.isdigit()]
            cluster_id_map = {
                index: label for index, label in enumerate(cluster_ids, start=1)
            }

            for label_id in label_ids:
                cluster_id = cluster_id_map[int(label_id)]

                cluster_row = clusters_table.loc[
                    clusters_table["Cluster ID"] == cluster_id
                ]
                coord = cluster_row[["X", "Y", "Z"]].values[0].tolist()
                coord_name = "_".join([str(x) for x in coord])

                label_mask_fdata = np.zeros_like(label_map.get_fdata())
                label_mask_fdata[label_map.get_fdata() == label_id] = 1

                label_mask_filename = label_base_dir / (
                    f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}"
                    f"_clusterid-{cluster_id}_tail-{tail}_desc-{method}_cluster_mask_{coord_name}.nii.gz"
                )

                save_binary_mask(
                    label_mask_fdata,
                    label_map.affine,
                    label_map.header.copy(),
                    label_mask_filename,
                )

    return cluster_table_filename


def plot_thresholded_img(
    dst_dir,
    thresholded_img,
    template_img_path,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    method,
):
    max_stat = np.max(np.abs(thresholded_img.get_fdata()))
    vmax = max_stat
    vmin = max_stat * -1

    kwargs = {
        "stat_map_img": thresholded_img,
        "draw_cross": False,
        "vmax": vmax,
        "vmin": vmin,
    }
    if template_img_path and Path(template_img_path).exists():
        kwargs.update({"bg_img": template_img_path})

    for mode in ["ortho", "mosaic"]:
        for bg_color, black_bg in [("black", True), ("white", False)]:
            display = plot_stat_map(**kwargs, display_mode=mode, black_bg=black_bg)

            statistic = "Z-score" if method == "parametric" else "T-score"
            display._cbar.set_label(f"{statistic} Intensity")

            plot_filename = (
                dst_dir
                / "stat_plots"
                / method
                / (
                    f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}"
                    f"_displaymode-{mode}_background-{bg_color}_desc-{method}_cluster.png"
                )
            )
            plot_filename.parent.mkdir(parents=True, exist_ok=True)
            display.savefig(plot_filename, dpi=720)

            display.close()


def create_seed_masks(
    dst_dir,
    method,
    cluster_table_filename,
    template_mask_path,
    template_img_path,
    thresholded_img,
    sphere_radius,
):
    sphere_parent_path = dst_dir / "sphere_masks" / method
    sphere_parent_path.mkdir(parents=True, exist_ok=True)

    plot_parent_path = sphere_parent_path / "plots"
    plot_parent_path.mkdir(parents=True, exist_ok=True)

    clusters_table = pd.read_csv(cluster_table_filename, sep=",")
    template_mask = nib.load(template_mask_path)

    clusters_table["Cluster ID"] = clusters_table["Cluster ID"].astype(str)
    mask_primary = clusters_table["Cluster ID"].str.isdigit()

    truncated_clusters_table = clusters_table.loc[mask_primary, ["X", "Y", "Z"]]
    for index in truncated_clusters_table.index.to_list():
        coord = truncated_clusters_table.loc[index, ["X", "Y", "Z"]].to_list()

        sphere_mask = create_sphere_mask(tuple(coord), sphere_radius, template_mask)

        coord_name = "_".join([str(x) for x in coord])
        sphere_name = (
            cluster_table_filename.name.replace("_cluster_results.csv", "_sphere_mask_")
            + f"{coord_name}.nii.gz"
        )
        sphere_name = sphere_name.replace("_desc", f"_radius-{sphere_radius}mm_desc")
        sphere_filename = sphere_parent_path / sphere_name

        nib.save(sphere_mask, sphere_filename)

        kwargs = {
            "roi_img": sphere_filename,
            "draw_cross": False,
            "cmap": ListedColormap(["red"]),
            "colorbar": False,
        }
        if template_img_path and Path(template_img_path).exists():
            kwargs.update({"bg_img": template_img_path})

        for bg_color, black_bg in [("black", True), ("white", False)]:
            display = plot_roi(**kwargs, black_bg=black_bg)

            end_name = f"_desc-{method}_sphere_mask_{coord_name}.nii.gz"
            plot_filename = plot_parent_path / sphere_filename.name.replace(
                end_name,
                f"_background-{bg_color}_desc-{method}_sphere_mask_{coord_name}.png",
            )
            display.savefig(plot_filename, dpi=720)

            display.close()


def main(
    second_level_report_dir,
    analysis_dir,
    dst_dir,
    afni_img_path,
    cohort,
    task,
    analysis_type,
    seed_mask_path,
    method,
    parametric_connectivity,
    parametric_voxel_correction_p,
    parametric_cluster_correction_p,
    template_mask_path,
    template_img_path,
    sphere_radius,
):
    second_level_report_dir = Path(second_level_report_dir)
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    delete_dir(dst_dir / "cluster_results" / method)
    delete_dir(dst_dir / "cluster_masks" / method)
    delete_dir(dst_dir / "stat_plots" / method)
    delete_dir(dst_dir / "sphere_masks" / method)

    LGR.info(f"TASK: {task}, METHOD: {method}")

    first_level_glt_labels = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="get_cluster_results"
    )
    first_level_glt_label_list = list(
        itertools.product(
            first_level_glt_labels,
            get_second_level_glt_codes(cohort),
        )
    )

    appended_cluster_correction = False

    for first_level_glt_label, second_level_glt_code in first_level_glt_label_list:
        report_path = (
            second_level_report_dir
            / f"task-{task}_contrast-{first_level_glt_label}_desc-{method}_{analysis_type}_report.html"
        )

        entity_key = get_contrast_entity_key(first_level_glt_label)
        LGR.info(
            f"FIRST LEVEL GLTLABEL: {first_level_glt_label}, SECOND LEVEL GLTCODE: {second_level_glt_code}"
        )

        if method == "parametric":
            if not afni_img_path:
                LGR.warning("afni_img_path is required when method is parametric.")
                sys.exit(1)

            zcore_map_filename, group_mask_filename = get_zscore_map_and_mask(
                analysis_dir,
                afni_img_path,
                task,
                entity_key,
                first_level_glt_label,
                second_level_glt_code,
            )

            cluster_correction_table = get_cluster_correction_table(
                analysis_dir,
                task,
                entity_key,
                first_level_glt_label,
                parametric_connectivity,
            )
            cluster_size = get_cluster_size(
                cluster_correction_table,
                parametric_voxel_correction_p,
                parametric_cluster_correction_p,
            )

            if not appended_cluster_correction:
                HTMLReport.append_section(
                    report_path,
                    "sections/parametric_cluster_correction.html",
                    {
                        "parametric_voxel_correction_p": parametric_voxel_correction_p,
                        "parametric_cluster_correction_p": parametric_cluster_correction_p,
                        "parametric_connectivity": parametric_connectivity,
                        "parametric_cluster_size_threshold": cluster_size,
                    },
                )
                appended_cluster_correction = True

            thresholded_img = threshold_img(
                nib.load(zcore_map_filename),
                mask_img=nib.load(group_mask_filename),
                threshold=p_to_z(parametric_voxel_correction_p),
                cluster_threshold=cluster_size,
            )
            thresholded_filename = str(zcore_map_filename).replace(
                f"-parametric_z_map", "-parametric_cluster_corrected"
            )
            LGR.info(f"Saving thresholded image to: {thresholded_filename}")
            nib.save(thresholded_img, thresholded_filename)
        else:
            thresholded_filename = next(
                analysis_dir.rglob(
                    f"task-{task}_{entity_key}-{first_level_glt_label}"
                    f"_gltcode-{second_level_glt_code}_desc-nonparametric_thresholded_bisided.nii.gz"
                )
            )

            thresholded_img = nib.load(thresholded_filename)

        cluster_table_filename = identify_clusters(
            dst_dir,
            thresholded_img,
            analysis_type,
            seed_mask_path,
            method,
            (
                parametric_voxel_correction_p
                if method == "parametric"
                else ZERO_STAT_THRESHOLD
            ),
            cluster_size if method == "parametric" else ZERO_CLUSTER_SIZE,
            task,
            entity_key,
            first_level_glt_label,
            second_level_glt_code,
        )

        base_str = (
            f"TASK: {task}, FIRST LEVEL GLTLABEL: {first_level_glt_label} "
            f"SECOND LEVEL GLTCODE: {second_level_glt_code}"
        )
        if not cluster_table_filename.exists():
            LGR.info(f"NO SIGNIFICANT CLUSTERS FOUND FOR {base_str}")
            continue
        else:
            LGR.info(f"***SIGNIFICANT CLUSTERS FOUND FOR {base_str}***")

        plot_thresholded_img(
            dst_dir,
            thresholded_img,
            template_img_path,
            task,
            entity_key,
            first_level_glt_label,
            second_level_glt_code,
            method,
        )

        if analysis_type == "glm" and second_level_glt_code == "mean":
            create_seed_masks(
                dst_dir,
                method,
                cluster_table_filename,
                template_mask_path,
                template_img_path,
                thresholded_img,
                sphere_radius,
            )
            HTMLReport.append_section(
                report_path,
                "sections/sphere_radius.html",
                {"sphere_radius": sphere_radius},
            )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
