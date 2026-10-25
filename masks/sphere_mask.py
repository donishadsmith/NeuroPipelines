"""
Better to use git submodule/lfs or fetch via OSF or AWS; however, this increases technical complexity
which defeats the purpose of the app and there may be firewall issues associated with downloading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nibabel as nib, numpy as np
from nilearn.plotting import plot_roi
from matplotlib.colors import ListedColormap

from bidsaid._helpers import iterable_to_str
from bidsaid.files import get_entity_value

from _general_utils import _check_coordinate, _create_sphere_mask, get_template_imgs


# https://github.com/fieldtrip/fieldtrip/blob/master/private/tal2mni.m
# Matthew Brent 2021 tal -> MNI
# https://brainmap.org/training/TalairachVersusMNI.pdf
def brett_transform(coord):
    rotn = np.array(
        [
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.9988, 0.0500, 0.0000],
            [0.0000, -0.0500, 0.9988, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    upz = np.array(
        [
            [0.9900, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.9700, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.9200, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    downz = np.array(
        [
            [0.9900, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.9700, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.8400, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    coord = np.array(coord + [1.0])
    coord = np.dot(np.linalg.inv(rotn), coord)

    # Matrix used depends on if its below or above
    # the AC
    if coord[2] < 0:
        coord = np.dot(np.linalg.inv(downz), coord)
    else:
        coord = np.dot(np.linalg.inv(upz), coord)

    return np.round(coord[:3]).tolist()


# Affine taken from nimare
# https://pmc.ncbi.nlm.nih.gov/articles/PMC6871323/
def lancaster_transform(coord):
    icbm_other_affine = np.array(
        [
            [0.9357, 0.0029, -0.0072, -1.0423],
            [-0.0065, 0.9396, -0.0726, -1.3940],
            [0.0103, 0.0752, 0.8967, 3.6475],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    coord = np.dot(np.array(coord + [1]), np.linalg.inv(icbm_other_affine.T))[:3]

    return np.round(coord[:3]).tolist()


transform_dict = {"Brett": brett_transform, "Lancaster": lancaster_transform}


def create_transform_text(
    original_coordinate_space, original_coordinate, coordinate, transform_method
):
    if original_coordinate_space == "MNI":
        return None

    return (
        f"Transformed the original Talairach coordinates ({iterable_to_str(original_coordinate)}) to MNI space "
        f"({iterable_to_str(coordinate)}) using the {transform_method} method"
    )


def run_pipeline(
    dst_dir,
    cohort,
    coordinate,
    sphere_radius,
    original_coordinate_space,
    transform_method,
    use_black_bg,
):
    _check_coordinate(coordinate)

    coordinate = list(map(float, coordinate))

    template_mask_path, template_img_path = get_template_imgs(cohort)

    template_mask = nib.load(template_mask_path)

    original_coordinate = coordinate
    if original_coordinate_space != "MNI":
        coordinate = transform_dict[transform_method](coordinate)

    coord_name = "_".join([str(x) for x in coordinate])
    tpl = get_entity_value(template_mask_path, "tpl", return_entity_prefix=True)
    res = get_entity_value(template_mask_path, "res", return_entity_prefix=True)
    sphere_name = (
        f"{tpl}_{res}_radius-{sphere_radius}mm_desc-sphere_mask_{coord_name}.nii.gz"
    )
    sphere_filename = (Path(dst_dir) if dst_dir else Path().home()) / sphere_name
    sphere_filename.parent.mkdir(parents=True, exist_ok=True)

    sphere_mask = _create_sphere_mask(coordinate, sphere_radius, template_mask)
    nib.save(sphere_mask, sphere_filename)

    display = plot_roi(
        sphere_filename,
        bg_img=template_img_path,
        draw_cross=False,
        cmap=ListedColormap(["red"]),
        colorbar=False,
        black_bg=use_black_bg,
    )

    plot_filename = sphere_filename.parent / sphere_filename.name.replace(
        ".nii.gz", ".png"
    )
    display.savefig(plot_filename, dpi=720)

    return (
        sphere_filename,
        plot_filename,
        create_transform_text(
            original_coordinate_space, original_coordinate, coordinate, transform_method
        ),
    )
