import requests, re
from functools import lru_cache
from pathlib import Path

from bidsaid.parsers import _is_float

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
TARGETS_DICT = {
    "FS.afni.MNI2009c_asym": {
        "name": "Freesurfer MNI2009c DK parcellation",
        "target": "Atlas FS.afni.MNI2009c_asym: Freesurfer MNI2009c DK parcellation",
    },
    "Brodmann_Pijn_AFNI": {
        "name": "Brodmann atlas for MNI 2009c - Pijnenburg AFNI version",
        "target": "Atlas Brodmann_Pijn_AFNI: Brodmann atlas for MNI 2009c - Pijnenburg AFNI version",
    },
}
SERVER_URL_PATH = Path(__file__).parent / "client" / "server_url.txt"
CERT_PATH = Path(__file__).parent / "client" / "cert.pem"


def _check_coordinate(coordinate):
    if not all([_is_float(x) for x in coordinate]):
        coordinate = list(map(str, coordinate))
        raise ValueError(f"Invalid MNI coordinate: {','.join(coordinate)}")


def get_template_imgs(cohort):
    return (
        COHORT_MAP[cohort]["template_mask_path"],
        COHORT_MAP[cohort]["template_img_path"],
    )


def get_server():
    with open(SERVER_URL_PATH) as f:
        return f.read().strip()


# Could use diskcache for long term storage but if the parcellations ever change
# then that cache would need to be manually cleared vs just restarting streamlit
# CLI path will still repeatedly send requests to server but Streamlit path
# is most likely to be used
@lru_cache(maxsize=1024)
def query_afni(x, y, z):
    request_data = {"x": x, "y": y, "z": z}

    # Verification is overkill but good practice
    response = requests.post(
        f"{get_server()}/", json=request_data, verify=CERT_PATH, timeout=30
    )
    response.raise_for_status()

    return response.json()["output"]


def identify_target(output_text):
    new_dict = {}
    for atlas, content in TARGETS_DICT.items():
        sphere_radius = None
        is_focus_point = False
        target_str = content["target"]

        if target_str in output_text:
            distance_info_index = output_text.index(target_str) + 1
            distance_info = output_text[distance_info_index]

            if "Focus point" not in distance_info:
                sphere_radius = float(
                    re.search(r"\d+(\.\d+)?", distance_info.split(":")[0]).group()
                )
            else:
                sphere_radius = 1.0
                is_focus_point = True

        new_dict[atlas] = dict(content)
        new_dict[atlas].update(
            {"sphere_radius": sphere_radius, "is_focus_point": is_focus_point}
        )

    return new_dict


def run_pipeline(cohort, mni_coordinate, plot_dst_dir):
    import nibabel as nib, matplotlib.pyplot as plt
    from nilearn.plotting import plot_roi
    from matplotlib.colors import ListedColormap

    from bidsaid.masking import create_sphere_mask
    from bidsaid.files import get_entity_value

    _check_coordinate(mni_coordinate)

    # public ip is reserved but just in case it changes, add in signature
    response = query_afni(mni_coordinate[0], mni_coordinate[1], mni_coordinate[2])
    output_text = [x.strip() for x in response.split("\n") if len(x) > 0]
    target_dict = identify_target(output_text)

    template_mask_path, template_img_path = get_template_imgs(cohort)
    tpl = get_entity_value(template_mask_path, "tpl", return_entity_prefix=True)
    res = get_entity_value(template_mask_path, "res", return_entity_prefix=True)

    mni_coordinate = list(map(float, mni_coordinate))

    output = {}
    for atlas, content in target_dict.items():
        sphere_radius = content["sphere_radius"]
        output.update({atlas: content})
        sphere_filename = None
        figure = None
        is_focus_point = content["is_focus_point"]

        if sphere_radius:
            if is_focus_point:
                sphere_radius_text = "at the focus point"
            else:
                sphere_radius_text = (
                    f"within {sphere_radius:g}mm" if sphere_radius else None
                )

            sphere_mask = create_sphere_mask(
                mni_coordinate, sphere_radius, nib.load(template_mask_path)
            )

            figure, ax = plt.subplots(figsize=(12, 6))
            ax.remove()

            display = plot_roi(
                sphere_mask,
                figure=figure,
                axes=(0, 0, 1, 0.86),
                bg_img=template_img_path,
                draw_cross=False,
                cmap=ListedColormap(["red"]),
                colorbar=False,
                black_bg=False,
            )

            figure.suptitle(
                f"MNI coordinate ({mni_coordinate[0]:g}, {mni_coordinate[1]:g}, {mni_coordinate[2]:g}): "
                f"nearest label {sphere_radius_text}\n{content['name']}",
                fontsize=18,
            )

            if plot_dst_dir:
                sphere_name = f"{tpl}_{res}_radius-{sphere_radius}mm_atlas-{atlas}_desc-sphere_plot_{mni_coordinate[0]}_{mni_coordinate[1]}_{mni_coordinate[2]}.png"

                sphere_filename = Path(plot_dst_dir) / sphere_name
                sphere_filename.parent.mkdir(parents=True, exist_ok=True)

                display.savefig(sphere_filename, dpi=720)

            output[atlas].update(
                {
                    "afni_sphere_filename": sphere_filename,
                    "afni_sphere_figure": figure,
                }
            )

    return response, output
