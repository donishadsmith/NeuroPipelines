import requests, re
from functools import lru_cache
from pathlib import Path

from _general_utils import _check_coordinate, get_template_imgs

SERVER_URL_PATH = Path(__file__).parent / "client" / "server_url.txt"
CERT_PATH = Path(__file__).parent / "client" / "cert.pem"


def get_server():
    with open(SERVER_URL_PATH) as f:
        return f.read().strip()


# Could use diskcache for long term storage but if the parcellations ever change
# then that cache would need to be manually cleared vs just restarting streamlit
# CLI path will still repeatedly send requests to server but Streamlit path
# is most likely to be used
@lru_cache(maxsize=1024)
def query_afni(x, y, z, cohort):
    request_data = {"x": x, "y": y, "z": z, "cohort": cohort}

    # Verification is overkill but good practice
    response = requests.post(f"{get_server()}/", json=request_data, verify=CERT_PATH)

    return response.json()["output"]


def run_pipeline(cohort, mni_coordinate, plot_dst_dir):
    import nibabel as nib, matplotlib.pyplot as plt
    from nilearn.plotting import plot_roi
    from matplotlib.colors import ListedColormap

    from bidsaid.masking import create_sphere_mask
    from bidsaid.files import get_entity_value

    _check_coordinate(mni_coordinate)

    # public ip is reserved but just in case it changes, add in signature
    response = query_afni(
        mni_coordinate[0], mni_coordinate[1], mni_coordinate[2], cohort
    )
    output_text = [x.strip() for x in response.split("\n")]
    target_str = "Atlas FS.afni.MNI2009c_asym: Freesurfer MNI2009c DK parcellation"

    is_focus_point = False
    if target_str in output_text:
        distance_info_index = (
            output_text.index(
                "Atlas FS.afni.MNI2009c_asym: Freesurfer MNI2009c DK parcellation"
            )
            + 1
        )
        distance_info = output_text[distance_info_index]
        if "Focus point" not in distance_info:
            sphere_radius = float(
                re.search(r"\d+(\.\d+)?", distance_info.split(":")[0]).group()
            )
        else:
            sphere_radius = 1.0
            is_focus_point = True
    else:
        sphere_radius = None

    template_mask_path, template_img_path = get_template_imgs(cohort)
    tpl = get_entity_value(template_mask_path, "tpl", return_entity_prefix=True)
    res = get_entity_value(template_mask_path, "res", return_entity_prefix=True)

    mni_coordinate = list(map(float, mni_coordinate))

    sphere_filename = None
    figure = None
    if sphere_radius:
        sphere_mask = create_sphere_mask(
            mni_coordinate, sphere_radius, nib.load(template_mask_path)
        )

        figure, ax = plt.subplots(figsize=(10, 4))
        ax.remove()

        display = plot_roi(
            sphere_mask,
            figure=figure,
            bg_img=template_img_path,
            draw_cross=False,
            cmap=ListedColormap(["red"]),
            colorbar=False,
            black_bg=False,
        )

        if plot_dst_dir:
            sphere_name = f"{tpl}_{res}_radius-{sphere_radius}mm_desc-sphere_plot_{mni_coordinate[0]}_{mni_coordinate[1]}_{mni_coordinate[2]}.png"

            sphere_filename = Path(plot_dst_dir) / sphere_name
            sphere_filename.parent.mkdir(parents=True, exist_ok=True)

            display.savefig(sphere_filename, dpi=720)

    if is_focus_point:
        sphere_radius_text = "at the focus point"
    else:
        sphere_radius_text = f"within {sphere_radius}mm" if sphere_radius else None

    return response, figure, sphere_filename, sphere_radius_text
