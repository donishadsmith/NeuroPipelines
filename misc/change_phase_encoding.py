import argparse, json

from bids import BIDSLayout

from bidsaid.metadata import get_img_orientation, direction_to_voxel_axis


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Change phase encoding direction.")
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="The root of the BIDS directory.",
    )
    parser.add_argument(
        "--phase_encoding_axis",
        dest="phase_encoding_axis",
        default=("A", "P"),
        required=False,
        nargs=2,
        help="The axis. Note that two letters need to be passed.",
    )
    parser.add_argument(
        "--fat_shift_direction",
        dest="fat_shift_direction",
        default="P",
        required=False,
        help="The direction of the fat shift.",
    )

    return parser


def main(bids_dir, phase_encoding_axis, fat_shift_direction):
    layout = BIDSLayout(bids_dir)
    nifti_files = layout.get(
        return_type="file",
        target="subject",
        suffix="bold",
        extension=".nii.gz",
        scope="raw",
    )

    for nifti_file in nifti_files:
        _, orient = get_img_orientation(nifti_file)
        phase_axis, phase_index = direction_to_voxel_axis(
            nifti_file, phase_encoding_axis
        )

        json_file = str(nifti_file).replace(".nii.gz", ".json")
        with open(json_file, "r") as f:
            json_data = json.load(f)

        with open(json_file, "w") as f:
            json_data["PhaseEncodingDirection"] = (
                phase_axis
                if orient[phase_index] != fat_shift_direction
                else f"{phase_axis}-"
            )
            json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
