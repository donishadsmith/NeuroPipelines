import argparse

from whereami import run_pipeline


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Pipeline for looking up MNI coordinates with AFNI's whereami."
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["kids", "adults"],
        help="The name of the cohort.",
    )
    parser.add_argument(
        "--mni_coordinate",
        dest="mni_coordinate",
        required=True,
        nargs=3,
        help="The MNI coordinate.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="plot_dst_dir",
        required=False,
        default=None,
        help=(
            "The root of the output directory for the plot. "
            "Nothing is saved if no location is given."
        ),
    )

    return parser


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    output_text, output = run_pipeline(**vars(args))

    if output:
        for _, content in output.items():
            print(f"File save to {content.get('afni_sphere_filename')}")

    print(output_text)
