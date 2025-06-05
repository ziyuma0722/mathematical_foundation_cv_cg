from argparse import ArgumentParser
from graph_cut_controller import GraphCutController


def main():
    parser = ArgumentParser(description="Interactive Segmentation with Graph Cut")
    parser.add_argument(
        "--autoload",
        required=False,
        choices=["batman", "VanDamme"],
        help="Automatically load the selected image, load scribbles, and segment",
    )
    args = parser.parse_args()

    GraphCutController(args)


if __name__ == "__main__":
    main()
