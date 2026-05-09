"""Generate processed VisualDNA data from a raw CSV or Parquet dataset.

This script is dataset-agnostic. It can be used for HG38, rice, or any other
raw dataset that follows the VisualDNA directory layout:

    <dataroot>/<dataset>/raw/<dataset>.csv
or
    <dataroot>/<dataset>/raw/<dataset>.parquet

Example:
    python generate_processed.py \
      --dataroot /path/to/opticaldna_dataset \
      --dataset hg38-2048 \
      --raw-format parquet \
      --seq-columns seq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _parse_seq_columns(value: Optional[str]) -> Optional[list[str]]:
    """Parse comma-separated sequence columns.

    Use ``None`` or an empty string to let VisualDNA detect sequence columns
    automatically.
    """
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "none":
        return None
    return [col.strip() for col in value.split(",") if col.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate processed VisualDNA data for OpticalDNA pre-training."
    )
    parser.add_argument(
        "--visualdna-root",
        type=str,
        default=None,
        help=(
            "Optional path to the local VisualDNA repository. "
            "Use this only if VisualDNA is not installed as a package."
        ),
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Parent directory that contains dataset subdirectories.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name under dataroot, such as hg38-2048 or rice.",
    )
    parser.add_argument(
        "--seq-columns",
        type=str,
        default="seq",
        help=(
            "Comma-separated sequence columns. Use 'None' to enable automatic "
            "detection by VisualDNA. Example: seq or seq1,seq2."
        ),
    )
    parser.add_argument(
        "--raw-format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Raw file format used by VisualDNA.",
    )
    parser.add_argument(
        "--raw-csv-url",
        type=str,
        default=None,
        help="Optional raw CSV URL. Leave empty when raw files already exist locally.",
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Regenerate processed data even if existing files are found.",
    )
    parser.add_argument(
        "--shard-size",
        type=str,
        default="auto",
        help="Shard size passed to ShardedBuilder. Default: auto.",
    )
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=640)
    parser.add_argument("--font-size", type=int, default=14)
    parser.add_argument("--line-spacing", type=float, default=1.6)
    parser.add_argument(
        "--merge-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to merge rendered pages. Default: true.",
    )
    parser.add_argument(
        "--save-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save bounding-box annotations. Default: true.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.visualdna_root:
        sys.path.insert(0, str(Path(args.visualdna_root).expanduser().resolve()))

    from visualdna.visualdna.data import ShardedBuilder
    from visualdna.visualdna.render import BaseRenderConfig

    config = BaseRenderConfig(
        img_width=args.img_width,
        img_height=args.img_height,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        merge_pages=args.merge_pages,
        save_bbox=args.save_bbox,
    )

    ShardedBuilder(
        dataroot=args.dataroot,
        dataset=args.dataset,
        render_config=config,
        seq_columns=_parse_seq_columns(args.seq_columns),
        raw_csv_url=args.raw_csv_url,
        force_generate=args.force_generate,
        shard_size=args.shard_size,
        raw_format=args.raw_format,
    )


if __name__ == "__main__":
    main()
