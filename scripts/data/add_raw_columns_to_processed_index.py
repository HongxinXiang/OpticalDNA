"""Add metadata columns from raw data to a processed VisualDNA index file.

This script is dataset-agnostic. It reads ``processed/<render-id>/index.csv``
and merges selected columns from a raw CSV or Parquet file using a shared key.

Example:
    python add_raw_columns_to_processed_index.py \
      --dataroot /path/to/opticaldna_dataset \
      --processed-dataset hg38-2048 \
      --raw-dataset hg38-2048 \
      --render-id render_w640_h640_fs14_ls1.6_hash_xxxxxxxx \
      --columns chr_name
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _parse_columns(value: str) -> list[str]:
    columns = [col.strip() for col in value.split(",") if col.strip()]
    if not columns:
        raise ValueError("At least one column must be provided through --columns.")
    return columns


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pq.read_table(path).to_pandas()
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported raw file format: {path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge selected raw metadata columns into processed/index.csv."
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Parent directory that contains dataset subdirectories.",
    )
    parser.add_argument(
        "--processed-dataset",
        type=str,
        required=True,
        help="Dataset name containing the processed index.csv.",
    )
    parser.add_argument(
        "--raw-dataset",
        type=str,
        default=None,
        help=(
            "Dataset name containing the raw file. Defaults to --processed-dataset. "
            "Use a different value when the processed dataset is a subset."
        ),
    )
    parser.add_argument(
        "--render-id",
        type=str,
        required=True,
        help="Processed render directory name under processed/.",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Optional explicit raw CSV/Parquet path. Overrides --raw-dataset.",
    )
    parser.add_argument(
        "--raw-format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Raw file format when --raw-path is not provided.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="index",
        help="Merge key shared by processed index and raw data. Default: index.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="chr_name",
        help="Comma-separated raw columns to add, for example: chr_name,source.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing values after merging. By default, missing values raise an error.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite columns that already exist in processed/index.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to overwriting processed/index.csv.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    dataroot = Path(args.dataroot).expanduser().resolve()
    raw_dataset = args.raw_dataset or args.processed_dataset
    columns = _parse_columns(args.columns)

    processed_csv_path = (
        dataroot / args.processed_dataset / "processed" / args.render_id / "index.csv"
    )
    if not processed_csv_path.exists():
        raise FileNotFoundError(f"Processed index.csv not found: {processed_csv_path}")

    if args.raw_path:
        raw_path = Path(args.raw_path).expanduser().resolve()
    else:
        raw_path = dataroot / raw_dataset / "raw" / f"{raw_dataset}.{args.raw_format}"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df_processed = pd.read_csv(processed_csv_path)
    df_raw = _read_table(raw_path)

    required_cols = [args.key] + columns
    missing_in_processed = [args.key] if args.key not in df_processed.columns else []
    missing_in_raw = [col for col in required_cols if col not in df_raw.columns]
    if missing_in_processed:
        raise KeyError(f"Missing merge key in processed index: {missing_in_processed}")
    if missing_in_raw:
        raise KeyError(f"Missing required columns in raw data: {missing_in_raw}")

    existing_cols = [col for col in columns if col in df_processed.columns]
    if existing_cols:
        if args.overwrite_existing:
            df_processed = df_processed.drop(columns=existing_cols)
        else:
            raise ValueError(
                "Columns already exist in processed index: "
                f"{existing_cols}. Use --overwrite-existing to replace them."
            )

    raw_subset = df_raw[required_cols].drop_duplicates(subset=[args.key])
    df_merged = df_processed.merge(raw_subset, on=args.key, how="left")

    if not args.allow_missing:
        missing_counts = df_merged[columns].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            raise ValueError(
                "Missing values were found after merging raw columns: "
                f"{missing_counts.to_dict()}. Use --allow-missing to keep them."
            )

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else processed_csv_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)

    print(f"Saved updated index to: {output_path}")
    print(f"Added columns: {columns}")
    print(f"Rows: {len(df_merged)}")


if __name__ == "__main__":
    main()
