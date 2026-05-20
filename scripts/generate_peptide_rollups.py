#!/usr/bin/env python3
"""Generate peptide rollup / normalization parquet files from a Skyline PRISM report.

Given a folder containing ``PRISM.parquet`` (a Skyline "PRISM" report exported
as parquet at transition level), this runs the skyline-prism transition ->
peptide rollup and peptide-level normalization, writing:

    unnormalized_polished_peptides.parquet   (rollup: median_polish)
    unnormalized_summed_peptides.parquet     (rollup: sum)
    median_normalized_peptides.parquet       (median_polish + median normalization)
    rtloess_normalized_peptides.parquet      (median_polish + RT-lowess normalization)

The first two are the *un-normalized* peptide quantities (the pipeline's
``peptides_rollup.parquet``), isolating the transition -> peptide aggregation.
The last two apply the pipeline's Stage 2b peptide normalization to the
median-polish rollup, isolating each normalization method (no batch correction).

All files are in LOG2 scale, matching the pipeline's intermediate peptide
matrices. They are used by Skyline's ``MedianPolishScenariosTest`` to verify
that Skyline computes the same peptide quantities as skyline-prism.

The merge, column auto-detection, rollup, and normalization reuse the same
skyline-prism functions that ``prism run`` uses, so the output matches what the
full pipeline produces.

Usage:
    python scripts/generate_peptide_rollups.py PATH/TO/FOLDER
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Allow running directly (e.g. `python scripts/generate_peptide_rollups.py`)
# without installing the package, by putting the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyline_prism.chunked_processing import ChunkedRollupConfig, rollup_transitions_sorted
from skyline_prism.cli import find_column
from skyline_prism.data_io import merge_and_sort_streaming
from skyline_prism.normalization import apply_rt_lowess_normalization

INPUT_FILENAME = "PRISM.parquet"

# Rollup method -> un-normalized output filename.
ROLLUP_OUTPUTS = {
    "median_polish": "unnormalized_polished_peptides.parquet",
    "sum": "unnormalized_summed_peptides.parquet",
}

# Output filenames for the peptide-level normalizations applied to the
# median-polish rollup.
MEDIAN_NORMALIZED_OUTPUT = "median_normalized_peptides.parquet"
RTLOESS_NORMALIZED_OUTPUT = "rtloess_normalized_peptides.parquet"

# Metadata (non-sample) columns in a rollup parquet.
META_COLS = ("PeptideModifiedSequenceUnimodIds", "n_transitions", "mean_rt")


def _merge(prism_path: Path, work_dir: Path) -> tuple[Path, dict[str, str]]:
    """Run Stage 1 (merge + sort) and auto-detect column names.

    Returns the merged parquet path and the detected column-name mapping used
    to configure the rollup. The batch name is the input file stem ("PRISM"),
    so sample columns are named ``<replicate>__@__PRISM``.
    """
    merged_path = work_dir / "merged_data.parquet"
    merge_and_sort_streaming(
        [prism_path],
        merged_path,
        sort_column=None,  # auto-detect peptide column
        batch_names=[prism_path.stem],
    )

    available = set(pq.ParquetFile(merged_path).schema_arrow.names)
    cols = {
        "peptide_col": find_column(
            available,
            "Peptide Modified Sequence Unimod Ids",
            "Peptide Modified Sequence",
            "Peptide",
        ),
        "sample_col": find_column(available, "Sample ID"),
        "abundance_col": find_column(available, "Area"),
        "transition_col": find_column(available, "Fragment Ion"),
        "precursor_charge_col": find_column(available, "Precursor Charge") or "Precursor Charge",
        "product_charge_col": find_column(available, "Product Charge") or "Product Charge",
        "shape_corr_col": find_column(available, "Shape Correlation") or "Shape Correlation",
        "rt_col": find_column(available, "Retention Time") or "Retention Time",
        "mz_col": find_column(available, "Product Mz") or "Product Mz",
    }
    if cols["peptide_col"] is None:
        raise SystemExit(f"No peptide column found in {prism_path}")
    return merged_path, cols


def _rollup(merged_path: Path, cols: dict[str, str], method: str, out_path: Path) -> pd.DataFrame:
    """Run Stage 2 (transition -> peptide rollup) for a single method.

    Defaults mirror ``prism run``: min_transitions=3, use_ms1=False
    (=> exclude_precursor=True), log_transform=True. Output is in log2 scale,
    matching the pipeline's intermediate ``peptides_rollup.parquet``. Returns the
    rolled-up peptide DataFrame.
    """
    config = ChunkedRollupConfig(
        method=method,
        min_transitions=3,
        log_transform=True,
        exclude_precursor=True,
        **cols,
    )
    rollup_out = merged_path.parent / f"rollup_{method}.parquet"
    rollup_transitions_sorted(
        parquet_path=merged_path,
        output_path=rollup_out,
        config=config,
        save_residuals=False,
        pre_sorted=True,  # merge_and_sort_streaming already sorted by peptide
    )
    shutil.copyfile(rollup_out, out_path)
    df = pq.read_table(out_path).to_pandas()
    print(f"  {method:>13} -> {out_path.name}  ({len(df)} peptides)")
    return df


def _sample_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _median_normalize(peptide_df: pd.DataFrame) -> pd.DataFrame:
    """Apply global median normalization, mirroring ``prism run`` Stage 2b.

    Shifts each sample by (sample median - global median) on the log2 peptide
    abundances, so every sample ends up with the same median.
    """
    result = peptide_df.copy()
    sample_cols = _sample_columns(result)
    sample_medians = result[sample_cols].median()
    global_median = sample_medians.median()
    norm_factors = sample_medians - global_median
    for col in sample_cols:
        result[col] = result[col] - norm_factors[col]
    return result


def _rt_lowess_normalize(peptide_df: pd.DataFrame) -> pd.DataFrame:
    """Apply RT-lowess normalization, mirroring ``prism run`` Stage 2b.

    Aligns each sample's lowess(log2 abundance vs mean RT) curve to the global
    median curve. Uses the same defaults as the CLI (frac=0.3, 100 grid points).
    """
    result = apply_rt_lowess_normalization(
        peptide_df,
        _sample_columns(peptide_df),
        rt_col="mean_rt",
        frac=0.3,
        n_grid_points=100,
    )
    return result.normalized_df


def _write(df: pd.DataFrame, out_path: Path, label: str) -> None:
    df.to_parquet(out_path, index=False)
    print(f"  {label:>13} -> {out_path.name}  ({len(df)} peptides)")


def generate(folder: Path) -> None:
    prism_path = folder / INPUT_FILENAME
    if not prism_path.exists():
        raise SystemExit(f"{prism_path} not found")

    print(f"Generating peptide rollups for {folder}")
    work_dir = Path(tempfile.mkdtemp(prefix="prism_rollups_"))
    try:
        merged_path, cols = _merge(prism_path, work_dir)

        # Un-normalized rollups (median polish + sum).
        polished_df = _rollup(merged_path, cols, "median_polish",
                              folder / ROLLUP_OUTPUTS["median_polish"])
        _rollup(merged_path, cols, "sum", folder / ROLLUP_OUTPUTS["sum"])

        # Peptide-level normalizations applied to the median-polish rollup.
        _write(_median_normalize(polished_df), folder / MEDIAN_NORMALIZED_OUTPUT, "median")
        _write(_rt_lowess_normalize(polished_df), folder / RTLOESS_NORMALIZED_OUTPUT, "rt_lowess")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "folder",
        type=Path,
        help=f"Folder containing {INPUT_FILENAME}; output parquet files are written here too.",
    )
    args = parser.parse_args(argv)
    generate(args.folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
