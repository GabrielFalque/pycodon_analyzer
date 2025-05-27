# src/pycodon_analyzer/cli.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Command-Line Interface for the Codon Usage Analyzer.

Analyzes codon usage and sequence properties from gene alignment
FASTA files located within a specified directory.
Performs analysis per gene and combines results for global analysis
and plotting, including an analysis of concatenated sequences per genome ID.
"""

import argparse
import os
import sys
import glob # To find files matching a pattern
import re   # To extract gene name from filename
import traceback # For detailed error printing during development/debugging
import logging # <-- Import logging module
from typing import List, Dict, Optional, Tuple, Set, Any, Counter, TYPE_CHECKING # <-- Import typing helpers
import seaborn as sns
from pathlib import Path
import csv
import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn # type: ignore # Import Progress components
    from rich.logging import RichHandler # type: ignore
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback using standard logging and no progress bar
    def track(sequence, description="", total=None):
        logging.info(f"{description} (running)...")
        return sequence
    RichHandler = logging.StreamHandler
    logging.warning("WARNING: 'rich' library not found. Progress bar disabled, logging will be basic. Install with 'pip install rich'")
    # Define Progress dummy for type hinting
    Progress = Any
    SpinnerColumn = Any
    BarColumn = Any
    TextColumn = Any
    TimeElapsedColumn = Any
    TimeRemainingColumn = Any
try:
    from scipy import stats as scipy_stats # Import for direct use
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None
    SCIPY_AVAILABLE = False
    # Warning logged later if stats needed but missing

# Import prince for CA - handle optional dependency for typing
if TYPE_CHECKING:
    import prince # Import only for type checking
    PrinceCA = prince.CA # Actual type for type checker
else:
    PrinceCA = Any # Fallback type for runtime if prince not installed
    try:
        import prince # Try to import at runtime for actual use
    except ImportError:
        prince = None # Set to None if not found

# Import necessary functions from local modules
# Assuming type checkers can find these or using forward references if needed
from . import io
from . import analysis
from . import plotting
from . import utils
from . import extraction
from . import reporting
from .utils import load_reference_usage, get_genetic_code, clean_and_filter_sequences
from .analysis import PrinceCA, FullAnalysisResultType

# Import modules for parallelization
try:
    import multiprocessing as mp
    from functools import partial
    MP_AVAILABLE = True
except ImportError:
    mp = None
    partial = None
    MP_AVAILABLE = False
    # Warning will be logged later if parallel execution is requested but unavailable


# --- Configure logging ---
# Get a logger specific to this application
logger = logging.getLogger("pycodon_analyzer")


# --- Find default reference path (Unchanged logic) ---
try:
    from importlib.resources import files as pkg_resources_files
except ImportError:
    try:
        from importlib_resources import files as pkg_resources_files # type: ignore
    except ImportError:
        pkg_resources_files = None

DEFAULT_REF_FILENAME = "human_codon_usage.csv"
DEFAULT_HUMAN_REF_PATH: Optional[str] = None

if pkg_resources_files:
    try:
        ref_path_obj = pkg_resources_files('pycodon_analyzer').joinpath('data').joinpath(DEFAULT_REF_FILENAME)
        if ref_path_obj.is_file():
             DEFAULT_HUMAN_REF_PATH = str(ref_path_obj)
    except Exception:
         pass # Errors finding the path are handled later if needed
    
# --- Type Alias for clarity ---
AnalyzeGeneResultType = Tuple[
    Optional[str],                  # gene_name
    str,                            # status ("success", "empty file", "no valid seqs", "exception")
    Optional[pd.DataFrame],         # per_sequence_df
    Optional[pd.DataFrame],         # ca_input_df (RSCU per sequence for this gene)
    Optional[Dict[str, float]],     # nucleotide_frequencies
    Optional[Dict[str, float]],     # dinucleotide_frequencies
    Optional[Dict[str, Seq]]        # cleaned_sequences_map {original_id: Bio.Seq}
]

ProcessGeneFileResultType = Tuple[
    Optional[str],                          # gene_name
    str,                                    # status
    Optional[pd.DataFrame],                 # per_sequence_df_gene (metrics)
    Optional[pd.DataFrame],                 # ca_input_df_gene (RSCU wide for this gene)
    Optional[Dict[str, float]],             # nucl_freqs_gene (aggregate for this gene)
    Optional[Dict[str, float]],             # dinucl_freqs_gene (aggregate for this gene)
    Optional[Dict[str, Seq]],               # cleaned_seq_map {original_id: Bio.Seq}
    Optional[Dict[str, Dict[str, float]]],  # per_sequence_nucl_freqs (NEW)
    Optional[Dict[str, Dict[str, float]]]   # per_sequence_dinucl_freqs (NEW)
]

# ADDED to cli.py near the top, or within a utility section if you prefer

def _ensure_output_subdirectories(output_dir_path: Path) -> None:
    """
    Creates the 'data' and 'images' subdirectories in the main output path if they don't exist.
    The 'html' subdirectory for the report will be created by HTMLReportGenerator.
    """
    try:
        (output_dir_path / "data").mkdir(parents=True, exist_ok=True)
        (output_dir_path / "images").mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured 'data' and 'images' subdirectories exist in '{output_dir_path}'.")
    except OSError as e:
        logger.error(f"Error creating output subdirectories 'data' or 'images' in '{output_dir_path}': {e}. Exiting.")
        sys.exit(1)

def _create_output_readme(output_dir_path: Path, args: argparse.Namespace) -> None:
    """Creates a README.txt file in the output directory explaining the structure."""

    # Prepare dynamic parts for the README content
    # Using f-strings directly in the text block for args that are always present
    metadata_file_info = f"  - Metadata File: {args.metadata if args.metadata else 'Not provided'}"
    metadata_id_col_info = ""
    if args.metadata:
        metadata_id_col_info = f"\n  - Metadata ID Column: {args.metadata_id_col}"
    color_by_metadata_info = ""
    if args.color_by_metadata:
        color_by_metadata_info = f"\n  - Plots Colored by Metadata Column: {args.color_by_metadata}"
    html_report_info = f"  - HTML Report Generated: {'No' if args.no_html_report else 'Yes'}"

    readme_content = f"""========================================
PyCodon Analyzer - Output Directory
========================================

This directory contains the results generated by the PyCodon Analyzer `analyze` command.

----------------------------------------
Run Parameters Summary
----------------------------------------
The analysis was run with the following key parameters:
  - Input Directory: {args.directory}
  - Output Directory: {args.output}
  - Genetic Code ID: {args.genetic_code}
  - Reference Usage File: {args.reference_usage_file if args.reference_usage_file else 'None'}
  - Max Ambiguity: {args.max_ambiguity}%
  - Threads: {args.threads}
  - Plots Skipped: {'Yes' if args.skip_plots else 'No'}
  - CA Skipped: {'Yes' if args.skip_ca else 'No'}
{metadata_file_info}{metadata_id_col_info}{color_by_metadata_info}
{html_report_info}

----------------------------------------
Directory Structure and File Descriptions
----------------------------------------

The output is organized into the following main components:

1.  `report.html`
    This is the main interactive HTML report (if generated, i.e., if --no-html-report
    was not specified). It serves as the primary entry point for exploring your
    analysis results. It provides an overview, run parameters, and navigation
    to all detailed sections.

2.  `data/` (Subdirectory)
    This folder contains all data tables generated during the analysis, primarily
    in CSV (Comma Separated Values) format. These files can be opened with
    spreadsheet software (like Excel, LibreOffice Calc) or loaded into data
    analysis environments (e.g., Python with Pandas, R).

    Key files typically include:
    * `per_sequence_metrics_all_genes.csv`:
        Comprehensive metrics for each individual sequence that passed the
        cleaning and filtering stages. If metadata was provided, it will be
        merged into this table.
    * `mean_features_per_gene.csv`:
        A summary table showing the average values for key codon usage and
        sequence property metrics (e.g., GC, GC1-3, ENC, CAI, RCDI, GRAVY,
        Aromaticity), aggregated for each gene and for the "complete"
        concatenated set (if applicable).
    * `gene_comparison_stats.csv`:
        (If at least two gene groups were analyzed) Results from statistical
        tests (e.g., Kruskal-Wallis H-test) performed to compare the
        distributions of key features across different gene groups.
    * `per_sequence_rscu_wide.csv`:
        (If CA was not skipped and data was available) Relative Synonymous
        Codon Usage (RSCU) values for all codons across all analyzed
        sequences. Each row represents a sequence, and each column a codon.
        This table is used as input for the combined Correspondence Analysis (CA).
    * `ca_row_coordinates.csv`:
        (If CA was performed) Row (sequence) coordinates derived from the
        combined Correspondence Analysis.
    * `ca_col_coordinates.csv`:
        (If CA was performed) Column (codon) coordinates derived from the
        combined Correspondence Analysis.
    * `ca_col_contributions.csv`:
        (If CA was performed) The contribution of each codon to the inertia
        of the dimensions in the combined Correspondence Analysis.
    * `ca_eigenvalues.csv`:
        (If CA was performed) Eigenvalues, percentage of variance, and
        cumulative variance explained by each dimension of the combined
        Correspondence Analysis.
    * `gene_sequence_summary.csv`:
        (If HTML report generated and data available) A summary table providing
        sequence counts (number of valid sequences) and basic length statistics
        (mean, min, max) for each gene analyzed, including the "complete" set
        if applicable.

    Note: The presence of CA-related files depends on whether CA was run
          (not skipped and sufficient data was available).

3.  `images/` (Subdirectory)
    This folder contains all plot images generated by the analysis (if
    `--skip_plots` was not specified). The format of these images (e.g., SVG,
    PNG) depends on the `--plot_formats` option specified during the run.

    * Combined Plots:
        Plots summarizing trends across all genes or all sequences are
        typically found directly within the `images/` directory. Examples:
        `gc_means_barplot_by_Gene.<fmt>`,
        `enc_vs_gc3_plot_grouped_by_gene.<fmt>`, combined CA plots,
        correlation heatmaps.
    * Per-Gene RSCU Boxplots:
        Individual RSCU boxplots for each analyzed gene (and the "complete"
        set), e.g., `RSCU_boxplot_GENENAME.<fmt>`.
    * Metadata-Specific Plots (if `--color_by_metadata` was used):
        These plots are organized into further subdirectories:
        `images/<METADATA_COLUMN_NAME>_per_gene_plots/`
        This directory will contain one subdirectory for each gene (and the
        "complete" set), e.g.,
        `images/<METADATA_COLUMN_NAME>_per_gene_plots/<GENE_NAME>/`.
        Inside each gene-specific directory, you will find plots like
        ENC vs GC3, Neutrality plots, per-gene CA biplots, and per-gene
        dinucleotide abundance plots, all colored or grouped by the
        categories from the specified metadata column.

4.  `html/` (Subdirectory)
    (If HTML report was generated) This folder contains all the secondary HTML
    pages that make up the interactive report. The main `report.html` (located
    in the root of this output directory) links to these pages. Examples
    include `sequence_metrics.html`, `gene_aggregates.html`, etc.

----------------------------------------
How to Navigate the Results
----------------------------------------

1.  Start with `report.html` (if generated):
    Open this file in a web browser. It provides a summary and a navigation
    menu to access all detailed sections of the analysis.

2.  Explore Data Tables:
    The CSV files in the `data/` directory can be opened with spreadsheet
    software for detailed inspection or used for further custom analyses.

3.  View Plots:
    The images in the `images/` directory (if generated) provide visual
    summaries of various analyses. They are also embedded within the HTML report.

---
This output was generated by PyCodon Analyzer on {pd.Timestamp("now").strftime('%Y-%m-%d %H:%M:%S')}.
"""
    try:
        # MODIFIED: Save as README.txt
        readme_path = output_dir_path / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        logger.info(f"Output directory README.txt created at: {readme_path}")
    except IOError as e:
        logger.error(f"Failed to write output README.txt: {e}")


# --- Helper function to extract gene name from filename (with type hints) ---
def extract_gene_name_from_file(filename: str) -> Optional[str]:
    """
    Extracts gene name from filenames like 'gene_XYZ.fasta' or 'gene_ABC.fa'.
    Returns the extracted name (e.g., 'XYZ') or None if pattern doesn't match.

    Args:
        filename (str): The full path or basename of the file.

    Returns:
        Optional[str]: The extracted gene name or None.
    """
    base = os.path.basename(filename)
    # Match pattern 'gene_' followed by name, ending with common FASTA extensions
    match = re.match(r'gene_([\w\-.]+)\.(fasta|fa|fna|fas|faa)$', base, re.IGNORECASE)
    if match:
        return match.group(1) # Return the captured gene name part
    else:
        # No warning logged here by default, handled by caller if needed
        return None
    
def _setup_output_directory(output_path_str: str) -> Path:
    """Creates the output directory if it doesn't exist."""
    output_dir_path = Path(output_path_str)
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{output_dir_path}' created or already exists.")
        return output_dir_path
    except OSError as e:
        logger.error(f"Error creating output directory '{output_dir_path}': {e}. Exiting.")
        sys.exit(1)

def _load_reference_data(args: argparse.Namespace) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """Loads reference codon usage data based on command-line arguments."""
    reference_weights: Optional[Dict[str, float]] = None
    reference_data_for_plot: Optional[pd.DataFrame] = None

    if args.reference_usage_file and args.reference_usage_file.lower() != 'none':
        ref_path_to_load: Optional[str] = None
        if args.reference_usage_file.lower() == 'human':
            ref_path_to_load = DEFAULT_HUMAN_REF_PATH
            if not ref_path_to_load or not os.path.isfile(ref_path_to_load):
                logger.error("Default human reference file ('human') requested but not found. Exiting.")
                sys.exit(1)
        elif os.path.isfile(args.reference_usage_file):
            ref_path_to_load = args.reference_usage_file
        else:
            logger.error(f"Specified reference file not found: {args.reference_usage_file}. Exiting.")
            sys.exit(1)

        logger.info(f"Loading codon usage reference table: {ref_path_to_load}...")
        try:
            current_genetic_code = utils.get_genetic_code(args.genetic_code)
            # Assuming load_reference_usage now takes an optional delimiter from args
            ref_delimiter = getattr(args, 'ref_delimiter', None)
            reference_data_for_plot = utils.load_reference_usage(
                ref_path_to_load, current_genetic_code, args.genetic_code, delimiter=ref_delimiter
            )
            if reference_data_for_plot is not None and 'Weight' in reference_data_for_plot.columns:
                reference_weights = reference_data_for_plot['Weight'].to_dict()
                logger.info("Reference data loaded and weights extracted.")
            elif reference_data_for_plot is not None:
                logger.error("'Weight' column missing in loaded reference data. Exiting.")
                sys.exit(1)
            else:
                logger.error(f"Failed to load or process reference data from {ref_path_to_load}. Exiting.")
                sys.exit(1)
        except Exception as e:
            logger.exception(f"Error loading reference file {ref_path_to_load}: {e}. Exiting.")
            sys.exit(1)
    else:
        logger.info("No reference file specified. Reference-based metrics will be NaN.")
    return reference_weights, reference_data_for_plot

def _load_metadata(
    metadata_path: Optional[Path],
    id_col_name: str,
    delimiter: Optional[str]
) -> Optional[pd.DataFrame]:
    """
    Loads and validates the metadata file (CSV or TSV).

    Args:
        metadata_path (Optional[Path]): Path to the metadata file.
        id_col_name (str): Name of the column containing sequence identifiers.
        delimiter (Optional[str]): Specified delimiter for the file. If None,
                                   attempts to auto-detect.

    Returns:
        Optional[pd.DataFrame]: DataFrame with metadata, indexed by the id_col_name,
                                or None if loading/validation fails.
    """
    if not metadata_path:
        logger.debug("No metadata file path provided.")
        return None

    if not metadata_path.is_file():
        logger.error(f"Metadata file not found: {metadata_path}. \
                     Skipping metadata integration.")
        return None

    meta_df: Optional[pd.DataFrame] = None
    used_delimiter: Optional[str] = delimiter
    file_basename = metadata_path.name # For logging

    logger.info(f"Loading metadata from: {file_basename}")

    try:
        if used_delimiter:
            logger.debug(f"Attempting to read metadata file '{file_basename}' \
                         with specified delimiter: '{used_delimiter}'")
            # Ensure id_col_name is read as string to prevent type issues during merge
            meta_df = pd.read_csv(metadata_path, sep=used_delimiter, comment='#', dtype={id_col_name: str})
        else:
            logger.debug(f"Attempting to auto-detect delimiter for metadata file '{file_basename}'...")
            try:
                with open(metadata_path, 'r', newline='', encoding='utf-8') as csvfile: # Added encoding
                    sample = csvfile.read(2048) # Read a sample for sniffing
                    if not sample.strip():
                        logger.error(f"Metadata file '{file_basename}' appears to be empty or contains only whitespace.")
                        return None
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=',\t;|') # Common delimiters
                    used_delimiter = dialect.delimiter
                    logger.info(f"Sniffed delimiter '{used_delimiter}' for metadata file '{file_basename}'.")
                    meta_df = pd.read_csv(metadata_path, sep=used_delimiter, comment='#', dtype={id_col_name: str})
            except (csv.Error, pd.errors.ParserError, UnicodeDecodeError) as sniff_err:
                logger.warning(f"Could not reliably sniff delimiter or parse metadata file '{file_basename}' with sniffed delimiter: {sniff_err}. "
                               "Falling back to trying common delimiters.")
                fallback_delimiters = ['\t', ',', ';']
                for delim_try in fallback_delimiters:
                    logger.debug(f"Fallback: Trying delimiter '{delim_try}' for '{file_basename}'.")
                    try:
                        meta_df = pd.read_csv(metadata_path, sep=delim_try, comment='#', dtype={id_col_name: str})
                        # A simple check if parsing was reasonable (e.g., more than one column if expected, or header matches)
                        if meta_df.shape[1] > 0 : # or some other heuristic
                            logger.info(f"Successfully read metadata file '{file_basename}' with fallback delimiter: '{delim_try}'.")
                            used_delimiter = delim_try
                            break # Success
                        meta_df = None # Reset if parse was not good
                    except Exception: # Catch errors during fallback attempts silently
                        meta_df = None
                if meta_df is None:
                    logger.error(f"Failed to parse metadata file '{file_basename}' with any common fallback delimiter after sniffing failed.")
                    return None
            except FileNotFoundError: # Should be caught by initial check, but defensive
                logger.error(f"Metadata file not found during read attempt: {metadata_path}") # Should not happen
                return None
            except Exception as e_sniff_or_read:
                logger.exception(f"Unexpected error reading metadata file '{file_basename}' (delimiter: {used_delimiter}): {e_sniff_or_read}")
                return None

        if meta_df is None or meta_df.empty: # Check after all attempts
            logger.error(f"Metadata file '{file_basename}' could not be loaded or is empty.")
            return None

        # --- Validate DataFrame content ---
        if id_col_name not in meta_df.columns:
            logger.error(f"Metadata ID column '{id_col_name}' not found in '{file_basename}'. "
                         f"Available columns: {meta_df.columns.tolist()}. Skipping metadata integration.")
            return None

        # Ensure ID column is string type for reliable merging and drop rows where ID is NaN
        meta_df[id_col_name] = meta_df[id_col_name].astype(str)
        meta_df.dropna(subset=[id_col_name], inplace=True) # Remove rows where the ID itself is NaN

        if meta_df[id_col_name].duplicated().any():
            logger.warning(f"Duplicate IDs found in metadata column '{id_col_name}' in '{file_basename}'. "
                           "Using the first occurrence for each duplicated ID.")
            meta_df.drop_duplicates(subset=[id_col_name], keep='first', inplace=True)

        if meta_df.empty:
            logger.error(f"No valid entries remaining in metadata file '{file_basename}' after processing ID column '{id_col_name}'.")
            return None
            
        meta_df.set_index(id_col_name, inplace=True)
        logger.info(f"Successfully loaded and validated metadata from '{file_basename}' with {len(meta_df)} unique sequence entries.")
        return meta_df

    except pd.errors.EmptyDataError:
        logger.error(f"Metadata file '{file_basename}' is empty.")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading or processing metadata file '{file_basename}': {e}")
        return None

def _get_gene_files_and_names(directory_str: str) -> Tuple[List[str], Set[str]]:
    """Finds gene alignment files and extracts their names."""
    logger.info(f"Searching for gene_*.fasta files in: {directory_str}")
    gene_files: List[str] = glob.glob(os.path.join(directory_str, "gene_*.*"))
    valid_extensions: Set[str] = {".fasta", ".fa", ".fna", ".fas", ".faa"}
    gene_files = sorted([f for f in gene_files if Path(f).suffix.lower() in valid_extensions])

    if not gene_files:
        logger.error(f"No gene alignment files found in directory: {directory_str}. Exiting.")
        sys.exit(1)

    expected_gene_names: Set[str] = {name for fpath in gene_files if (name := extract_gene_name_from_file(fpath))}
    if not expected_gene_names:
        logger.error("Could not extract valid gene names from input filenames. Exiting.")
        sys.exit(1)
    logger.info(f"Found {len(gene_files)} potential gene files for {len(expected_gene_names)} unique genes.")
    return gene_files, expected_gene_names

def _determine_num_processes(requested_threads: int, num_gene_files: int) -> int:
    """Determines the number of processes for parallel analysis."""
    num_processes = requested_threads
    if num_processes <= 0:
        if MP_AVAILABLE:
            try: num_processes = os.cpu_count() or 1
            except NotImplementedError: num_processes = 1
        else: num_processes = 1
    if num_processes > 1 and not MP_AVAILABLE:
        logger.warning("Multiprocessing requested but not available. Using 1 process.")
        num_processes = 1
    
    actual_num_processes = min(num_processes, num_gene_files)
    if actual_num_processes < num_processes and num_processes > 1:
         logger.info(f"Adjusted number of processes to {actual_num_processes} (number of files or available cores).")
    logger.info(f"Using {actual_num_processes} process(es) for gene file analysis.")
    return actual_num_processes

def _run_gene_file_analysis_in_parallel(
    gene_files: List[str],
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    expected_gene_names: Set[str],
    num_processes: int,
    output_dir_path: Path
) -> List[Optional[AnalyzeGeneResultType]]: # Type alias might need update if ProcessGeneFileResultType changes
    """Runs process_analyze_gene_file in parallel or sequentially."""
    processing_task = partial(process_analyze_gene_file, 
                              args=args, 
                              reference_weights=reference_weights, 
                              expected_gene_names=expected_gene_names, 
                              output_dir_path_for_plots=output_dir_path)
    results_raw: List[Optional[AnalyzeGeneResultType]] = [] # Ensure this matches the return type of process_analyze_gene_file

    # ... (progress bar logic inchangée) ...
    progress_columns = [SpinnerColumn(), 
                        TextColumn("[progress.description]{task.description}"), 
                        BarColumn(), 
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
                        TextColumn("({task.completed}/{task.total} genes)"), 
                        TimeElapsedColumn(), TextColumn("<"), 
                        TimeRemainingColumn()]
    disable_rich = not sys.stderr.isatty() or not RICH_AVAILABLE

    with Progress(*progress_columns, 
                  transient=False, 
                  disable=disable_rich) as progress:
        analysis_task_id = progress.add_task("Analyzing Gene Files", 
                                             total=len(gene_files))
        if num_processes > 1 and MP_AVAILABLE:
            try:
                with mp.Pool(processes=num_processes) as pool:
                    for result in pool.imap(processing_task, 
                                            gene_files): # type: ignore
                        results_raw.append(result)
                        progress.update(analysis_task_id, advance=1)
            except Exception as pool_err: # pragma: no cover
                logger.exception(f"Parallel analysis error: {pool_err}. No results collected from parallel run.")
                results_raw = [None] * len(gene_files)
        else:
            for gene_file in gene_files:
                results_raw.append(processing_task(gene_file)) # type: ignore
                progress.update(analysis_task_id, advance=1)
    return results_raw # type: ignore

def _collect_and_aggregate_results(
    analyze_results_raw: List[Optional[ProcessGeneFileResultType]],
    expected_gene_names: Set[str]
) -> Tuple[
    List[pd.DataFrame],                     # all_per_sequence_dfs
    Dict[str, pd.DataFrame],                # all_ca_input_dfs (RSCU wide by gene)
    Set[str],                               # successfully_processed_genes
    List[str],                              # failed_genes_info
    Dict[str, Dict[str, Seq]],              # sequences_by_original_id (gene -> seq_id -> seq)
    Dict[str, Dict[str, float]],            # all_nucl_freqs_by_gene_agg (gene -> nucl_freq dict)
    Dict[str, Dict[str, float]],            # all_dinucl_freqs_by_gene_agg (gene -> dinucl_freq dict)
    Dict[str, Dict[str, Dict[str, float]]], # all_nucl_freqs_per_seq_in_gene
    Dict[str, Dict[str, Dict[str, float]]]  # all_dinucl_freqs_per_seq_in_gene
]:
    """Collects results from individual gene analyses, including per-sequence frequencies."""
    all_per_sequence_dfs = []
    all_ca_input_dfs = {}
    successfully_processed_genes = set()
    failed_genes_info = []
    sequences_by_original_id = {}
    all_nucl_freqs_by_gene_agg = {}
    all_dinucl_freqs_by_gene_agg = {}
    all_nucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_dinucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]] = {}

    logger.info("Collecting and aggregating analysis results from gene processing...")
    for result in analyze_results_raw:
        if result is None:
            continue
        
        # Unpack all items from ProcessGeneFileResultType
        (gene_name_res, status, per_seq_df, ca_input_df,
         nucl_freqs_agg, dinucl_freqs_agg, cleaned_map,
         per_seq_nucl_f, per_seq_dinucl_f) = result # Unpack items

        if status == "success" and gene_name_res:
            successfully_processed_genes.add(gene_name_res)
            if per_seq_df is not None: 
                all_per_sequence_dfs.append(per_seq_df)
            if ca_input_df is not None: 
                all_ca_input_dfs[gene_name_res] = ca_input_df
            if nucl_freqs_agg: 
                all_nucl_freqs_by_gene_agg[gene_name_res] = nucl_freqs_agg
            if dinucl_freqs_agg: 
                all_dinucl_freqs_by_gene_agg[gene_name_res] = dinucl_freqs_agg
            
            if cleaned_map: # {original_id: Seq}
                for seq_id, seq_obj in cleaned_map.items():
                    sequences_by_original_id.setdefault(seq_id, {})[gene_name_res] = seq_obj
            
            # Store per-sequence frequencies, nested by gene name
            if per_seq_nucl_f: # This is Dict[seq_id, Dict[nucl, freq]]
                all_nucl_freqs_per_seq_in_gene[gene_name_res] = per_seq_nucl_f
            if per_seq_dinucl_f: # This is Dict[seq_id, Dict[dinucl, freq]]
                all_dinucl_freqs_per_seq_in_gene[gene_name_res] = per_seq_dinucl_f
                
        elif gene_name_res:
            failed_genes_info.append(f"{gene_name_res} ({status})")

    # --- Logging for processed/failed genes ---
    if not successfully_processed_genes:
        logger.error("No genes were successfully processed. Exiting.")
        sys.exit(1)
    if len(successfully_processed_genes) < len(expected_gene_names):
        logger.warning(f"Processed {len(successfully_processed_genes)} genes out of {len(expected_gene_names)} expected.")
        if failed_genes_info: 
            logger.warning(f"  Failed genes/reasons: {'; '.join(failed_genes_info)}")
    
    return (all_per_sequence_dfs, all_ca_input_dfs, successfully_processed_genes,
            failed_genes_info, sequences_by_original_id,
            all_nucl_freqs_by_gene_agg, all_dinucl_freqs_by_gene_agg,
            all_nucl_freqs_per_seq_in_gene, all_dinucl_freqs_per_seq_in_gene)

def _analyze_complete_sequences_cli(
    sequences_by_original_id: Dict[str, Dict[str, Seq]],
    successfully_processed_genes: Set[str],
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    output_dir_path: Path
) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, float]],
    Optional[Dict[str, float]], Optional[Dict[str, Dict[str, float]]],
    Optional[Dict[str, Dict[str, float]]], Optional[pd.DataFrame]
]:
    logger.info("Analyzing concatenated 'complete' sequences...")
    complete_seq_records: List[SeqRecord] = []
    max_ambiguity_pct_complete = args.max_ambiguity

    for original_id, gene_seq_map in sequences_by_original_id.items():
        if set(gene_seq_map.keys()) == successfully_processed_genes:
            try:
                concat_str = "".join(str(gene_seq_map[g_name]) for g_name in sorted(gene_seq_map.keys()))
                if concat_str and len(concat_str) % 3 == 0:
                    n_count, seq_len = concat_str.count('N'), len(concat_str)
                    ambiguity = (n_count / seq_len) * 100 if seq_len > 0 else 0
                    if ambiguity <= max_ambiguity_pct_complete:
                        complete_seq_records.append(SeqRecord(Seq(concat_str), 
                                                              id=original_id, 
                                                              description=f"Concatenated {len(gene_seq_map)} genes"))
            except Exception as e: # pragma: no cover
                logger.warning(f"Error concatenating sequence for ID {original_id}: {e}")


    if not complete_seq_records:
        logger.info("No valid 'complete' sequences to analyze.")
        return None, None, None, None, None, None, None

    logger.info(f"Running analysis on {len(complete_seq_records)} 'complete' sequence records...")
    try:
        res_comp: FullAnalysisResultType = analysis.run_full_analysis(
            complete_seq_records, args.genetic_code, reference_weights
        )
        agg_usage_df_complete, per_seq_df_complete, nucl_freqs_complete_agg, \
        dinucl_freqs_complete_agg, per_seq_nucl_freqs_complete, \
        per_seq_dinucl_freqs_complete, _, _, ca_input_df_complete_plot = res_comp

        # Save RSCU boxplot for "complete" set
        if not args.skip_plots and agg_usage_df_complete is not None and not agg_usage_df_complete.empty and \
           ca_input_df_complete_plot is not None and not ca_input_df_complete_plot.empty:
            logger.info("Generating RSCU boxplot for 'complete' data...")
            try:
                long_rscu_df_comp = ca_input_df_complete_plot.reset_index().rename(
                    columns={'index': 'SequenceID'}
                )
                long_rscu_df_comp = long_rscu_df_comp.melt(id_vars=['SequenceID'],
                                                           var_name='Codon',
                                                           value_name='RSCU')
                current_gc_dict = utils.get_genetic_code(args.genetic_code)
                long_rscu_df_comp['AminoAcid'] = long_rscu_df_comp['Codon'].map(current_gc_dict.get)
                for fmt in args.plot_formats:
                    plot_filename = f"RSCU_boxplot_complete.{fmt}"
                    rscu_boxplot_complete_filepath = output_dir_path / "images" / plot_filename
                    plotting.plot_rscu_boxplot_per_gene(
                        long_rscu_df_comp,
                        agg_usage_df_complete,
                        'complete', # gene_name for title
                        str(rscu_boxplot_complete_filepath) # Full save path
                    )
            except Exception as e: # pragma: no cover
                logger.error(f"Failed to generate 'complete' RSCU boxplot: {e}")
        elif not args.skip_plots: # pragma: no cover
             logger.warning("Cannot generate 'complete' RSCU boxplot due to missing aggregate usage data for complete set.")

        return (agg_usage_df_complete, per_seq_df_complete, nucl_freqs_complete_agg,
                dinucl_freqs_complete_agg, per_seq_nucl_freqs_complete,
                per_seq_dinucl_freqs_complete, ca_input_df_complete_plot)
    except Exception as e: # pragma: no cover
        logger.exception(f"Error during 'complete' sequence analysis: {e}")
        return None, None, None, None, None, None, None


def _update_aggregate_data_with_complete_results(
    complete_results_tuple: Tuple[ # Matches return of _analyze_complete_sequences_cli
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, float]],
        Optional[Dict[str, float]], Optional[Dict[str, Dict[str, float]]],
        Optional[Dict[str, Dict[str, float]]], Optional[pd.DataFrame]
    ],
    all_per_sequence_dfs: List[pd.DataFrame],
    all_ca_input_dfs: Dict[str, pd.DataFrame],
    all_nucl_freqs_by_gene_agg: Dict[str, Dict[str, float]], # Aggregate
    all_dinucl_freqs_by_gene_agg: Dict[str, Dict[str, float]], # Aggregate
    all_nucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]], # Per-sequence
    all_dinucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]], # Per-sequence
    args: argparse.Namespace
) -> None:
    """Updates the main data collections with results from 'complete' sequence analysis."""
    (agg_usage_df_complete, per_seq_df_complete, nucl_freqs_complete_agg,
     dinucl_freqs_complete_agg, per_seq_nucl_freqs_complete,
     per_seq_dinucl_freqs_complete, ca_input_df_complete_plot) = complete_results_tuple

    if nucl_freqs_complete_agg: 
        all_nucl_freqs_by_gene_agg['complete'] = nucl_freqs_complete_agg
    if dinucl_freqs_complete_agg: 
        all_dinucl_freqs_by_gene_agg['complete'] = dinucl_freqs_complete_agg
    
    # Add per-sequence frequencies for "complete" set
    if per_seq_nucl_freqs_complete:
        all_nucl_freqs_per_seq_in_gene['complete'] = per_seq_nucl_freqs_complete
    if per_seq_dinucl_freqs_complete:
        all_dinucl_freqs_per_seq_in_gene['complete'] = per_seq_dinucl_freqs_complete

    if per_seq_df_complete is not None and not per_seq_df_complete.empty:
        if 'ID' in per_seq_df_complete.columns:
            per_seq_df_complete['Original_ID'] = per_seq_df_complete['ID']
            per_seq_df_complete['ID'] = "complete__" + per_seq_df_complete['ID'].astype(str)
        per_seq_df_complete['Gene'] = 'complete'
        all_per_sequence_dfs.append(per_seq_df_complete)

    if ca_input_df_complete_plot is not None and not ca_input_df_complete_plot.empty:
        ca_input_df_complete_combined = ca_input_df_complete_plot.copy()
        ca_input_df_complete_combined.index = "complete__" + ca_input_df_complete_combined.index.astype(str)
        all_ca_input_dfs['complete'] = ca_input_df_complete_combined

        if not args.skip_plots and agg_usage_df_complete is not None:
            logger.info("Generating RSCU boxplot for 'complete' data...")
            try:
                long_rscu_df_comp = ca_input_df_complete_plot.reset_index().rename(
                    columns={'index': 'SequenceID'}
                    )
                long_rscu_df_comp = long_rscu_df_comp.melt(id_vars=['SequenceID'], 
                                                           var_name='Codon', 
                                                           value_name='RSCU')
                current_gc_dict = utils.get_genetic_code(args.genetic_code)
                long_rscu_df_comp['AminoAcid'] = long_rscu_df_comp['Codon'].map(current_gc_dict.get)
                for fmt in args.plot_formats:
                    plotting.plot_rscu_boxplot_per_gene(long_rscu_df_comp, 
                                                        agg_usage_df_complete, 
                                                        'complete', 
                                                        args.output, fmt)
            except Exception as e: # pragma: no cover
                logger.error(f"Failed to generate 'complete' RSCU boxplot: {e}")
        elif not args.skip_plots: # pragma: no cover
             logger.warning("Cannot generate 'complete' RSCU boxplot due to missing aggregate usage data for complete set.")

def _finalize_and_save_per_sequence_metrics(all_per_sequence_dfs: List[pd.DataFrame], output_dir_path: Path) -> Optional[pd.DataFrame]:
    """Combines and saves the per-sequence metrics to output_dir_path / "data"."""
    if not all_per_sequence_dfs:
        logger.error("No per-sequence results collected. Cannot save combined metrics.")
        return None
    try:
        combined_df = pd.concat(all_per_sequence_dfs, ignore_index=True)
        # MODIFIED: Save to data subdirectory
        data_subdir = output_dir_path / "data"
        # data_subdir.mkdir(parents=True, exist_ok=True) # Already created by _ensure_output_subdirectories
        filepath = data_subdir / "per_sequence_metrics_all_genes.csv"
        combined_df.to_csv(filepath, index=False, float_format='%.5f')
        logger.info(f"Combined per-sequence metrics saved: {filepath}")
        return combined_df
    except Exception as e: # pragma: no cover
        logger.exception(f"Error concatenating/saving per-sequence results: {e}")
        return None

def _generate_summary_tables_and_stats(
    combined_per_sequence_df: pd.DataFrame,
    all_nucl_freqs_by_gene: Dict[str, Dict[str, float]],
    all_dinucl_freqs_by_gene: Dict[str, Dict[str, float]],
    output_dir_path: Path
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """Generates mean features table, statistical comparisons, and relative dinucleotide abundance table."""
    mean_summary_df: Optional[pd.DataFrame] = None
    comparison_results_df: Optional[pd.DataFrame] = None
    rel_abund_df: pd.DataFrame = pd.DataFrame()

    # Mean Features
    logger.info("Calculating mean features per gene...")
    mean_features_list = ['GC', 'GC1', 'GC2', 'GC3', 'GC12', 'RCDI', 'ENC', 'CAI', 'Aromaticity', 'GRAVY']
    available_mf = [f for f in mean_features_list if f in combined_per_sequence_df.columns]
    if 'Gene' in combined_per_sequence_df.columns and available_mf:
        try:
            temp_df = combined_per_sequence_df.copy()
            for col in available_mf: temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
            mean_summary_df = temp_df.groupby('Gene')[available_mf].mean(numeric_only=True).reset_index()
        except Exception as e: logger.exception(f"Error calculating mean features: {e}")
    else: logger.warning("Cannot calculate mean features: 'Gene' column missing or no features available.")

    # Statistical Comparisons
    logger.info("Performing statistical comparison between genes...")
    try:
        comparison_results_df = analysis.compare_features_between_genes(combined_per_sequence_df, features=mean_features_list, method='kruskal')
    except Exception as e: logger.exception(f"Error during statistical comparison: {e}")

    # Relative Dinucleotide Abundance
    logger.info("Calculating relative dinucleotide abundances...")
    rel_abund_data_list: List[Dict[str, Any]] = []
    valid_genes_for_dinucl = sorted(list(set(all_nucl_freqs_by_gene.keys()) & set(all_dinucl_freqs_by_gene.keys())))
    if valid_genes_for_dinucl:
        for gene_name in valid_genes_for_dinucl:
            nucl_f = all_nucl_freqs_by_gene.get(gene_name)
            dinucl_f = all_dinucl_freqs_by_gene.get(gene_name)
            if nucl_f and dinucl_f:
                try:
                    rel_abund = analysis.calculate_relative_dinucleotide_abundance(nucl_f, dinucl_f)
                    for dinucl, ratio in rel_abund.items():
                        rel_abund_data_list.append({'Gene': gene_name, 'Dinucleotide': dinucl, 'RelativeAbundance': ratio})
                except Exception as e: logger.warning(f"Could not calc rel dinucl abund for '{gene_name}': {e}")
        if rel_abund_data_list: rel_abund_df = pd.DataFrame(rel_abund_data_list)
    else: logger.warning("Missing data for relative dinucleotide abundance calculation.")
    
    return mean_summary_df, comparison_results_df, rel_abund_df

def _perform_and_save_combined_ca(
    all_ca_input_dfs: Dict[str, pd.DataFrame],
    output_dir_path: Path, # Main output directory
    args: argparse.Namespace
) -> Tuple[Optional[pd.DataFrame], Optional[PrinceCA], Optional[pd.Series], Optional[pd.DataFrame]]:
    if args.skip_ca:
        logger.info("Skipping combined Correspondence Analysis as requested.")
        return None, None, None, None
    if not all_ca_input_dfs:
        logger.info("Skipping combined CA: No CA input data available from gene analyses.")
        return None, None, None, None

    logger.info("Performing combined Correspondence Analysis...")
    combined_ca_input_df: Optional[pd.DataFrame] = None
    ca_results_combined: Optional[PrinceCA] = None # type: ignore
    gene_groups_for_plotting: Optional[pd.Series] = None
    ca_row_coords_df: Optional[pd.DataFrame] = None
    data_subdir = output_dir_path / "data" # Define data subdirectory path
    # data_subdir.mkdir(parents=True, exist_ok=True) # Already created by _ensure_output_subdirectories

    try:
        combined_ca_input_df = pd.concat(all_ca_input_dfs.values())
        if combined_ca_input_df.empty:
            logger.warning("Combined CA input DataFrame is empty before cleaning. Skipping CA.")
            return None, None, None, None
        logger.debug(f"Shape of combined_ca_input_df before CLI cleaning: {combined_ca_input_df.shape}")
        for col in combined_ca_input_df.columns:
            combined_ca_input_df[col] = pd.to_numeric(combined_ca_input_df[col], 
                                                      errors='coerce')
        combined_ca_input_df.fillna(0.0, inplace=True)
        combined_ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True)
        logger.debug(f"Shape of combined_ca_input_df after CLI cleaning: {combined_ca_input_df.shape}")

        if not combined_ca_input_df.empty:
            split_idx = combined_ca_input_df.index.str.split('__', n=1, expand=True)
            if split_idx.nlevels > 0 and not split_idx.empty:
                 gene_groups_for_plotting = pd.Series(data=split_idx.get_level_values(0), 
                                                      index=combined_ca_input_df.index, 
                                                      name='Gene')
            else: # pragma: no cover
                 logger.warning("Could not reliably parse gene groups from CA input DataFrame index for plotting.")
                 gene_groups_for_plotting = None

            ca_results_combined = analysis.perform_ca(combined_ca_input_df.copy())

            if ca_results_combined:
                logger.info("Combined CA complete. Saving details...")
                ca_row_coords_df = ca_results_combined.row_coordinates(combined_ca_input_df)

                # Save to data_subdir
                ca_row_coords_df.to_csv(data_subdir / "ca_row_coordinates.csv", 
                                        float_format='%.5f')
                ca_results_combined.column_coordinates(
                    combined_ca_input_df
                ).to_csv(
                    data_subdir / "ca_col_coordinates.csv", 
                    float_format='%.5f'
                )
                if hasattr(ca_results_combined, 
                           'column_contributions_'):
                    ca_results_combined.column_contributions_.to_csv(data_subdir / "ca_col_contributions.csv", float_format='%.5f')
                if hasattr(ca_results_combined, 
                           'eigenvalues_summary'):
                    ca_results_combined.eigenvalues_summary.to_csv(data_subdir / "ca_eigenvalues.csv", float_format='%.5f')

                rscu_wide_path = data_subdir / "per_sequence_rscu_wide.csv"
                combined_ca_input_df.to_csv(rscu_wide_path, float_format='%.4f')
                logger.info(f"Per-sequence RSCU (wide) saved: {rscu_wide_path}")
            else: # pragma: no cover
                logger.warning("Combined CA fitting failed or produced no result.")
                combined_ca_input_df = None
        else: # pragma: no cover
            logger.warning("Combined CA input data is empty. Skipping CA.")
            combined_ca_input_df = None
    except Exception as e: # pragma: no cover
        logger.exception(f"Error during combined CA: {e}")
        combined_ca_input_df, ca_results_combined, gene_groups_for_plotting, ca_row_coords_df = None, None, None, None

    return combined_ca_input_df, ca_results_combined, \
        gene_groups_for_plotting, ca_row_coords_df


def _generate_color_palette_for_groups(combined_per_sequence_df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """Generates a color palette for unique gene groups."""
    if combined_per_sequence_df is None or 'Gene' not in combined_per_sequence_df.columns:
        logger.warning("Cannot generate color map: Missing combined data or 'Gene' column.")
        return None
    
    unique_genes = sorted([g for g in combined_per_sequence_df['Gene'].unique() if g != 'complete'])
    if 'complete' in combined_per_sequence_df['Gene'].unique():
        unique_genes.append('complete')
    
    if not unique_genes:
        logger.warning("No unique gene groups found for color map generation.")
        return None
        
    try:
        palette_list = sns.color_palette("husl", n_colors=len(unique_genes))
        gene_color_map = {gene: color for gene, color in zip(unique_genes, palette_list)}
        logger.debug(f"Generated color map for {len(unique_genes)} groups.")
        return gene_color_map
    except Exception as e:
        logger.warning(f"Could not generate color palette: {e}. Plot colors may be default.")
        return None
    
def _generate_plots_per_gene_colored_by_metadata(
    args: argparse.Namespace,
    combined_per_sequence_df_with_meta: pd.DataFrame,
    all_ca_input_dfs: Dict[str, pd.DataFrame],
    all_nucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]],
    all_dinucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]],
    metadata_col_for_color: str,
    top_n_categories_map: Dict[str, str],
    output_dir_path: Path, # This is the main output directory (e.g., codon_analysis_results)
    gene_color_map_for_metadata: Optional[Dict[str, Any]]
):
    logger.info(f"Generating per-gene plots colored by metadata column '{metadata_col_for_color}'...")
    # Base directory for these specific plots, under the main "images" directory
    metadata_plots_image_base_dir = output_dir_path / "images" / f"{utils.sanitize_filename(metadata_col_for_color)}_per_gene_plots"
    # metadata_plots_image_base_dir.mkdir(parents=True, exist_ok=True) # Plotting functions will create if needed

    plot_hue_col = f"{metadata_col_for_color}_topN"
    if metadata_col_for_color in combined_per_sequence_df_with_meta:
        if plot_hue_col not in combined_per_sequence_df_with_meta.columns:
             combined_per_sequence_df_with_meta[plot_hue_col] = combined_per_sequence_df_with_meta[metadata_col_for_color].astype(str).map(top_n_categories_map)
    else: # pragma: no cover
        logger.error(f"Metadata column '{metadata_col_for_color}' not found for hue mapping. Skipping per-gene metadata plots.")
        return
    unique_genes = combined_per_sequence_df_with_meta['Gene'].unique()


    for gene_name in unique_genes:
        # Full path to the directory for this gene's metadata-colored plots
        gene_specific_metadata_plots_dir_abs = metadata_plots_image_base_dir / utils.sanitize_filename(gene_name)
        # gene_specific_metadata_plots_dir_abs.mkdir(parents=True, exist_ok=True) # Plotting functions will create

        logger.info(f"  Processing gene '{gene_name}' for metadata-colored plots (output to: {gene_specific_metadata_plots_dir_abs})")
        gene_specific_metrics_meta_df = combined_per_sequence_df_with_meta[
            combined_per_sequence_df_with_meta['Gene'] == gene_name
        ].copy()
        if gene_specific_metrics_meta_df.empty: # pragma: no cover
            logger.warning(f"No metric data for gene '{gene_name}' with metadata. Skipping plots for this gene.")
            continue
        if plot_hue_col not in gene_specific_metrics_meta_df.columns: # pragma: no cover
            logger.error(f"Plot hue column '{plot_hue_col}' is missing from data for gene '{gene_name}'. Skipping plots.")
            continue

        plot_title_prefix = f"Gene {gene_name}: "
        filename_suffix_for_gene = f"_{utils.sanitize_filename(gene_name)}_by_{utils.sanitize_filename(metadata_col_for_color)}"

        if not args.skip_plots:
            for fmt in args.plot_formats:
                # Construct full output filepaths for each plot
                enc_plot_filepath = gene_specific_metadata_plots_dir_abs / f"enc_vs_gc3_plot{filename_suffix_for_gene}.{fmt}"
                plotting.plot_enc_vs_gc3(
                    per_sequence_df=gene_specific_metrics_meta_df,
                    output_filepath=str(enc_plot_filepath), # Pass full path
                    # output_dir and file_format are removed from plotting function
                    group_by_col=plot_hue_col, palette=gene_color_map_for_metadata,
                    # filename_suffix and plot_title_prefix are for internal use by plotting fn if it constructs its own titles/names
                    # but since we pass full path, they might be less critical for the plotting fn itself.
                    # However, keeping them allows the plotting function to still generate consistent titles.
                    filename_suffix=filename_suffix_for_gene, # Keep for title consistency
                    plot_title_prefix=plot_title_prefix
                )
                neutrality_plot_filepath = gene_specific_metadata_plots_dir_abs / f"neutrality_plot{filename_suffix_for_gene}.{fmt}"
                plotting.plot_neutrality(
                    per_sequence_df=gene_specific_metrics_meta_df,
                    output_filepath=str(neutrality_plot_filepath),
                    group_by_col=plot_hue_col, palette=gene_color_map_for_metadata,
                    filename_suffix=filename_suffix_for_gene, plot_title_prefix=plot_title_prefix
                )

        rscu_df_for_gene: Optional[pd.DataFrame] = all_ca_input_dfs.get(gene_name)
        if not args.skip_ca and rscu_df_for_gene is not None and not rscu_df_for_gene.empty:
            temp_metrics_for_hue = gene_specific_metrics_meta_df.set_index('ID', drop=False)
            common_indices_ca = rscu_df_for_gene.index.intersection(temp_metrics_for_hue.index)

            if not common_indices_ca.empty:
                rscu_df_for_ca_plot = rscu_df_for_gene.loc[common_indices_ca]
                hue_series_for_ca = temp_metrics_for_hue.loc[common_indices_ca, plot_hue_col]

                if not rscu_df_for_ca_plot.empty and rscu_df_for_ca_plot.shape[0] >=2 and rscu_df_for_ca_plot.shape[1] >=2 : # type: ignore
                    ca_results_gene = analysis.perform_ca(rscu_df_for_ca_plot) # type: ignore
                    if ca_results_gene:
                        for fmt in args.plot_formats:
                            # Construct full output filepath for CA plot
                            ca_plot_filename = f"ca_biplot_comp{args.ca_dims[0]+1}v{args.ca_dims[1]+1}{filename_suffix_for_gene}.{fmt}"
                            ca_plot_filepath = gene_specific_metadata_plots_dir_abs / ca_plot_filename
                            plotting.plot_ca(
                                ca_results=ca_results_gene, ca_input_df=rscu_df_for_ca_plot, # type: ignore
                                output_filepath=str(ca_plot_filepath), # Pass full path
                                comp_x=args.ca_dims[0], comp_y=args.ca_dims[1],
                                groups=hue_series_for_ca, palette=gene_color_map_for_metadata,
                                filename_suffix=filename_suffix_for_gene, plot_title_prefix=plot_title_prefix
                            )
                else: # pragma: no cover
                    logger.warning(f"Not enough data for CA for gene '{gene_name}' after filtering for metadata plot ({rscu_df_for_ca_plot.shape if rscu_df_for_ca_plot is not None else 'None'}).") # type: ignore
            else: # pragma: no cover
                 logger.warning(f"No common sequences between RSCU data and metrics for CA plot of gene '{gene_name}'.")


        nucl_freqs_per_seq_this_gene = all_nucl_freqs_per_seq_in_gene.get(gene_name)
        dinucl_freqs_per_seq_this_gene = all_dinucl_freqs_per_seq_in_gene.get(gene_name)
        if not args.skip_plots and nucl_freqs_per_seq_this_gene and dinucl_freqs_per_seq_this_gene:
            per_sequence_oe_ratios_list = []
            for original_seq_id_for_dinucl in gene_specific_metrics_meta_df['Original_ID'].unique():
                seq_meta_row = gene_specific_metrics_meta_df[
                    gene_specific_metrics_meta_df['Original_ID'] == original_seq_id_for_dinucl
                ].iloc[0]
                metadata_category = seq_meta_row[plot_hue_col]
                nucl_freqs_s = nucl_freqs_per_seq_this_gene.get(str(original_seq_id_for_dinucl))
                dinucl_freqs_s = dinucl_freqs_per_seq_this_gene.get(str(original_seq_id_for_dinucl))
                if nucl_freqs_s and dinucl_freqs_s:
                    oe_ratios_seq = analysis.calculate_relative_dinucleotide_abundance(nucl_freqs_s, dinucl_freqs_s)
                    for dinucl, ratio in oe_ratios_seq.items():
                        per_sequence_oe_ratios_list.append({
                            'SequenceID': original_seq_id_for_dinucl,
                            'Dinucleotide': dinucl,
                            'RelativeAbundance': ratio,
                            plot_hue_col: metadata_category
                        })
            if per_sequence_oe_ratios_list:
                per_sequence_oe_ratios_df = pd.DataFrame(per_sequence_oe_ratios_list)
                for fmt in args.plot_formats:
                    # Construct full output filepath
                    dinucl_plot_filename = f"dinucl_abundance{filename_suffix_for_gene}.{fmt}"
                    dinucl_plot_filepath = gene_specific_metadata_plots_dir_abs / dinucl_plot_filename
                    plotting.plot_per_gene_dinucleotide_abundance_by_metadata(
                        per_sequence_oe_ratios_df=per_sequence_oe_ratios_df,
                        metadata_hue_col=plot_hue_col,
                        output_filepath=str(dinucl_plot_filepath), # Pass full path
                        palette=gene_color_map_for_metadata,
                        gene_name=gene_name,
                        filename_suffix_extra=filename_suffix_for_gene # Keep for title consistency
                    )
            else: # pragma: no cover
                logger.warning(f"No per-sequence O/E ratios calculated for dinucleotide plot of gene '{gene_name}'.")
        elif not args.skip_plots: # pragma: no cover
            logger.info(f"Skipping per-sequence dinucleotide plot for gene '{gene_name}' due to missing per-sequence frequency data.")

def _generate_all_combined_plots(
    args: argparse.Namespace,
    combined_per_sequence_df: Optional[pd.DataFrame],
    gene_color_map: Optional[Dict[str, Any]],
    rel_abund_df: pd.DataFrame,
    ca_results_combined: Optional[PrinceCA], # type: ignore
    combined_ca_input_df: Optional[pd.DataFrame],
    gene_groups_for_plotting: Optional[pd.Series],
    ca_row_coords: Optional[pd.DataFrame],
    reference_data_for_plot: Optional[pd.DataFrame],
    output_dir_path: Path # Main output directory
) -> None:
    if args.skip_plots:
        logger.info("Skipping combined plot generation as requested.")
        return

    logger.info("Generating combined plots...")
    # Base directory for combined images
    output_images_dir_abs = output_dir_path / "images"
    # output_images_dir_abs.mkdir(parents=True, exist_ok=True) # Plotting functions will handle parent dir creation

    plot_formats = args.plot_formats
    n_ca_dims_variance, n_ca_contrib_top = 10, 10

    for fmt in plot_formats:
        logger.debug(f"Generating combined plots in format: {fmt}")
        try:
            if combined_per_sequence_df is not None:
                # Construct full filepaths
                gc_means_filename = f"gc_means_barplot_by_Gene.{fmt}"
                plotting.plot_gc_means_barplot(combined_per_sequence_df, str(output_images_dir_abs / gc_means_filename), 'Gene')

                enc_gc3_filename = f"enc_vs_gc3_plot_{utils.sanitize_filename('_grouped_by_gene')}.{fmt}"
                plotting.plot_enc_vs_gc3(combined_per_sequence_df,
                                         output_filepath=str(output_images_dir_abs / enc_gc3_filename),
                                         group_by_col='Gene', palette=gene_color_map,
                                         filename_suffix="_grouped_by_gene")

                neutrality_filename = f"neutrality_plot_{utils.sanitize_filename('_grouped_by_gene')}.{fmt}"
                plotting.plot_neutrality(combined_per_sequence_df,
                                         output_filepath=str(output_images_dir_abs / neutrality_filename),
                                         group_by_col='Gene', palette=gene_color_map,
                                         filename_suffix="_grouped_by_gene")

                features_for_corr = ['GC', 'GC1', 'GC2', 'GC3', 'GC12', 'ENC', 'CAI', 'RCDI', 'Aromaticity', 'GRAVY', 'Length', 'TotalCodons']
                available_corr_feat = [f for f in features_for_corr if f in combined_per_sequence_df.columns]
                
                if len(available_corr_feat) > 1:
                    corr_heatmap_filename = f"feature_correlation_heatmap_spearman.{fmt}"
                    plotting.plot_correlation_heatmap(combined_per_sequence_df, available_corr_feat, str(output_images_dir_abs / corr_heatmap_filename))

            if not rel_abund_df.empty:
                rel_dinucl_filename = f"relative_dinucleotide_abundance.{fmt}"
                plotting.plot_relative_dinucleotide_abundance(rel_abund_df, str(output_images_dir_abs / rel_dinucl_filename), palette=gene_color_map) # type: ignore

            if ca_results_combined and combined_ca_input_df is not None:
                ca_suffix = utils.sanitize_filename('_combined_by_gene')
                ca_biplot_filename = f"ca_biplot_comp{args.ca_dims[0]+1}v{args.ca_dims[1]+1}_{ca_suffix}.{fmt}"
                plotting.plot_ca(ca_results_combined, combined_ca_input_df, str(output_images_dir_abs / ca_biplot_filename),
                                args.ca_dims[0], args.ca_dims[1], groups=gene_groups_for_plotting,
                                palette=gene_color_map, filename_suffix="_combined_by_gene") # type: ignore

                ca_var_filename = f"ca_variance_explained_top{n_ca_dims_variance}.{fmt}"
                plotting.plot_ca_variance(ca_results_combined, n_ca_dims_variance, 
                                          str(output_images_dir_abs / ca_var_filename)) # type: ignore
                
                if hasattr(ca_results_combined, 'column_contributions_'):
                    
                    if ca_results_combined.column_contributions_.shape[1] > 0:
                         ca_contrib1_filename = f"ca_contribution_dim1_top{n_ca_contrib_top}.{fmt}"
                         plotting.plot_ca_contribution(ca_results_combined, 0, n_ca_contrib_top, str(output_images_dir_abs / ca_contrib1_filename)) # type: ignore
                    
                    if ca_results_combined.column_contributions_.shape[1] > 1:
                         ca_contrib2_filename = f"ca_contribution_dim2_top{n_ca_contrib_top}.{fmt}"
                         plotting.plot_ca_contribution(ca_results_combined, 1, n_ca_contrib_top, str(output_images_dir_abs / ca_contrib2_filename)) # type: ignore

            if ca_row_coords is not None and combined_per_sequence_df is not None and combined_ca_input_df is not None:
                dim_x_idx, dim_y_idx = args.ca_dims[0], args.ca_dims[1]
                max_available_dim = ca_row_coords.shape[1] - 1
                ca_dims_prepared_df_for_plot: Optional[pd.DataFrame] = None
                
                if not (dim_x_idx > max_available_dim or dim_y_idx > max_available_dim or dim_x_idx == dim_y_idx):
                    ca_dims_prepared_df_for_plot = ca_row_coords[[dim_x_idx, dim_y_idx]].copy()
                    ca_dims_prepared_df_for_plot.columns = [f'CA_Dim{dim_x_idx+1}', f'CA_Dim{dim_y_idx+1}']

                    if not ca_dims_prepared_df_for_plot.index.is_unique: # pragma: no cover
                        ca_dims_prepared_df_for_plot = ca_dims_prepared_df_for_plot[~ca_dims_prepared_df_for_plot.index.duplicated(keep='first')]
                
                metrics_prepared_df_for_plot: Optional[pd.DataFrame] = None
                if 'ID' in combined_per_sequence_df.columns:
                    metrics_prepared_df_for_plot = combined_per_sequence_df.copy()

                    if not metrics_prepared_df_for_plot['ID'].is_unique: # pragma: no cover
                        metrics_prepared_df_for_plot.drop_duplicates(subset=['ID'], keep='first', inplace=True)
                    metrics_prepared_df_for_plot.set_index('ID', inplace=True)
                
                rscu_prepared_df_for_plot = combined_ca_input_df.copy()
                if not rscu_prepared_df_for_plot.index.is_unique: # pragma: no cover
                     rscu_prepared_df_for_plot = rscu_prepared_df_for_plot[~rscu_prepared_df_for_plot.index.duplicated(keep='first')]

                if ca_dims_prepared_df_for_plot is not None and metrics_prepared_df_for_plot is not None:
                    metric_features = ['Length', 'TotalCodons', 'GC', 'GC1', 'GC2', 'GC3', 'GC12', 
                                       'ENC', 'CAI', 'Fop', 'RCDI', 'ProteinLength', 'GRAVY', 'Aromaticity']
                    available_metric_f = [f for f in metric_features if f in metrics_prepared_df_for_plot.columns]
                    available_rscu_c = sorted([col for col in rscu_prepared_df_for_plot.columns if len(col) == 3 and col.isupper()]) # type: ignore
                    features_corr_plot = available_metric_f + available_rscu_c
                    
                    if features_corr_plot:
                        ca_feat_corr_filename = f"ca_axes_feature_corr_{utils.sanitize_filename('Spearman')}.{fmt}"
                        plotting.plot_ca_axes_feature_correlation(
                            ca_dims_df=ca_dims_prepared_df_for_plot,
                            metrics_df=metrics_prepared_df_for_plot,
                            rscu_df=rscu_prepared_df_for_plot, # type: ignore
                            output_filepath=str(output_images_dir_abs / ca_feat_corr_filename), # Pass full path
                            features_to_correlate=features_corr_plot
                        )
            # RSCU comparison plot logic would also need to save to output_images_dir_abs
            # if reference_data_for_plot is not None and combined_per_sequence_df is not None:
            #    ...
            #    rscu_comp_filename = f"rscu_comparison_scatter.{fmt}"
            #    plotting.plot_usage_comparison(agg_usage_df_for_comp, reference_data_for_plot, str(output_images_dir_abs / rscu_comp_filename))


        except Exception as plot_err: # pragma: no cover
            logger.exception(f"Error during combined plot generation for format '{fmt}': {plot_err}")


def _save_main_output_tables(
    output_dir_path: Path, # Main output directory
    combined_per_sequence_df: Optional[pd.DataFrame], # This is already saved by _finalize_and_save_per_sequence_metrics
    mean_summary_df: Optional[pd.DataFrame],
    comparison_results_df: Optional[pd.DataFrame]
) -> None:
    """Saves the main output CSV tables to output_dir_path / "data"."""
    data_subdir = output_dir_path / "data"
    # data_subdir.mkdir(parents=True, exist_ok=True) # Already created by _ensure_output_subdirectories

    # combined_per_sequence_df is saved by _finalize_and_save_per_sequence_metrics
    if mean_summary_df is not None and not mean_summary_df.empty:
        filepath = data_subdir / "mean_features_per_gene.csv"
        mean_summary_df.to_csv(filepath, index=False, float_format='%.4f')
        logger.info(f"Mean features per gene saved: {filepath}")

    if comparison_results_df is not None and not comparison_results_df.empty:
        filepath = data_subdir / "gene_comparison_stats.csv"
        comparison_results_df.to_csv(filepath, index=False, float_format='%.4g')
        logger.info(f"Gene comparison statistics saved: {filepath}")


# --- Main Command Handler Refactored ---

def handle_analyze_command(args: argparse.Namespace) -> None:
    logger.info(f"Running 'analyze' command with input directory: {args.directory}")

    # 1. Setup output directory (deferred from original, now done early by this helper)
    output_dir_path = _setup_output_directory(args.output)
    _ensure_output_subdirectories(output_dir_path)

    # 2. Load Reference Data
    reference_weights, reference_data_for_plot = _load_reference_data(args)

    # 3. Load metadata
    metadata_df = _load_metadata(args.metadata, args.metadata_id_col, args.metadata_delimiter) # type: ignore

    # 4. Get Gene Files and Expected Names
    gene_files, expected_gene_names = _get_gene_files_and_names(args.directory)

    # 5. Determine Number of Processes
    num_processes = _determine_num_processes(args.threads, len(gene_files))

    # 6. Run Gene File Analysis (Parallel/Sequential)
    analyze_results_raw = _run_gene_file_analysis_in_parallel(
        gene_files, args, 
        reference_weights, 
        expected_gene_names, 
        num_processes, 
        output_dir_path
    )

    # 7. Collect and Aggregate Initial Results
    (all_per_sequence_dfs,
     all_ca_input_dfs,
     successfully_processed_genes, _,
     sequences_by_original_id,
     all_nucl_freqs_by_gene_agg,
     all_dinucl_freqs_by_gene_agg,
     all_nucl_freqs_per_seq_in_gene,
     all_dinucl_freqs_per_seq_in_gene) = \
        _collect_and_aggregate_results(analyze_results_raw, # type: ignore
                                       expected_gene_names)

    # 8. Analyze "Complete" Sequences and Update Aggregates
    complete_analysis_results_tuple = (None, None, None, None, None, None, None) # Adjusted tuple size
    if sequences_by_original_id:
        complete_analysis_results_tuple = _analyze_complete_sequences_cli( # type: ignore
            sequences_by_original_id, successfully_processed_genes, args, reference_weights, output_dir_path
        )
        _update_aggregate_data_with_complete_results( # type: ignore
            complete_analysis_results_tuple,
            all_per_sequence_dfs,
            all_ca_input_dfs,
            all_nucl_freqs_by_gene_agg,
            all_dinucl_freqs_by_gene_agg,
            all_nucl_freqs_per_seq_in_gene,
            all_dinucl_freqs_per_seq_in_gene,
            args
        )

    # 9. Finalize and Save Per-Sequence Metrics (main combined table)
    combined_per_sequence_df = _finalize_and_save_per_sequence_metrics(all_per_sequence_dfs, output_dir_path) # type: ignore
    if combined_per_sequence_df is None:
        logger.error("Failed to produce combined per-sequence metrics. Exiting.")
        sys.exit(1)

    # 10. Merge Metadata
    combined_per_sequence_df_with_meta = combined_per_sequence_df.copy()
    if metadata_df is not None:
        logger.info("Merging metadata with analysis results...")
        if 'Original_ID' in combined_per_sequence_df_with_meta.columns:
            combined_per_sequence_df_with_meta['Original_ID_str_for_merge'] = combined_per_sequence_df_with_meta['Original_ID'].astype(str)
            metadata_df.index = metadata_df.index.astype(str)

            combined_per_sequence_df_with_meta = pd.merge(
                combined_per_sequence_df_with_meta, metadata_df,
                left_on='Original_ID_str_for_merge', right_index=True,
                how='left', suffixes=('', '_meta')
            )
            combined_per_sequence_df_with_meta.drop(columns=['Original_ID_str_for_merge'], inplace=True, errors='ignore')
            logger.info("Metadata merged with per-sequence metrics.")
        else:
            logger.warning("Could not merge metadata: 'Original_ID' column missing in analysis results.")

    # 11. Generate plots colored by metadata if option is set
    if args.color_by_metadata and metadata_df is not None:
        metadata_col_for_color = args.color_by_metadata
        if metadata_col_for_color not in combined_per_sequence_df_with_meta.columns:
            logger.error(f"Metadata column '{metadata_col_for_color}' specified for coloring not found after merge. Skipping these plots.")
        else:
            # Ensure the column is treated as categorical (string) for value_counts
            combined_per_sequence_df_with_meta[metadata_col_for_color] = \
                combined_per_sequence_df_with_meta[metadata_col_for_color].astype(str).fillna("Unknown") # Fill NaNs in metadata for grouping
            category_counts = combined_per_sequence_df_with_meta[metadata_col_for_color].value_counts()

            top_n_categories_map: Dict[str, str] = {}
            palette_categories_for_map: List[str]

            if len(category_counts) > args.metadata_max_categories:
                logger.warning(f"Metadata column '{metadata_col_for_color}' has {len(category_counts)} categories.\
                                Limiting to top {args.metadata_max_categories} and 'Other'.")
                top_categories_list = category_counts.nlargest(args.metadata_max_categories).index.tolist()

                all_unique_cats_in_col = combined_per_sequence_df_with_meta[metadata_col_for_color].unique()
                for cat_val in all_unique_cats_in_col:
                    top_n_categories_map[cat_val] = cat_val if cat_val in top_categories_list else "Other"
                palette_categories_for_map = top_categories_list + (["Other"] if "Other" in top_n_categories_map.values() else [])
            else:
                all_unique_cats_in_col = combined_per_sequence_df_with_meta[metadata_col_for_color].unique()
                for cat_val in all_unique_cats_in_col:
                    top_n_categories_map[cat_val] = cat_val
                palette_categories_for_map = category_counts.index.tolist()

            metadata_category_color_map_final: Optional[Dict[str, Any]] = None
            if palette_categories_for_map:
                 palette_list_meta = sns.color_palette("husl", n_colors=len(palette_categories_for_map))
                 metadata_category_color_map_final = {cat: color for cat, color in zip(palette_categories_for_map, palette_list_meta)}

            # This is where we pass the per-sequence frequency data
        
            _generate_plots_per_gene_colored_by_metadata( # type: ignore
                args,
                combined_per_sequence_df_with_meta,
                all_ca_input_dfs,
                all_nucl_freqs_per_seq_in_gene,
                all_dinucl_freqs_per_seq_in_gene,
                metadata_col_for_color,
                top_n_categories_map, # type: ignore
                output_dir_path, # This is the main output directory
                metadata_category_color_map_final # type: ignore
            )

    # 12. Generate Summary Tables (Means, Stats) and Relative Dinucleotide Abundance Table
    logger.info("Proceeding with standard combined analysis and plots...")

    mean_summary_df, comparison_results_df, rel_abund_df = _generate_summary_tables_and_stats( # type: ignore
        combined_per_sequence_df_with_meta,
        all_nucl_freqs_by_gene_agg,
        all_dinucl_freqs_by_gene_agg,
        output_dir_path # Pass for context, though it might not use it directly for saving.
    )

    # 13. Perform Combined CA and Save Details
    (combined_ca_input_df_final, ca_results_combined,
     gene_groups_for_plotting, ca_row_coords_final) = _perform_and_save_combined_ca( # type: ignore
        all_ca_input_dfs, output_dir_path, args
    )

    # 14. Prepare Color Palette for Plotting
    gene_color_map_standard = _generate_color_palette_for_groups(combined_per_sequence_df_with_meta) # type: ignore

    # 15. Generate All Combined Plots
    _generate_all_combined_plots( # type: ignore
        args, combined_per_sequence_df_with_meta, gene_color_map_standard,
        rel_abund_df, ca_results_combined, combined_ca_input_df_final,
        gene_groups_for_plotting, ca_row_coords_final,
        reference_data_for_plot,
        output_dir_path
    )

    # 16. Save Remaining Output Tables
    _save_main_output_tables(output_dir_path, combined_per_sequence_df_with_meta, # type: ignore
                             mean_summary_df, comparison_results_df)

    # 17. Generate HTML report if not disabled by --no-html-report
    if not args.no_html_report: # Check for --no-html-report flag
        if not reporting.JINJA2_AVAILABLE:
            logger.error("Cannot generate HTML report: Jinja2 library is not installed. Please install it (`pip install Jinja2`).")
        else:
            logger.info("Preparing data for HTML report generation...")
            # output_dir_path is the main analysis output directory
            report_gen = reporting.HTMLReportGenerator(output_dir_path, vars(args))

            # 1. Add summary statistics
            num_processed_genes = len(successfully_processed_genes) if successfully_processed_genes else 0
            total_valid_seqs = len(combined_per_sequence_df_with_meta) if combined_per_sequence_df_with_meta is not None else 0
            report_gen.add_summary_data(num_genes_processed=num_processed_genes, total_valid_sequences=total_valid_seqs)

            # --- Calculate the summary of sequence count/length per gene ---
            gene_sequence_summary_df = None
            if combined_per_sequence_df_with_meta is not None and not combined_per_sequence_df_with_meta.empty and \
               'Gene' in combined_per_sequence_df_with_meta.columns and \
               'Length' in combined_per_sequence_df_with_meta.columns:
                try:
                    gene_sequence_summary_df = combined_per_sequence_df_with_meta.groupby('Gene').agg(
                        num_sequences=('ID', 'count'),
                        mean_length=('Length', 'mean'),
                        min_length=('Length', 'min'),
                        max_length=('Length', 'max')
                    ).reset_index()
                    gene_sequence_summary_df.rename(columns={
                        'Gene': 'Gene Name',
                        'num_sequences': 'Number of Sequences',
                        'mean_length': 'Mean Length (bp)',
                        'min_length': 'Min Length (bp)',
                        'max_length': 'Max Length (bp)'
                    }, inplace=True)
                    # Format float columns
                    for col in ['Mean Length (bp)']:
                        if col in gene_sequence_summary_df.columns:
                             gene_sequence_summary_df[col] = gene_sequence_summary_df[col].round(1)
                    logger.debug("Generated gene sequence summary table for HTML report.")
                except Exception as e:
                    logger.error(f"Could not generate gene sequence summary table for HTML report: {e}")
                    gene_sequence_summary_df = pd.DataFrame() # Empty df if error


            # 2. Add tables (DataFrames)
            report_gen.add_table(
                "per_sequence_metrics",
                combined_per_sequence_df_with_meta,
                table_csv_path_relative_to_outdir="data/per_sequence_metrics_all_genes.csv", # Example path
                display_in_html=False
            )
            report_gen.add_table("gene_sequence_summary",
                                 gene_sequence_summary_df,
                                 table_csv_path_relative_to_outdir="data/gene_sequence_summary.csv" if gene_sequence_summary_df is not None and not gene_sequence_summary_df.empty else None
                                 ) # Need to save this CSV first
            if gene_sequence_summary_df is not None and not gene_sequence_summary_df.empty:
                 (output_dir_path / "data" / "gene_sequence_summary.csv").parent.mkdir(parents=True, exist_ok=True)
                 gene_sequence_summary_df.to_csv(output_dir_path / "data" / "gene_sequence_summary.csv", index=False, float_format='%.1f')


            report_gen.add_table("mean_features_per_gene",
                                 mean_summary_df,
                                 table_csv_path_relative_to_outdir="data/mean_features_per_gene.csv" if mean_summary_df is not None else None)
            report_gen.add_table("gene_comparison_stats",
                                 comparison_results_df,
                                 table_csv_path_relative_to_outdir="data/gene_comparison_stats.csv" if comparison_results_df is not None else None)

            # For CA tables, ensure they are DataFrames before passing
            ca_performed_successfully = False
            if ca_results_combined and combined_ca_input_df_final is not None:
                ca_performed_successfully = True
                report_gen.add_table("ca_combined_row_coordinates",
                                     ca_row_coords_final, # This is already a DataFrame
                                     table_csv_path_relative_to_outdir="data/ca_row_coordinates.csv" if ca_row_coords_final is not None else None,
                                     display_in_html=False, display_index=True)
                # ... (similar for other CA tables) ...
                if hasattr(ca_results_combined, 'column_coordinates'):
                    df_ca_col = ca_results_combined.column_coordinates(combined_ca_input_df_final)
                    report_gen.add_table("ca_combined_col_coordinates", df_ca_col,
                                         table_csv_path_relative_to_outdir="data/ca_col_coordinates.csv",
                                         display_in_html=False, display_index=True)
                if hasattr(ca_results_combined, 'column_contributions_'):
                    df_ca_contrib = ca_results_combined.column_contributions_
                    report_gen.add_table("ca_combined_col_contributions", df_ca_contrib,
                                         table_csv_path_relative_to_outdir="data/ca_col_contributions.csv",
                                         display_index=True)
                if hasattr(ca_results_combined, 'eigenvalues_summary'):
                    df_ca_eigen = ca_results_combined.eigenvalues_summary
                    report_gen.add_table("ca_combined_eigenvalues", df_ca_eigen,
                                         table_csv_path_relative_to_outdir="data/ca_eigenvalues.csv",
                                         display_index=True)

            report_gen.set_ca_performed_status(ca_performed_successfully)

            if combined_ca_input_df_final is not None:
                 report_gen.add_table("per_sequence_rscu_wide",
                                      combined_ca_input_df_final,
                                      table_csv_path_relative_to_outdir="data/per_sequence_rscu_wide.csv",
                                      display_in_html=False)

            # 3. Add paths to COMBINED plots
            report_plot_format = args.plot_formats[0]
            def get_plot_rel_path_for_report(base_filename_pattern: str) -> Optional[str]:
                # Constructs path like "images/plot_name.format"
                # This path is relative to output_dir_path
                filename = f"{base_filename_pattern}.{report_plot_format}"
                # Check if the actual file exists in output_dir_path / "images" / filename
                if (output_dir_path / "images" / filename).exists():
                    return f"images/{filename}"
                # For plots in subdirs of "images"
                elif (output_dir_path / "images" / base_filename_pattern.split('/')[0] / f"{base_filename_pattern.split('/')[-1]}.{report_plot_format}").exists() and '/' in base_filename_pattern:
                     # This part is tricky if base_filename_pattern includes subdirs already.
                     # Let's assume base_filename_pattern is just the core name for now.
                     # For complex paths, the caller to add_plot should construct the correct relative path.
                     return f"images/{base_filename_pattern.split('/')[0]}/{base_filename_pattern.split('/')[-1]}.{report_plot_format}"
                logger.warning(f"Plot file for report not found: images/{filename} or similar structured path.")
                return None
            
            report_gen.add_plot(
                "gc_means_barplot_by_Gene",
                get_plot_rel_path_for_report("gc_means_barplot_by_Gene")
            )
            report_gen.add_plot("enc_vs_gc3_combined",
                get_plot_rel_path_for_report(f"enc_vs_gc3_plot_{utils.sanitize_filename('_grouped_by_gene')}"))
            report_gen.add_plot("neutrality_plot_combined",
                get_plot_rel_path_for_report(f"neutrality_plot_{utils.sanitize_filename('_grouped_by_gene')}"))
            report_gen.add_plot("relative_dinucleotide_abundance_combined",
                get_plot_rel_path_for_report("relative_dinucleotide_abundance"))

            if not args.skip_ca and ca_results_combined:
                ca_suffix_combined = utils.sanitize_filename('_combined_by_gene')
                report_gen.add_plot("ca_biplot_combined",
                    get_plot_rel_path_for_report(f"ca_biplot_comp{args.ca_dims[0]+1}v{args.ca_dims[1]+1}_{ca_suffix_combined}"))
                report_gen.add_plot("ca_variance_explained",
                    get_plot_rel_path_for_report(f"ca_variance_explained_top{10}"))
                report_gen.add_plot("ca_contribution_dim1",
                    get_plot_rel_path_for_report(f"ca_contribution_dim1_top{10}"))
                report_gen.add_plot("ca_contribution_dim2",
                    get_plot_rel_path_for_report(f"ca_contribution_dim2_top{10}"))

            report_gen.add_plot("feature_correlation_heatmap",
                get_plot_rel_path_for_report(f"feature_correlation_heatmap_spearman"))
            ca_axes_corr_method_name_sanitized = utils.sanitize_filename("Spearman")
            report_gen.add_plot("ca_axes_feature_corr",
                get_plot_rel_path_for_report(f"ca_axes_feature_corr_{ca_axes_corr_method_name_sanitized}"))


            # 4. Add paths to per-gene RSCU boxplots
            if 'successfully_processed_genes' in locals() and successfully_processed_genes: # type: ignore
                for gene_name_for_plot in successfully_processed_genes: # type: ignore
                    plot_filename_base = f"RSCU_boxplot_{utils.sanitize_filename(gene_name_for_plot)}"
                    target_dict_rscu = report_gen.report_data["plot_paths"]["per_gene_rscu_boxplots"].setdefault(gene_name_for_plot, {})
                    report_gen.add_plot(
                        plot_key="rscu_boxplot",
                        plot_path_relative_to_outdir=get_plot_rel_path_for_report(plot_filename_base),
                        plot_dict_target=target_dict_rscu
                    )
            if "complete" in all_nucl_freqs_by_gene_agg: # type: ignore
                plot_filename_base_complete = "RSCU_boxplot_complete"
                target_dict_rscu_complete = report_gen.report_data["plot_paths"]["per_gene_rscu_boxplots"].setdefault("complete", {})
                report_gen.add_plot(
                    plot_key="rscu_boxplot",
                    plot_path_relative_to_outdir=get_plot_rel_path_for_report(plot_filename_base_complete),
                    plot_dict_target=target_dict_rscu_complete
                )

            # Add per-gene plots colored by metadata to report
            if args.color_by_metadata and metadata_df is not None and 'palette_categories_for_map' in locals(): # type: ignore
                meta_col = args.color_by_metadata
                report_gen.report_data["metadata_info"]["column_used_for_coloring"] = meta_col
                report_gen.report_data["metadata_info"]["categories_shown"] = palette_categories_for_map # type: ignore
                report_gen.report_data["metadata_info"]["other_category_used"] = "Other" in palette_categories_for_map # type: ignore

                meta_col_plot_data = report_gen.report_data["plot_paths"]["per_gene_metadata_plots"].setdefault(meta_col, {})
                # Base dir where these plots are saved: output_dir_path / "images" / SANITIZED_META_COL_per_gene_plots /
                meta_plots_image_subdir_name = f"{utils.sanitize_filename(meta_col)}_per_gene_plots"

                genes_with_metadata_plots = []
                if combined_per_sequence_df_with_meta is not None:
                    genes_with_metadata_plots = combined_per_sequence_df_with_meta['Gene'].unique()

                for gene_name_for_meta_plot in genes_with_metadata_plots:
                    gene_specific_meta_plot_subdir_name = utils.sanitize_filename(gene_name_for_meta_plot)
                    gene_plot_target_dict = meta_col_plot_data.setdefault(gene_name_for_meta_plot, {})
                    filename_suffix_meta_gene = f"_{utils.sanitize_filename(gene_name_for_meta_plot)}_by_{utils.sanitize_filename(meta_col)}"

                    plot_types_filename_map = {
                        "enc_vs_gc3": f"enc_vs_gc3_plot{filename_suffix_meta_gene}",
                        "neutrality": f"neutrality_plot{filename_suffix_meta_gene}",
                        "ca_biplot": f"ca_biplot_comp{args.ca_dims[0]+1}v{args.ca_dims[1]+1}{filename_suffix_meta_gene}",
                        "dinucl_abundance": f"dinucl_abundance{filename_suffix_meta_gene}"
                    }
                    for plot_type_key, filename_base in plot_types_filename_map.items():
                        # Construct the path relative to output_dir_root, like "images/METAPLOTS_SUBDIR/GENE_SUBDIR/plotname.fmt"
                        plot_rel_path = Path("images") / meta_plots_image_subdir_name / gene_specific_meta_plot_subdir_name / f"{filename_base}.{report_plot_format}"
                        # We need to check if this file actually exists
                        if (output_dir_path / plot_rel_path).exists():
                            report_gen.add_plot(
                                plot_key=plot_type_key,
                                plot_path_relative_to_outdir=str(plot_rel_path),
                                plot_dict_target=gene_plot_target_dict
                            )
                        else:
                            report_gen.add_plot(plot_key=plot_type_key, plot_path_relative_to_outdir=None, plot_dict_target=gene_plot_target_dict)
            else:
                report_gen.report_data["metadata_info"]["column_used_for_coloring"] = None
                report_gen.report_data["metadata_info"]["categories_shown"] = []
                report_gen.report_data["metadata_info"]["other_category_used"] = False

            report_gen.generate_report()
    else:
        logger.info("Skipping HTML report generation as per --no-html-report flag.")

    _create_output_readme(output_dir_path, args)

    logger.info("PyCodon Analyzer 'analyze' command finished successfully.")

def process_analyze_gene_file(
    gene_filepath: str,
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    expected_gene_names: Set[str],
    output_dir_path_for_plots: Path # ADDED: This is the main output_dir_path
) -> Optional[ProcessGeneFileResultType]:
    """
    Worker function for the 'analyze' subcommand.
    Saves its per-gene RSCU boxplot to output_dir_path_for_plots / "images" / ...
    """
    try:
        import matplotlib
        matplotlib.use('Agg') # type: ignore
    except ImportError: # pragma: no cover
        logging.getLogger("pycodon_analyzer.worker").error(f"[Worker {os.getpid()}] Matplotlib not found.")
    except Exception as backend_err: # pragma: no cover
        logging.getLogger("pycodon_analyzer.worker").warning(f"[Worker {os.getpid()}] Could not set Matplotlib backend: {backend_err}")

    worker_logger = logging.getLogger(f"pycodon_analyzer.worker.{os.getpid()}")
    gene_name: Optional[str] = extract_gene_name_from_file(gene_filepath)

    if not gene_name or gene_name not in expected_gene_names:
        return None

    worker_logger.debug(f"Processing gene: {gene_name} (File: {Path(gene_filepath).name})")
    per_sequence_df_gene: Optional[pd.DataFrame] = None
    ca_input_df_gene: Optional[pd.DataFrame] = None
    nucl_freqs_gene_agg: Optional[Dict[str, float]] = None
    dinucl_freqs_gene_agg: Optional[Dict[str, float]] = None
    cleaned_seq_map: Optional[Dict[str, Seq]] = None
    per_seq_nucl_freqs: Optional[Dict[str, Dict[str, float]]] = None
    per_seq_dinucl_freqs: Optional[Dict[str, Dict[str, float]]] = None

    try:
        raw_sequences: List[SeqRecord] = io.read_fasta(gene_filepath)
        if not raw_sequences:
            return (gene_name, "empty file", None, None, None, None, None, None, None)

        cleaned_sequences: List[SeqRecord] = utils.clean_and_filter_sequences(raw_sequences, 
                                                                              args.max_ambiguity)
        if not cleaned_sequences:
            return (gene_name, "no valid seqs", 
                    None, None, None, None, None, None, None)

        cleaned_seq_map = {rec.id: rec.seq for rec in cleaned_sequences if rec.id}

        analysis_results_tuple: FullAnalysisResultType = analysis.run_full_analysis(
            cleaned_sequences, args.genetic_code, reference_weights=reference_weights
        )
        agg_usage_df_gene, per_sequence_df_gene, nucl_freqs_gene_agg, dinucl_freqs_gene_agg, \
        per_seq_nucl_freqs, per_seq_dinucl_freqs, _, _, ca_input_df_gene = analysis_results_tuple

        worker_logger.debug(f"Core analysis complete for {gene_name}.")

        if not args.skip_plots and agg_usage_df_gene is not None and not agg_usage_df_gene.empty and \
           ca_input_df_gene is not None and not ca_input_df_gene.empty:
            try:
                long_rscu_df = ca_input_df_gene.reset_index().rename(columns={'index': 'SequenceID'})
                long_rscu_df = long_rscu_df.melt(id_vars=['SequenceID'], var_name='Codon', value_name='RSCU')
                current_genetic_code: Dict[str, str] = utils.get_genetic_code(args.genetic_code)
                long_rscu_df['AminoAcid'] = long_rscu_df['Codon'].map(current_genetic_code.get)
                for fmt in args.plot_formats:
                    # Construct full save path for the plot
                    plot_filename = f"RSCU_boxplot_{utils.sanitize_filename(gene_name)}.{fmt}"
                    # Plots are saved in output_dir_path_for_plots / "images"
                    rscu_boxplot_filepath = output_dir_path_for_plots / "images" / plot_filename
                    # The plotting function now takes the full filepath
                    plotting.plot_rscu_boxplot_per_gene(
                        long_rscu_df,
                        agg_usage_df_gene,
                        gene_name, # Still needed for plot title
                        str(rscu_boxplot_filepath) # Pass the full path
                    )
            except Exception as plot_prep_err: # pragma: no cover
                worker_logger.error(f"Failed to prepare/plot RSCU boxplot for {gene_name}: {plot_prep_err}")

        if per_sequence_df_gene is not None and not per_sequence_df_gene.empty:
            if 'ID' in per_sequence_df_gene.columns:
                 per_sequence_df_gene['Original_ID'] = per_sequence_df_gene['ID']
                 per_sequence_df_gene['ID'] = f"{gene_name}__" + per_sequence_df_gene['ID'].astype(str)
            per_sequence_df_gene['Gene'] = gene_name

        if ca_input_df_gene is not None and not ca_input_df_gene.empty:
            ca_input_df_gene.index = f"{gene_name}__" + ca_input_df_gene.index.astype(str)

        worker_logger.debug(f"Finished processing for gene: {gene_name}")
        return (gene_name, "success", per_sequence_df_gene, ca_input_df_gene,
                nucl_freqs_gene_agg, dinucl_freqs_gene_agg, cleaned_seq_map,
                per_seq_nucl_freqs, per_seq_dinucl_freqs)

    except FileNotFoundError: # pragma: no cover
        worker_logger.error(f"File not found error for gene {gene_name}: {gene_filepath}")
        return (gene_name, "file not found error", None, None, None, None, None, None, None)
    except ValueError as ve: # pragma: no cover
        worker_logger.error(f"ValueError processing gene {gene_name}: {ve}")
        return (gene_name, "value error", None, None, None, None, None, None, None)
    except Exception as e: # pragma: no cover
        worker_logger.exception(f"UNEXPECTED ERROR processing gene {gene_name} (file {Path(gene_filepath).name}): {e}")
        return (gene_name, "exception", None, None, None, None, None, None, None)

def handle_extract_command(args: argparse.Namespace) -> None:
    """Handles the 'extract' subcommand and its arguments."""
    logger.info(f"Running 'extract' command. Annotations: {args.annotations}, Alignment: {args.alignment}, Output: {args.output_dir}")

    # --- Validate input files before creating output directory ---
    if not args.annotations.is_file():
        logger.error(f"Annotation file not found: {args.annotations}. Exiting.")
        sys.exit(1)
    if not args.alignment.is_file():
        logger.error(f"Alignment file not found: {args.alignment}. Exiting.")
        sys.exit(1)
    
    # --- Deferred Output Directory Creation ---
    try:
        # args.output_dir is a Path object from argparse
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{args.output_dir}' created or already exists for extracted genes.")
    except OSError as e:
        logger.error(f"Error creating output directory '{args.output_dir}': {e}. Exiting.")
        sys.exit(1)

    try:
        # Call the main extraction function from the extraction module
        # Assuming extraction module is imported
        extraction.extract_gene_alignments_from_genome_msa(
            annotations_path=args.annotations,
            alignment_path=args.alignment,
            ref_id=args.ref_id,
            output_dir=args.output_dir
        )
        logger.info("'extract' command finished successfully.")
    except FileNotFoundError as fnf_err: # Should be caught by earlier checks, but defensive
        logger.error(f"Extraction error: {fnf_err}")
        sys.exit(1)
    except ValueError as val_err: # For parsing or data integrity errors from extraction module
        logger.error(f"Extraction error: {val_err}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during 'extract' command: {e}")
        sys.exit(1)

def main() -> None:
    """Main CLI entry point with subcommands."""
    # --- Main Parser ---
    parser = argparse.ArgumentParser(
        description="PyCodon Analyzer: Codon usage analysis & gene alignment extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Increase output verbosity (set logging to DEBUG)."
    )
    try:
        from . import __version__ as pkg_version
    except ImportError: # pragma: no cover
        pkg_version = "unknown"
    parser.add_argument('--version',
                        action='version',
                        version=f'%(prog)s {pkg_version}')

    subparsers = parser.add_subparsers(dest="command",
                                       required=True,
                                       title="Available subcommands",
                                       help="Run 'pycodon_analyzer <subcommand> --help' for more information.")

    # --- Sub-parser for 'analyze' command ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze codon usage from pre-extracted gene alignment files.",
        description="Performs codon usage analysis on a directory of individual gene FASTA alignments (gene_NAME.fasta format).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    analyze_parser.add_argument("-d",
                                "--directory",
                                required=True,
                                type=str, # Consider type=Path for direct Path object if preferred
                                help="Path to input directory with gene_*.fasta files.")
    analyze_parser.add_argument("-o",
                                "--output",
                                default="codon_analysis_results",
                                type=str, # Consider type=Path
                                help="Path to output directory for analysis results.")
    analyze_parser.add_argument("--genetic_code",
                                type=int, default=1,
                                help="NCBI genetic code ID.")
    analyze_parser.add_argument("--ref",
                                dest="reference_usage_file",
                                type=str,
                                default=DEFAULT_HUMAN_REF_PATH if DEFAULT_HUMAN_REF_PATH else "human",
                                help="Reference codon usage table ('human', 'none', or path).")
    # New argument for reference file delimiter, related to previous refactoring
    analyze_parser.add_argument("--ref_delimiter",
                                type=str,
                                default=None,
                                help="Delimiter for the reference codon usage file (e.g., ',', '\\t'). Auto-detects if not provided.")
    analyze_parser.add_argument("-t",
                                "--threads",
                                type=int,
                                default=1,
                                help="Number of processes for parallel gene file analysis (0 or negative for all available cores).")
    analyze_parser.add_argument("--max_ambiguity",
                                type=float,
                                default=15.0,
                                help="Max allowed 'N' percentage per sequence (0-100).")
    analyze_parser.add_argument("--plot_formats",
                                nargs='+',
                                default=['svg'],
                                choices=['svg', 'png', 'pdf', 'jpg'], # Added more choices
                                type=str.lower, # Convert to lowercase
                                help="Output format(s) for plots (e.g., svg png).")
    analyze_parser.add_argument("--skip_plots",
                                action='store_true',
                                help="Disable all plot generation.")
    analyze_parser.add_argument("--ca_dims",
                                nargs=2,
                                type=int,
                                default=[0, 1],
                                metavar=('X_DIM_IDX', 'Y_DIM_IDX'), # Clarified metavar
                                help="Indices of CA components for combined CA plot (0-based, e.g., 0 1 for Dim1 vs Dim2).")
    analyze_parser.add_argument("--skip_ca",
                                action='store_true',
                                help="Skip combined Correspondence Analysis.")
    # --- Arguments for metadata
    analyze_parser.add_argument(
        "--metadata",
        type=Path, # Use pathlib.Path for easier path handling
        default=None,
        help="Optional path to a metadata file (CSV or TSV) for sequences. "
             "The file should contain a column matching original sequence IDs."
    )
    analyze_parser.add_argument(
        "--metadata_id_col",
        type=str,
        default="seq_id", # Default column name for sequence IDs in the metadata file
        help="Name of the column in the metadata file that contains sequence identifiers. "
             "These IDs should match the original sequence IDs from your FASTA files "
             "(i.e., before any gene name prefixing by this tool)."
    )
    analyze_parser.add_argument(
        "--metadata_delimiter",
        type=str,
        default=None, # Auto-detect
        help="Delimiter for the metadata file (e.g., ',', '\\t'). If not provided, attempts to auto-detect."
    )
    analyze_parser.add_argument(
        "--color_by_metadata",
        type=str,
        default=None,
        metavar="METADATA_COLUMN_NAME",
        help="If --metadata is provided, specify a categorical column name from the metadata "
             "to use for coloring in newly generated per-gene plots. These plots will be "
             "saved in a dedicated subdirectory."
    )
    analyze_parser.add_argument(
        "--metadata_max_categories",
        type=int,
        default=15,
        help="When using --color_by_metadata, if the specified metadata column has more unique "
             "categories than this number, plots will only show the top N most frequent "
             "categories, and the rest will be grouped into an 'Other' category. (Default: 15)."
    )
    analyze_parser.add_argument(
        "--no-html-report",
        action="store_true", # Keeps action as store_true, default will be False (meaning report IS generated)
        default=False,       # Explicitly set default to False
        help="Disable generation of the HTML report."
    )
    # Example for a future feature related to metadata:
    # analyze_parser.add_argument(
    #     "--correlate_ca_with_metadata_cols",
    #     nargs='*', # 0 or more arguments
    #     default=[],
    #     metavar="NUMERIC_METADATA_COL",
    #     help="List of numeric columns from the metadata file to correlate with CA axes."
    # )
    analyze_parser.set_defaults(func=handle_analyze_command)


    # --- Sub-parser for 'extract' command ---
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract individual gene alignments from a whole genome MSA.",
        description="Extracts gene alignments using an annotation file (FASTA with GenBank-style tags) and a whole genome alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    extract_parser.add_argument("-a", 
                                "--annotations", 
                                type=Path, 
                                required=True, 
                                help="Path to reference gene annotation file (multi-FASTA with [gene=/locus_tag=] and [location=] tags).")
    extract_parser.add_argument("-g", 
                                "--alignment", 
                                type=Path, 
                                required=True, 
                                help="Path to whole genome multiple sequence alignment file (FASTA).")
    extract_parser.add_argument("-r", 
                                "--ref_id", 
                                type=str, 
                                required=True, 
                                help="Sequence ID of reference genome in the alignment file.")
    extract_parser.add_argument("-o", 
                                "--output_dir", 
                                type=Path, 
                                required=True, 
                                help="Output directory for extracted gene_*.fasta files.")
    extract_parser.set_defaults(func=handle_extract_command)

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Configure Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if RICH_AVAILABLE: # pragma: no cover
        handler_to_use: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=True, show_level=True, log_time_format="[%X]"
        )
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        handler_to_use.setFormatter(formatter)
    else: # pragma: no cover
        handler_to_use = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler_to_use.setFormatter(formatter)

    # Configure the application's specific logger
    app_logger = logging.getLogger("pycodon_analyzer") # Use the consistent logger name
    if app_logger.hasHandlers(): # pragma: no cover
        app_logger.handlers.clear()
    app_logger.addHandler(handler_to_use)
    app_logger.setLevel(log_level)
    # app_logger.propagate = False # Optional: if you don't want messages to go to root logger


    logger.info(f"PyCodon Analyzer - Command: {args.command}") # Use the module-level logger for this message too
    if args.verbose:
        logger.debug(f"Full arguments: {args}")

    # --- Execute the function associated with the subcommand ---
    if hasattr(args, 'func'):
        args.func(args)
    else: # pragma: no cover
        parser.print_help()

if __name__ == '__main__': # pragma: no cover
    main()