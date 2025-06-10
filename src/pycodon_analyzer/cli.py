# src/pycodon_analyzer/cli.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Command-Line Interface for PyCodon Analyzer.

This script provides the main entry point for the PyCodon Analyzer tool,
offering two primary subcommands:
1.  'analyze': Processes a directory of gene alignment FASTA files to calculate
               codon usage metrics, sequence properties, and generate an HTML report.
               Can be enhanced with metadata for stratified analysis and plotting.
2.  'extract': Extracts individual gene alignments from a whole genome multiple
               sequence alignment (MSA) using gene annotations from a reference file.

The tool utilizes Python's standard logging module (optionally enhanced by 'rich')
for progress and error reporting. It supports parallel processing for the 'analyze'
subcommand to speed up the analysis of multiple gene files.
"""

# Standard library imports
import argparse
import glob
import logging
import os
import re
import sys
import traceback
from logging.handlers import RotatingFileHandler
import csv
from pathlib import Path
from typing import (
    List, Dict, Optional, Tuple, Set, Any, Counter as TypingCounter, TYPE_CHECKING
)

# Third-party library imports
import pandas as pd
import numpy as np
import seaborn as sns
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Optional dependency: rich (for progress bars and enhanced logging)
try:
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    )
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy classes for type hinting and fallback when rich is not available
    Progress = Any
    SpinnerColumn = Any
    BarColumn = Any
    TextColumn = Any
    TimeElapsedColumn = Any
    TimeRemainingColumn = Any
    RichHandler = logging.StreamHandler # Fallback to standard stream handler
    # Logging a warning here might be too early if logger isn't configured.
    # It's handled during logger setup in main().

# Optional dependency: scipy (for statistical tests)
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None
    SCIPY_AVAILABLE = False
    # Warning about missing scipy will be logged if statistical tests are attempted.

# Optional dependency: prince (for Correspondence Analysis)
if TYPE_CHECKING:
    import prince
    PrinceCA = prince.CA
    PRINCE_AVAILABLE = True
else:
    PrinceCA = Any
    try:
        import prince
        PRINCE_AVAILABLE = True
    except ImportError:
        prince = None # Will be checked at runtime if CA is attempted.
        PRINCE_AVAILABLE = False

# Local module imports
from . import io
from . import analysis
from . import plotting
from . import utils
from . import extraction
from . import reporting
from .utils import load_reference_usage, get_genetic_code, clean_and_filter_sequences
from .analysis import PrinceCA, FullAnalysisResultType # For type hinting

# Optional dependency: multiprocessing
try:
    import multiprocessing as mp
    from functools import partial
    MP_AVAILABLE = True
except ImportError:
    mp = None
    partial = None
    MP_AVAILABLE = False
    # Warning about missing multiprocessing will be logged if parallel processing is requested.

# ------------------------------------------------------------------------------
# GLOBAL CONFIGURATION & LOGGER SETUP
# ------------------------------------------------------------------------------

# Primary logger for the application
logger = logging.getLogger("pycodon_analyzer")

# Default path for human codon usage reference data (bundled with the package)
try:
    from importlib.resources import files as pkg_resources_files
except ImportError: # Fallback for Python < 3.9
    try:
        from importlib_resources import files as pkg_resources_files # type: ignore
    except ImportError:
        pkg_resources_files = None

DEFAULT_REF_FILENAME = "human_codon_usage.csv"
DEFAULT_HUMAN_REF_PATH: Optional[str] = None

if pkg_resources_files:
    try:
        # Attempt to locate the bundled data file
        ref_path_obj = pkg_resources_files('pycodon_analyzer').joinpath('data').joinpath(DEFAULT_REF_FILENAME)
        if ref_path_obj.is_file():
            DEFAULT_HUMAN_REF_PATH = str(ref_path_obj)
    except Exception:
        # Errors finding the path will be handled if 'human' reference is requested
        # and this path remains None.
        pass

# ------------------------------------------------------------------------------
# TYPE ALIASES
# ------------------------------------------------------------------------------

AnalyzeGeneResultType = Tuple[
    Optional[str],                  # gene_name
    str,                            # status ("success", "empty file", "no valid seqs", "exception")
    Optional[pd.DataFrame],         # per_sequence_df_gene
    Optional[pd.DataFrame],         # ca_input_df_gene (RSCU wide for this gene)
    Optional[Dict[str, float]],     # nucleotide_frequencies_gene_aggregate
    Optional[Dict[str, float]],     # dinucleotide_frequencies_gene_aggregate
    Optional[Dict[str, Seq]]        # cleaned_sequences_map {original_id: Bio.Seq}
]

ProcessGeneFileResultType = Tuple[
    Optional[str],                          # gene_name
    str,                                    # status
    Optional[pd.DataFrame],                 # per_sequence_df_gene
    Optional[pd.DataFrame],                 # ca_input_df_gene
    Optional[Dict[str, float]],             # nucl_freqs_gene_agg
    Optional[Dict[str, float]],             # dinucl_freqs_gene_agg
    Optional[Dict[str, Seq]],               # cleaned_seq_map
    Optional[Dict[str, Dict[str, float]]],  # per_sequence_nucl_freqs {seq_id: {nucl: freq}}
    Optional[Dict[str, Dict[str, float]]]   # per_sequence_dinucl_freqs {seq_id: {dinucl: freq}}
]

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS FOR 'analyze' COMMAND
# ------------------------------------------------------------------------------

def _ensure_output_subdirectories(output_dir_path: Path) -> None:
    """
    Creates 'data' and 'images' subdirectories within the main output path.
    The 'html' subdirectory for the report is handled by HTMLReportGenerator.

    Args:
        output_dir_path: Path object for the main output directory.

    Raises:
        SystemExit: If directory creation fails.
    """
    subdirs_to_create = ["data", "images"]
    try:
        for subdir_name in subdirs_to_create:
            (output_dir_path / subdir_name).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured '{', '.join(subdirs_to_create)}' subdirectories exist in '{output_dir_path}'.")
    except OSError as e:
        logger.error(f"Error creating output subdirectories in '{output_dir_path}': {e}. Exiting.")
        sys.exit(1)


def _create_output_readme(output_dir_path: Path, args: argparse.Namespace) -> None:
    """
    Creates a README.txt file in the output directory explaining the structure and parameters.

    Args:
        output_dir_path: Path object for the main output directory.
        args: Parsed command-line arguments from argparse.
    """
    # Dynamic content based on arguments
    metadata_file_info = f"  - Metadata File: {args.metadata if args.metadata else 'Not provided'}"
    metadata_id_col_info = f"\n  - Metadata ID Column: {args.metadata_id_col}" if args.metadata else ""
    color_by_metadata_info = f"\n  - Plots Colored by Metadata Column: {args.color_by_metadata}" if args.color_by_metadata else ""
    html_report_info = f"  - HTML Report Generated: {'No' if args.no_html_report else 'Yes'}"
    ref_usage_file_display = args.reference_usage_file
    if args.reference_usage_file == DEFAULT_HUMAN_REF_PATH:
        ref_usage_file_display = "human (bundled)"
    elif not args.reference_usage_file or args.reference_usage_file.lower() == 'none':
        ref_usage_file_display = 'None'


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
  - Reference Usage File: {ref_usage_file_display}
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

1.  `report.html` (if generated)
    The main interactive HTML report. It provides an overview, run parameters,
    and navigation to all detailed sections.

2.  `data/` (Subdirectory)
    Contains all data tables generated during the analysis (primarily CSV format).
    Key files:
    * `per_sequence_metrics_all_genes.csv`: Comprehensive metrics for each valid sequence.
                                           If metadata provided, it's merged here.
    * `mean_features_per_gene.csv`: Average values for key metrics per gene.
    * `gene_comparison_stats.csv`: Results of statistical tests between genes.
    * `per_sequence_rscu_wide.csv`: RSCU values (wide format) for combined CA.
    * `ca_*.csv`: Files related to combined CA (coordinates, contributions, eigenvalues).
    * `gene_sequence_summary.csv`: Sequence counts and length stats per gene.

3.  `images/` (Subdirectory, if plots not skipped)
    Contains all plot images.
    * Combined Plots: Directly in `images/` (e.g., overall GC, ENC vs GC3, combined CA).
    * Per-Gene RSCU Boxplots: e.g., `RSCU_boxplot_GENENAME.<fmt>`.
    * Metadata-Specific Plots (if --color_by_metadata):
      Organized into `images/<METADATA_COLUMN_NAME>_per_gene_plots/<GENE_NAME>/`.
      Includes ENC vs GC3, Neutrality, CA biplots, Dinucleotide abundance plots,
      all colored by metadata categories.

4.  `html/` (Subdirectory, if HTML report generated)
    Contains secondary HTML pages for the interactive report.

5.  `{args.log_file}` (in this directory)
    The detailed log file for this analysis run.

----------------------------------------
How to Navigate the Results
----------------------------------------
1.  Start with `report.html` (if generated) in a web browser.
2.  Explore CSV files in `data/` for detailed data tables.
3.  View plots in `images/` (also embedded in the HTML report).

---
This output was generated by PyCodon Analyzer on {pd.Timestamp("now").strftime('%Y-%m-%d %H:%M:%S')}.
For support or issues, please visit the project repository.
"""
    try:
        readme_path = output_dir_path / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        logger.info(f"Output directory README.txt created at: {readme_path}")
    except IOError as e:
        logger.error(f"Failed to write output README.txt: {e}")


def extract_gene_name_from_file(filename: str) -> Optional[str]:
    """
    Extracts a gene name from a filename.
    Assumes filenames like 'gene_XYZ.fasta' or 'gene_ABC.fa'.

    Args:
        filename: The full path or basename of the file.

    Returns:
        The extracted gene name (e.g., 'XYZ') or None if the pattern doesn't match.
    """
    base = os.path.basename(filename)
    # Regex to match 'gene_' prefix, capture the gene name part,
    # and look for common FASTA extensions. Case-insensitive.
    match = re.match(r'gene_([\w\-.]+)\.(fasta|fa|fna|fas|faa)$', base, re.IGNORECASE)
    if match:
        return match.group(1)  # The captured gene name
    logger.debug(f"Could not extract gene name from filename: {base}")
    return None


def _setup_output_directory(output_path_str: str) -> Path:
    """
    Ensures the output directory exists, creating it if necessary.

    Args:
        output_path_str: The desired path for the output directory as a string.

    Returns:
        A Path object representing the output directory.

    Raises:
        SystemExit: If the directory cannot be created due to an OSError.
    """
    output_dir_path = Path(output_path_str)
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{output_dir_path}' is ready.")
        return output_dir_path
    except OSError as e:
        logger.error(f"Fatal: Error creating output directory '{output_dir_path}': {e}. Exiting.")
        sys.exit(1)


def _load_reference_data(args: argparse.Namespace) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """
    Loads reference codon usage data based on command-line arguments.
    This data is used for calculating reference-dependent metrics like CAI.

    Args:
        args: Parsed command-line arguments. Expected attributes include
              `reference_usage_file`, `genetic_code`, and `ref_delimiter`.

    Returns:
        A tuple containing:
        - A dictionary mapping codons to reference weights (for CAI, etc.), or None.
        - A DataFrame of the loaded reference data (for plotting), or None.

    Raises:
        SystemExit: If a specified reference file cannot be found or loaded.
    """
    reference_weights: Optional[Dict[str, float]] = None
    reference_data_for_plot: Optional[pd.DataFrame] = None

    # Determine path to reference file
    ref_file_arg = args.reference_usage_file
    ref_path_to_load: Optional[str] = None

    if ref_file_arg and ref_file_arg.lower() != 'none':
        if ref_file_arg.lower() == 'human':
            if DEFAULT_HUMAN_REF_PATH and os.path.isfile(DEFAULT_HUMAN_REF_PATH):
                ref_path_to_load = DEFAULT_HUMAN_REF_PATH
                logger.info("Using bundled 'human' reference codon usage table.")
            else:
                logger.error("Bundled 'human' reference file not found. Exiting.")
                sys.exit(1)
        elif os.path.isfile(ref_file_arg):
            ref_path_to_load = ref_file_arg
            logger.info(f"Using user-specified reference file: {ref_path_to_load}")
        else:
            logger.error(f"Specified reference file not found: {ref_file_arg}. Exiting.")
            sys.exit(1)

        # Load and process the reference file
        if ref_path_to_load:
            logger.info(f"Loading codon usage reference table from: {Path(ref_path_to_load).name}...")
            try:
                current_genetic_code = utils.get_genetic_code(args.genetic_code)
                reference_data_for_plot = utils.load_reference_usage(
                    filepath=ref_path_to_load,
                    genetic_code=current_genetic_code,
                    genetic_code_id=args.genetic_code,
                    delimiter=args.ref_delimiter  # Pass user-specified or auto-detected delimiter
                )

                if reference_data_for_plot is not None and 'Weight' in reference_data_for_plot.columns:
                    reference_weights = reference_data_for_plot['Weight'].to_dict()
                    logger.info("Reference data loaded and weights successfully extracted.")
                elif reference_data_for_plot is not None:
                    logger.error("Fatal: 'Weight' column crucial for CAI calculation is missing from the loaded reference data. Exiting.")
                    sys.exit(1)
                else:
                    # load_reference_usage should log specific errors.
                    logger.error(f"Fatal: Failed to load or process reference data from {Path(ref_path_to_load).name}. Exiting.")
                    sys.exit(1)
            except Exception as e:
                logger.exception(f"Fatal: Unexpected error loading reference file {Path(ref_path_to_load).name}: {e}. Exiting.")
                sys.exit(1)
    else:
        logger.info("No reference file specified ('none' or not provided). "
                    "Reference-based metrics (CAI, Fop, RCDI) will be NaN.")

    return reference_weights, reference_data_for_plot


def _load_metadata(
    metadata_path_arg: Optional[str], # Changed to Optional[str] to match argparse
    id_col_name: str,
    delimiter: Optional[str]
) -> Optional[pd.DataFrame]:
    """
    Loads and validates the metadata file (CSV or TSV).

    Args:
        metadata_path_arg: Path to the metadata file as a string, or None.
        id_col_name: Name of the column containing sequence identifiers.
        delimiter: Specified delimiter for the file. If None, attempts to auto-detect.

    Returns:
        DataFrame with metadata, indexed by id_col_name, or None if loading fails.
    """
    if not metadata_path_arg:
        logger.debug("No metadata file path provided.")
        return None

    metadata_path = Path(metadata_path_arg) # Convert to Path object

    if not metadata_path.is_file():
        logger.error(f"Metadata file not found: {metadata_path}. Skipping metadata integration.")
        return None

    meta_df: Optional[pd.DataFrame] = None
    file_basename = metadata_path.name
    logger.info(f"Loading metadata from: {file_basename}")

    try:
        dtype_spec = {id_col_name: str} # Ensure ID column is read as string

        if delimiter:
            logger.debug(f"Attempting to read metadata file '{file_basename}' with specified delimiter: '{delimiter}'")
            meta_df = pd.read_csv(metadata_path, sep=delimiter, comment='#', dtype=dtype_spec)
        else:
            logger.debug(f"Attempting to auto-detect delimiter for metadata file '{file_basename}'...")
            try:
                with open(metadata_path, 'r', newline='', encoding='utf-8') as csvfile:
                    sample = csvfile.read(2048)
                    if not sample.strip():
                        logger.error(f"Metadata file '{file_basename}' appears to be empty or contains only whitespace.")
                        return None
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=',\t;| ') # Added space to common delimiters
                    detected_delimiter = dialect.delimiter
                    logger.info(f"Sniffed delimiter '{detected_delimiter}' for metadata file '{file_basename}'.")
                    meta_df = pd.read_csv(metadata_path, sep=detected_delimiter, comment='#', dtype=dtype_spec)
            except (csv.Error, pd.errors.ParserError, UnicodeDecodeError) as sniff_err:
                logger.warning(f"Could not reliably sniff delimiter or parse metadata file '{file_basename}': {sniff_err}. "
                               "Falling back to trying common delimiters.")
                fallback_delimiters = ['\t', ',', ';'] # Common fallbacks
                for delim_try in fallback_delimiters:
                    logger.debug(f"Fallback: Trying delimiter '{delim_try}' for '{file_basename}'.")
                    try:
                        meta_df = pd.read_csv(metadata_path, sep=delim_try, comment='#', dtype=dtype_spec)
                        if meta_df.shape[1] > 0 : # Basic check for successful parsing
                            logger.info(f"Successfully read metadata file '{file_basename}' with fallback delimiter: '{delim_try}'.")
                            break
                        meta_df = None # Reset if parse was not good
                    except Exception:
                        meta_df = None # Try next delimiter
                if meta_df is None:
                    logger.error(f"Failed to parse metadata file '{file_basename}' with any common fallback delimiter.")
                    return None

        if meta_df is None or meta_df.empty:
            logger.error(f"Metadata file '{file_basename}' could not be loaded or is empty after attempts.")
            return None

        # Validate DataFrame content
        if id_col_name not in meta_df.columns:
            logger.error(f"Metadata ID column '{id_col_name}' not found in '{file_basename}'. "
                         f"Available columns: {meta_df.columns.tolist()}. Skipping metadata integration.")
            return None

        meta_df[id_col_name] = meta_df[id_col_name].astype(str).str.strip() # Ensure string and strip whitespace
        meta_df.dropna(subset=[id_col_name], inplace=True) # Remove rows where ID is NaN after stripping

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
        logger.error(f"Metadata file '{file_basename}' is empty (pandas error).")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading metadata from '{file_basename}': {e}")
        return None


def _get_gene_files_and_names(directory_str: str) -> Tuple[List[str], Set[str]]:
    """
    Finds gene alignment files in the specified directory and extracts their names.
    Filters for common FASTA extensions and expects 'gene_NAME.ext' format.

    Args:
        directory_str: Path to the directory containing gene alignment files.

    Returns:
        A tuple containing:
        - A sorted list of full file paths to the gene alignment files.
        - A set of unique, extracted gene names.

    Raises:
        SystemExit: If no gene files are found or if no valid gene names can be
                    extracted from the filenames.
    """
    logger.info(f"Searching for gene alignment files (gene_*.<fasta_extension>) in: {directory_str}")
    # Using Path.glob for more robust path handling
    input_dir = Path(directory_str)
    if not input_dir.is_dir():
        logger.error(f"Input directory '{directory_str}' does not exist or is not a directory. Exiting.")
        sys.exit(1)

    # Common FASTA extensions
    valid_extensions: Set[str] = {".fasta", ".fa", ".fna", ".fas", ".faa"}
    # Search for files matching the pattern 'gene_*' with any extension first
    potential_files = list(input_dir.glob("gene_*.*"))
    # Filter by valid FASTA extensions
    gene_files_paths: List[Path] = sorted([p for p in potential_files if p.suffix.lower() in valid_extensions])
    gene_files_str: List[str] = [str(p) for p in gene_files_paths]


    if not gene_files_str:
        logger.error(f"No gene alignment files matching 'gene_*' with extensions "
                     f"({', '.join(valid_extensions)}) found in directory: {directory_str}. Exiting.")
        sys.exit(1)

    # Extract gene names from the found files
    extracted_gene_names: Set[str] = set()
    for fpath_str in gene_files_str:
        gene_name = extract_gene_name_from_file(fpath_str)
        if gene_name:
            extracted_gene_names.add(gene_name)
        else:
            logger.warning(f"Could not extract gene name from file: {Path(fpath_str).name}. Skipping this file.")

    if not extracted_gene_names:
        logger.error("No valid gene names could be extracted from the found FASTA files. "
                     "Ensure filenames follow 'gene_GENENAME.fasta' (or similar extension) format. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(gene_files_str)} potential gene files corresponding to {len(extracted_gene_names)} unique gene names.")
    return gene_files_str, extracted_gene_names


def _determine_num_processes(requested_threads: int, num_gene_files: int) -> int:
    """
    Determines the optimal number of processes for parallel analysis.

    Args:
        requested_threads: Number of threads/processes requested by the user.
                           0 or negative values indicate auto-detection (all available cores).
        num_gene_files: The total number of gene files to be processed.

    Returns:
        The actual number of processes to use, capped by available cores (if auto)
        and the number of files. Returns 1 if multiprocessing is not available.
    """
    num_processes = requested_threads

    if num_processes <= 0: # Auto-detect
        if MP_AVAILABLE and mp is not None:
            try:
                available_cores = os.cpu_count()
                num_processes = available_cores if available_cores else 1
                logger.info(f"Auto-detected {num_processes} available CPU cores.")
            except NotImplementedError:
                num_processes = 1
                logger.warning("Could not auto-detect CPU count. Defaulting to 1 process.")
        else:
            num_processes = 1 # Fallback if MP not available
    elif num_processes > 1 and not (MP_AVAILABLE and mp is not None):
        logger.warning("Multiprocessing requested but 'multiprocessing' module is not available. "
                       "Defaulting to 1 process.")
        num_processes = 1

    # Cap the number of processes by the number of files to analyze
    actual_num_processes = min(num_processes, num_gene_files)

    if actual_num_processes < num_processes and num_processes > 1:
        logger.info(f"Adjusted number of processes from {num_processes} to {actual_num_processes} "
                    "(limited by number of files or available cores).")

    logger.info(f"Using {actual_num_processes} process(es) for gene file analysis.")
    return actual_num_processes


def _run_gene_file_analysis_in_parallel(
    gene_files: List[str],
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    expected_gene_names: Set[str], # Not directly used by partial, but for context
    num_processes: int,
    output_dir_path: Path # Passed to worker for plot saving
) -> List[Optional[ProcessGeneFileResultType]]:
    """
    Manages the parallel or sequential execution of `process_analyze_gene_file`.

    Args:
        gene_files: List of paths to gene files to analyze.
        args: Parsed command-line arguments.
        reference_weights: Pre-loaded reference codon weights.
        expected_gene_names: Set of expected gene names (for worker context, not directly used here).
        num_processes: Number of worker processes to use.
        output_dir_path: Main output directory path for worker functions.

    Returns:
        A list of results from `process_analyze_gene_file` for each gene file.
        Each item can be a ProcessGeneFileResultType or None if processing failed.
    """
    # Create a partial function with fixed arguments for the worker
    processing_task = partial(
        process_analyze_gene_file,
        args=args,
        reference_weights=reference_weights,
        expected_gene_names=expected_gene_names, # Passed for context if worker needs it
        output_dir_path_for_plots=output_dir_path # Pass the main output dir
    )
    results_raw: List[Optional[ProcessGeneFileResultType]] = []

    # Setup progress bar (rich or basic fallback)
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed} of {task.total} genes)"),
        TimeElapsedColumn(), TextColumn("<"), TimeRemainingColumn()
    ]
    # Disable rich progress bar if stdout is not a TTY or rich is unavailable
    disable_rich_progress = not sys.stderr.isatty() or not RICH_AVAILABLE

    with Progress(*progress_columns, transient=False, disable=disable_rich_progress) as progress:
        analysis_task_id = progress.add_task("Analyzing Gene Files", total=len(gene_files))

        if num_processes > 1 and MP_AVAILABLE and mp is not None and partial is not None:
            logger.info(f"Starting parallel analysis with {num_processes} processes...")
            try:
                with mp.Pool(processes=num_processes) as pool:
                    # Using imap to get results as they complete, allows progress update
                    for result in pool.imap(processing_task, gene_files):
                        results_raw.append(result)
                        progress.update(analysis_task_id, advance=1)
                logger.info("Parallel analysis finished.")
            except Exception as pool_err: # pragma: no cover
                logger.exception(f"Error during parallel analysis pool execution: {pool_err}. "
                                 "No results were collected from the parallel run. Check worker logs.")
                # Fill with None to match expected output length
                results_raw = [None] * len(gene_files)
        else:
            logger.info("Starting sequential analysis (multiprocessing not used or num_processes=1).")
            for gene_file_path in gene_files:
                result = processing_task(gene_file_path)
                results_raw.append(result)
                progress.update(analysis_task_id, advance=1)
            logger.info("Sequential analysis finished.")
    return results_raw


def _collect_and_aggregate_results(
    analyze_results_raw: List[Optional[ProcessGeneFileResultType]],
    expected_gene_names: Set[str]
) -> Tuple[
    List[pd.DataFrame],                     # all_per_sequence_dfs
    Dict[str, pd.DataFrame],                # all_ca_input_dfs (RSCU wide by gene)
    Set[str],                               # successfully_processed_genes
    List[str],                              # failed_genes_info
    Dict[str, Dict[str, Seq]],              # sequences_by_original_id {orig_id: {gene_name: Bio.Seq}}
    Dict[str, Dict[str, float]],            # all_nucl_freqs_by_gene_agg {gene_name: {nucl: freq}}
    Dict[str, Dict[str, float]],            # all_dinucl_freqs_by_gene_agg {gene_name: {dinucl: freq}}
    Dict[str, Dict[str, Dict[str, float]]], # all_nucl_freqs_per_seq_in_gene {gene: {seq_id: {nucl: freq}}}
    Dict[str, Dict[str, Dict[str, float]]]  # all_dinucl_freqs_per_seq_in_gene {gene: {seq_id: {dinucl: freq}}}
]:
    """
    Collects and aggregates results from individual gene file analyses.
    This includes per-sequence metrics, CA inputs, and frequency data.

    Args:
        analyze_results_raw: List of results from `_run_gene_file_analysis_in_parallel`.
        expected_gene_names: Set of gene names that were expected to be processed.

    Returns:
        A tuple containing various aggregated data structures.
    """
    all_per_sequence_dfs: List[pd.DataFrame] = []
    all_ca_input_dfs: Dict[str, pd.DataFrame] = {} # {gene_name: RSCU_wide_df}
    successfully_processed_genes: Set[str] = set()
    failed_genes_info: List[str] = []
    sequences_by_original_id: Dict[str, Dict[str, Seq]] = {}
    all_nucl_freqs_by_gene_agg: Dict[str, Dict[str, float]] = {}
    all_dinucl_freqs_by_gene_agg: Dict[str, Dict[str, float]] = {}
    all_nucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_dinucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]] = {}

    logger.info("Collecting and aggregating analysis results from gene processing...")
    for result in analyze_results_raw:
        if result is None:
            # This might happen if the pool.imap encountered an error for a task
            # or if the worker function explicitly returned None before full tuple.
            logger.warning("Encountered a None result in raw analysis output. Skipping this entry.")
            continue

        # Unpack all items from ProcessGeneFileResultType
        (gene_name_res, status, per_seq_df, ca_input_df_gene,
         nucl_freqs_agg, dinucl_freqs_agg, cleaned_map,
         per_seq_nucl_f, per_seq_dinucl_f) = result

        if gene_name_res is None: # Should ideally not happen if worker returns gene_name
            logger.warning(f"Result received with no gene name (status: {status}). Skipping.")
            continue

        if status == "success":
            successfully_processed_genes.add(gene_name_res)
            if per_seq_df is not None and not per_seq_df.empty:
                all_per_sequence_dfs.append(per_seq_df)
            if ca_input_df_gene is not None and not ca_input_df_gene.empty:
                all_ca_input_dfs[gene_name_res] = ca_input_df_gene
            if nucl_freqs_agg:
                all_nucl_freqs_by_gene_agg[gene_name_res] = nucl_freqs_agg
            if dinucl_freqs_agg:
                all_dinucl_freqs_by_gene_agg[gene_name_res] = dinucl_freqs_agg

            if cleaned_map: # {original_id: Bio.Seq object}
                for seq_id, seq_obj in cleaned_map.items():
                    # sequences_by_original_id structure: {original_id: {gene_name1: seq1, gene_name2: seq2}}
                    sequences_by_original_id.setdefault(seq_id, {})[gene_name_res] = seq_obj

            # Store per-sequence frequencies, nested by gene name
            if per_seq_nucl_f: # Dict[seq_id, Dict[nucl, freq]]
                all_nucl_freqs_per_seq_in_gene[gene_name_res] = per_seq_nucl_f
            if per_seq_dinucl_f: # Dict[seq_id, Dict[dinucl, freq]]
                all_dinucl_freqs_per_seq_in_gene[gene_name_res] = per_seq_dinucl_f
        else:
            failed_genes_info.append(f"{gene_name_res} (status: {status})")
            logger.warning(f"Gene '{gene_name_res}' processing status: {status}.")


    # Log summary of processed/failed genes
    if not successfully_processed_genes:
        logger.error("No genes were successfully processed. Cannot continue analysis. Exiting.")
        sys.exit(1)

    if len(successfully_processed_genes) < len(expected_gene_names):
        genes_not_processed_successfully = expected_gene_names - successfully_processed_genes
        logger.warning(
            f"Successfully processed {len(successfully_processed_genes)} out of "
            f"{len(expected_gene_names)} expected genes. "
            f"Genes not successfully processed: {', '.join(sorted(list(genes_not_processed_successfully)))}."
        )
        if failed_genes_info:
            logger.warning(f"  Details for failed/skipped genes: {'; '.join(failed_genes_info)}")
    else:
        logger.info(f"All {len(successfully_processed_genes)} expected genes were processed.")

    return (all_per_sequence_dfs, all_ca_input_dfs, successfully_processed_genes,
            failed_genes_info, sequences_by_original_id,
            all_nucl_freqs_by_gene_agg, all_dinucl_freqs_by_gene_agg,
            all_nucl_freqs_per_seq_in_gene, all_dinucl_freqs_per_seq_in_gene)


def _analyze_complete_sequences_cli(
    sequences_by_original_id: Dict[str, Dict[str, Seq]],
    successfully_processed_genes: Set[str],
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    output_dir_path: Path # Main output directory (for plot saving context)
) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, float]],
    Optional[Dict[str, float]], Optional[Dict[str, Dict[str, float]]],
    Optional[Dict[str, Dict[str, float]]], Optional[pd.DataFrame]
]:
    """
    Analyzes concatenated 'complete' sequences.
    A 'complete' sequence is formed by concatenating all successfully processed
    genes for a given original sequence ID.

    Args:
        sequences_by_original_id: Nested dict {original_id: {gene_name: Bio.Seq}}.
        successfully_processed_genes: Set of gene names that were reliably processed.
        args: Parsed command-line arguments.
        reference_weights: Pre-loaded reference codon weights.
        output_dir_path: Main output directory path, used for saving plots.

    Returns:
        A tuple containing analysis results for the 'complete' sequences, similar
        to `FullAnalysisResultType` but specific to the concatenated set.
    """
    logger.info("Analyzing concatenated 'complete' sequences...")
    complete_seq_records: List[SeqRecord] = []
    max_ambiguity_pct_complete = args.max_ambiguity # Use the same threshold as individual genes

    if not successfully_processed_genes:
        logger.warning("No genes were successfully processed. Cannot create 'complete' sequences.")
        return None, None, None, None, None, None, None

    for original_id, gene_seq_map in sequences_by_original_id.items():
        # Check if this original_id has sequences for ALL successfully processed genes
        if set(gene_seq_map.keys()) == successfully_processed_genes:
            try:
                # Concatenate sequences in a defined order (e.g., sorted gene names)
                concat_str = "".join(str(gene_seq_map[g_name]) for g_name in sorted(list(successfully_processed_genes)))

                if not concat_str: # Should not happen if individual gene seqs are valid
                    logger.warning(f"Concatenated sequence for ID {original_id} is empty. Skipping.")
                    continue
                if len(concat_str) % 3 != 0:
                    logger.warning(f"Concatenated sequence for ID {original_id} has length {len(concat_str)} "
                                   "(not multiple of 3). Skipping.")
                    continue

                # Check ambiguity of the concatenated sequence
                n_count = concat_str.count('N')
                seq_len = len(concat_str)
                ambiguity = (n_count / seq_len) * 100 if seq_len > 0 else 0.0
                if ambiguity <= max_ambiguity_pct_complete:
                    complete_seq_records.append(
                        SeqRecord(Seq(concat_str),
                                  id=original_id, # Keep original ID
                                  description=f"Concatenated {len(gene_seq_map)} successfully processed genes")
                    )
                else:
                    logger.debug(f"Concatenated sequence for ID {original_id} excluded due to high ambiguity "
                                 f"({ambiguity:.1f}% > {max_ambiguity_pct_complete}%).")
            except Exception as e: # pragma: no cover (should be rare)
                logger.warning(f"Error concatenating or processing sequence for ID {original_id}: {e}")
        else:
            logger.debug(f"Skipping ID {original_id} for 'complete' sequence analysis as it's missing one or more successfully processed genes.")


    if not complete_seq_records:
        logger.info("No valid 'complete' sequences could be constructed for analysis.")
        return None, None, None, None, None, None, None

    logger.info(f"Running analysis on {len(complete_seq_records)} 'complete' (concatenated) sequence records...")
    try:
        # Use the same run_full_analysis function
        res_comp: FullAnalysisResultType = analysis.run_full_analysis(
            complete_seq_records, args.genetic_code, reference_weights
        )
        # Unpack results from FullAnalysisResultType
        (agg_usage_df_complete, per_seq_df_complete, nucl_freqs_complete_agg,
         dinucl_freqs_complete_agg, per_seq_nucl_freqs_complete,
         per_seq_dinucl_freqs_complete, _, _, ca_input_df_complete_for_plot) = res_comp # Adjusted unpacking

        # Save RSCU boxplot for "complete" set if not skipping plots
        if not args.skip_plots and \
           agg_usage_df_complete is not None and not agg_usage_df_complete.empty and \
           ca_input_df_complete_for_plot is not None and not ca_input_df_complete_for_plot.empty:
            logger.info("Generating RSCU boxplot for 'complete' concatenated data...")
            try:
                # Prepare long format RSCU data for boxplot
                long_rscu_df_comp = ca_input_df_complete_for_plot.reset_index().rename(
                    columns={'index': 'SequenceID'} # Assuming index is original_id
                )
                long_rscu_df_comp = long_rscu_df_comp.melt(
                    id_vars=['SequenceID'], var_name='Codon', value_name='RSCU'
                )
                current_gc_dict = utils.get_genetic_code(args.genetic_code)
                long_rscu_df_comp['AminoAcid'] = long_rscu_df_comp['Codon'].map(current_gc_dict.get)

                for fmt in args.plot_formats:
                    plot_filename = f"RSCU_boxplot_complete.{fmt}"
                    # Plots are saved in output_dir_path / "images"
                    rscu_boxplot_complete_filepath = output_dir_path / "images" / plot_filename
                    plotting.plot_rscu_boxplot_per_gene(
                        long_rscu_df_comp,
                        agg_usage_df_complete, # Aggregate RSCU for this "complete" set
                        'complete', # Gene name for title/filename
                        str(rscu_boxplot_complete_filepath) # Full save path
                    )
            except Exception as e: # pragma: no cover
                logger.error(f"Failed to generate 'complete' RSCU boxplot: {e}")
        elif not args.skip_plots: # pragma: no cover
            logger.warning("Cannot generate 'complete' RSCU boxplot due to missing aggregate usage or CA input data for the complete set.")

        return (agg_usage_df_complete, per_seq_df_complete, nucl_freqs_complete_agg,
                dinucl_freqs_complete_agg, per_seq_nucl_freqs_complete,
                per_seq_dinucl_freqs_complete, ca_input_df_complete_for_plot)

    except Exception as e: # pragma: no cover (should be rare if run_full_analysis is robust)
        logger.exception(f"Error during 'complete' sequence analysis execution: {e}")
        return None, None, None, None, None, None, None


def _update_aggregate_data_with_complete_results(
    complete_results_tuple: Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, float]],
        Optional[Dict[str, float]], Optional[Dict[str, Dict[str, float]]],
        Optional[Dict[str, Dict[str, float]]], Optional[pd.DataFrame]
    ],
    all_per_sequence_dfs: List[pd.DataFrame],
    all_ca_input_dfs: Dict[str, pd.DataFrame], # {gene_name: RSCU_wide_df}
    all_nucl_freqs_by_gene_agg: Dict[str, Dict[str, float]],
    all_dinucl_freqs_by_gene_agg: Dict[str, Dict[str, float]],
    all_nucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]],
    all_dinucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]],
    # args parameter was here but not used, removed.
) -> None:
    """
    Updates the main aggregate data collections with results from 'complete' sequence analysis.
    Modifies the passed-in dictionary and list arguments in place.
    """
    (agg_usage_df_complete, per_seq_df_complete, nucl_freqs_complete_agg,
     dinucl_freqs_complete_agg, per_seq_nucl_freqs_complete,
     per_seq_dinucl_freqs_complete, ca_input_df_complete_for_plot) = complete_results_tuple

    # Update aggregate frequencies for the 'complete' set
    if nucl_freqs_complete_agg:
        all_nucl_freqs_by_gene_agg['complete'] = nucl_freqs_complete_agg
    if dinucl_freqs_complete_agg:
        all_dinucl_freqs_by_gene_agg['complete'] = dinucl_freqs_complete_agg

    # Update per-sequence frequencies for the 'complete' set
    if per_seq_nucl_freqs_complete:
        all_nucl_freqs_per_seq_in_gene['complete'] = per_seq_nucl_freqs_complete
    if per_seq_dinucl_freqs_complete:
        all_dinucl_freqs_per_seq_in_gene['complete'] = per_seq_dinucl_freqs_complete

    # Update per-sequence metrics DataFrame list
    if per_seq_df_complete is not None and not per_seq_df_complete.empty:
        # Create a copy to avoid modifying the original DataFrame from complete_analysis
        df_to_add = per_seq_df_complete.copy()
        if 'ID' in df_to_add.columns:
            # Store original ID before modifying for combined table
            df_to_add['Original_ID'] = df_to_add['ID']
            # Prefix ID with 'complete__' to distinguish in combined table
            df_to_add['ID'] = "complete__" + df_to_add['ID'].astype(str)
        df_to_add['Gene'] = 'complete' # Mark these rows as belonging to 'complete' set
        all_per_sequence_dfs.append(df_to_add)
        logger.info("Appended 'complete' sequence metrics to the main per-sequence DataFrame list.")

    # Update CA input DataFrame dictionary
    if ca_input_df_complete_for_plot is not None and not ca_input_df_complete_for_plot.empty:
        df_ca_to_add = ca_input_df_complete_for_plot.copy()
        # Prefix index (original_id) with 'complete__'
        df_ca_to_add.index = "complete__" + df_ca_to_add.index.astype(str)
        all_ca_input_dfs['complete'] = df_ca_to_add
        logger.info("Added 'complete' sequence RSCU data to CA input dictionary.")

    # The RSCU boxplot for "complete" is now handled inside _analyze_complete_sequences_cli
    # to ensure it has access to args.plot_formats and output_dir_path directly.


def _finalize_and_save_per_sequence_metrics(
    all_per_sequence_dfs: List[pd.DataFrame],
    output_dir_path: Path # Main output directory
) -> Optional[pd.DataFrame]:
    """
    Combines all per-sequence DataFrames into a single DataFrame and saves it to a CSV file
    in the 'data' subdirectory of the main output path.

    Args:
        all_per_sequence_dfs: List of DataFrames, each containing per-sequence metrics
                              for a gene or the 'complete' set.
        output_dir_path: Path object for the main output directory.

    Returns:
        The combined DataFrame, or None if no data or if saving fails.
    """
    if not all_per_sequence_dfs:
        logger.error("No per-sequence results collected from any gene. Cannot save combined metrics.")
        return None
    try:
        combined_df = pd.concat(all_per_sequence_dfs, ignore_index=True)
        if combined_df.empty:
            logger.warning("Combined per-sequence DataFrame is empty after concatenation.")
            return None

        # Define save path within the 'data' subdirectory
        data_subdir = output_dir_path / "data"
        # data_subdir.mkdir(parents=True, exist_ok=True) # Should be created by _ensure_output_subdirectories
        filepath = data_subdir / "per_sequence_metrics_all_genes.csv"

        combined_df.to_csv(filepath, index=False, float_format='%.5f')
        logger.info(f"Combined per-sequence metrics saved to: {filepath}")
        return combined_df
    except Exception as e: # pragma: no cover (should be rare with checks)
        logger.exception(f"Error concatenating or saving combined per-sequence results: {e}")
        return None


def _generate_summary_tables_and_stats(
    combined_per_sequence_df: pd.DataFrame, # Should include metadata if provided
    all_nucl_freqs_by_gene_agg: Dict[str, Dict[str, float]], # {gene_name: {nucl: freq}}
    all_dinucl_freqs_by_gene_agg: Dict[str, Dict[str, float]], # {gene_name: {dinucl: freq}}
    # output_dir_path is not used here as saving is done in a separate function
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """
    Generates summary tables: mean features per gene, statistical comparisons
    between genes, and relative dinucleotide abundance per gene.

    Args:
        combined_per_sequence_df: DataFrame containing all per-sequence metrics,
                                  including a 'Gene' column.
        all_nucl_freqs_by_gene_agg: Aggregate nucleotide frequencies per gene.
        all_dinucl_freqs_by_gene_agg: Aggregate dinucleotide frequencies per gene.

    Returns:
        A tuple containing:
        - DataFrame of mean features per gene (or None).
        - DataFrame of statistical comparison results (or None).
        - DataFrame of relative dinucleotide abundances per gene (can be empty).
    """
    mean_summary_df: Optional[pd.DataFrame] = None
    comparison_results_df: Optional[pd.DataFrame] = None
    rel_abund_df_long: pd.DataFrame = pd.DataFrame() # For long format: Gene, Dinucleotide, RelativeAbundance

    # 1. Calculate Mean Features per Gene
    logger.info("Calculating mean features per gene...")
    # Define a comprehensive list of features that might be calculated
    mean_features_list = [
        'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'RCDI', 'ENC', 'CAI',
        'Aromaticity', 'GRAVY', 'Length', 'TotalCodons', 'ProteinLength', 'Fop'
    ]
    # Filter for features actually present in the combined DataFrame
    available_features_for_mean = [f for f in mean_features_list if f in combined_per_sequence_df.columns]

    if 'Gene' in combined_per_sequence_df.columns and available_features_for_mean:
        try:
            # Ensure features are numeric before grouping
            temp_df_for_mean = combined_per_sequence_df.copy()
            for col in available_features_for_mean:
                temp_df_for_mean[col] = pd.to_numeric(temp_df_for_mean[col], errors='coerce')

            # Group by 'Gene' and calculate mean, handling potential all-NaN slices
            mean_summary_df = temp_df_for_mean.groupby('Gene')[available_features_for_mean].mean(numeric_only=True).reset_index()
            logger.debug(f"Mean features per gene calculated. Shape: {mean_summary_df.shape}")
        except Exception as e: # pragma: no cover
            logger.exception(f"Error calculating mean features per gene: {e}")
            mean_summary_df = None # Ensure it's None on error
    else:
        logger.warning("Cannot calculate mean features: 'Gene' column missing or no suitable features available in combined data.")

    # 2. Perform Statistical Comparisons Between Genes (if scipy available)
    if SCIPY_AVAILABLE and scipy_stats is not None:
        logger.info("Performing statistical comparison of features between genes (Kruskal-Wallis)...")
        try:
            # Use the same list of available features for consistency
            comparison_results_df = analysis.compare_features_between_genes(
                combined_per_sequence_df,
                features=available_features_for_mean, # Use features confirmed to be present
                method='kruskal'
            )
            logger.debug(f"Statistical comparisons completed. Results shape: {comparison_results_df.shape if comparison_results_df is not None else 'None'}")
        except Exception as e: # pragma: no cover
            logger.exception(f"Error during statistical comparison between genes: {e}")
            comparison_results_df = None # Ensure None on error
    else:
        logger.warning("Scipy library not available. Skipping statistical comparisons between genes.")

    # 3. Calculate Relative Dinucleotide Abundance (O/E Ratios) per Gene
    logger.info("Calculating relative dinucleotide abundances per gene...")
    rel_abund_data_list: List[Dict[str, Any]] = []
    # Ensure we only process genes for which both nucl and dinucl frequencies are available
    valid_genes_for_dinucl_calc = sorted(list(
        set(all_nucl_freqs_by_gene_agg.keys()) & set(all_dinucl_freqs_by_gene_agg.keys())
    ))

    if valid_genes_for_dinucl_calc:
        for gene_name_iter in valid_genes_for_dinucl_calc:
            nucl_f = all_nucl_freqs_by_gene_agg.get(gene_name_iter)
            dinucl_f = all_dinucl_freqs_by_gene_agg.get(gene_name_iter)
            if nucl_f and dinucl_f: # Both must be non-empty
                try:
                    rel_abund_gene = analysis.calculate_relative_dinucleotide_abundance(nucl_f, dinucl_f)
                    for dinucleotide, ratio_val in rel_abund_gene.items():
                        rel_abund_data_list.append({
                            'Gene': gene_name_iter,
                            'Dinucleotide': dinucleotide,
                            'RelativeAbundance': ratio_val
                        })
                except Exception as e: # pragma: no cover
                    logger.warning(f"Could not calculate relative dinucleotide abundance for gene '{gene_name_iter}': {e}")
        if rel_abund_data_list:
            rel_abund_df_long = pd.DataFrame(rel_abund_data_list)
            logger.debug(f"Relative dinucleotide abundances calculated. Long DataFrame shape: {rel_abund_df_long.shape}")
    else:
        logger.warning("Missing aggregate nucleotide or dinucleotide frequency data for one or more genes. "
                       "Relative dinucleotide abundance calculation might be incomplete or skipped.")

    return mean_summary_df, comparison_results_df, rel_abund_df_long


def _perform_and_save_combined_ca(
    all_ca_input_dfs: Dict[str, pd.DataFrame], # {gene_name: RSCU_wide_df}
    output_dir_path: Path, # Main output directory
    args: argparse.Namespace
) -> Tuple[Optional[pd.DataFrame], Optional[PrinceCA], Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Performs combined Correspondence Analysis (CA) on RSCU data from all genes.
    Saves CA output tables (coordinates, contributions, eigenvalues) to the
    'data' subdirectory.

    Args:
        all_ca_input_dfs: Dictionary mapping gene names to their RSCU (wide) DataFrames.
        output_dir_path: Main output directory path.
        args: Parsed command-line arguments.

    Returns:
        A tuple containing:
        - Combined CA input DataFrame (sequences as rows, codons as columns, RSCU values).
        - Fitted prince.CA object.
        - Series of gene groups for plotting.
        - DataFrame of CA row (sequence) coordinates.
        All can be None if CA is skipped or fails.
    """
    if args.skip_ca:
        logger.info("Skipping combined Correspondence Analysis as requested (--skip_ca).")
        return None, None, None, None
    if not PRINCE_AVAILABLE or prince is None:
        logger.error("'prince' library is not installed. Cannot perform Correspondence Analysis. "
                     "Please install it (e.g., 'pip install prince'). Skipping CA.")
        return None, None, None, None
    if not all_ca_input_dfs:
        logger.info("Skipping combined CA: No CA input data (RSCU DataFrames per gene) available.")
        return None, None, None, None

    logger.info("Performing combined Correspondence Analysis on RSCU data from all genes...")
    # Define path for saving CA-related data tables
    data_subdir = output_dir_path / "data"
    # data_subdir.mkdir(parents=True, exist_ok=True) # Should be created by _ensure_output_subdirectories

    combined_ca_input_df: Optional[pd.DataFrame] = None
    ca_results_combined: Optional[PrinceCA] = None # type: ignore
    gene_groups_for_plotting: Optional[pd.Series] = None
    ca_row_coords_df: Optional[pd.DataFrame] = None

    try:
        # Concatenate all RSCU_wide DataFrames. Index will be 'gene__originalID'.
        ca_input_dfs_list = list(all_ca_input_dfs.values())
        if not ca_input_dfs_list: # Should be caught by earlier check on all_ca_input_dfs
            logger.warning("CA input DFS list is empty. Skipping combined CA.")
            return None, None, None, None

        combined_ca_input_df = pd.concat(ca_input_dfs_list)
        if combined_ca_input_df.empty:
            logger.warning("Combined CA input DataFrame is empty after concatenation. Skipping CA.")
            return None, None, None, None

        logger.debug(f"Shape of combined CA input DataFrame before cleaning: {combined_ca_input_df.shape}")

        # Clean the combined DataFrame: ensure numeric, fill NaNs, replace Infs
        for col in combined_ca_input_df.columns:
            combined_ca_input_df[col] = pd.to_numeric(combined_ca_input_df[col], errors='coerce')
        combined_ca_input_df.fillna(0.0, inplace=True) # Fill NaNs that might result from coerce or missing codons
        combined_ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True) # Handle infinities

        logger.debug(f"Shape of combined CA input DataFrame after initial cleaning: {combined_ca_input_df.shape}")

        # --- Save the combined RSCU wide table used for CA ---
        rscu_wide_path = data_subdir / "per_sequence_rscu_wide.csv"
        combined_ca_input_df.to_csv(rscu_wide_path, float_format='%.4f')
        logger.info(f"Per-sequence RSCU (wide format) for combined CA saved: {rscu_wide_path}")

        # Perform CA using the analysis module's robust function
        # The perform_ca function in analysis.py handles more detailed filtering of rows/columns.
        ca_results_combined = analysis.perform_ca(combined_ca_input_df.copy(), n_components=args.ca_components) # Pass n_components from args

        if ca_results_combined:
            logger.info("Combined CA fitting complete. Saving detailed CA output tables...")
            # Extract gene groups from the index (e.g., "geneA__seq1" -> "geneA")
            # This is needed for coloring plots by gene.
            if combined_ca_input_df.index.name == "ID" or isinstance(combined_ca_input_df.index, pd.MultiIndex):
                 # If index is already 'ID' or MultiIndex, handle appropriately
                 # This part might need adjustment based on actual index structure after concat
                 ids_for_grouping = combined_ca_input_df.index.to_series()
            else: # Assume simple index from concat
                 ids_for_grouping = pd.Series(combined_ca_input_df.index)

            split_idx = ids_for_grouping.str.split('__', n=1, expand=True)
            if split_idx.shape[1] > 0 and not split_idx.empty: # Check if split was successful
                gene_groups_for_plotting = pd.Series(
                    data=split_idx.iloc[:, 0].values, # Get the first part (gene name)
                    index=combined_ca_input_df.index, # Use original index
                    name='Gene' # Name for the series
                )
            else: # pragma: no cover (should parse if index format is consistent)
                logger.warning("Could not reliably parse gene groups from combined CA input DataFrame index for plotting.")
                gene_groups_for_plotting = None

            # --- Save CA result tables ---
            # Note: perform_ca in analysis.py returns the fitted Prince object.
            # We use its methods to get coordinates etc.
            ca_row_coords_df = ca_results_combined.row_coordinates(combined_ca_input_df)
            ca_row_coords_df.to_csv(data_subdir / "ca_row_coordinates.csv", float_format='%.5f')

            col_coords_df = ca_results_combined.column_coordinates(combined_ca_input_df)
            col_coords_df.to_csv(data_subdir / "ca_col_coordinates.csv", float_format='%.5f')

            if hasattr(ca_results_combined, 'column_contributions_'):
                ca_results_combined.column_contributions_.to_csv(
                    data_subdir / "ca_col_contributions.csv", float_format='%.5f'
                )
            if hasattr(ca_results_combined, 'eigenvalues_summary'):
                ca_results_combined.eigenvalues_summary.to_csv(
                    data_subdir / "ca_eigenvalues.csv", float_format='%.5g' # Use 'g' for better display of percentages
                )
            logger.info("Combined CA output tables (coordinates, contributions, eigenvalues) saved.")
        else: # pragma: no cover (if perform_ca returns None)
            logger.warning("Combined CA fitting failed or produced no result. CA output tables will not be saved.")
            combined_ca_input_df = None # Ensure this is None if CA failed
            # Other related vars (ca_results_combined, gene_groups, ca_row_coords_df) will also be None
            ca_results_combined, gene_groups_for_plotting, ca_row_coords_df = None, None, None

    except Exception as e: # pragma: no cover
        logger.exception(f"Error during combined CA processing or saving: {e}")
        combined_ca_input_df, ca_results_combined, gene_groups_for_plotting, ca_row_coords_df = None, None, None, None

    return combined_ca_input_df, ca_results_combined, gene_groups_for_plotting, ca_row_coords_df


def _generate_color_palette_for_groups(
    data_df: Optional[pd.DataFrame],
    group_column_name: str = 'Gene',
    include_complete: bool = True # Whether to include 'complete' in palette if present
) -> Optional[Dict[str, Any]]:
    """
    Generates a color palette for unique categories in a specified column of a DataFrame.

    Args:
        data_df: DataFrame containing the group column.
        group_column_name: Name of the column to use for grouping and coloring.
        include_complete: If True and 'complete' is a category, it will be included.

    Returns:
        A dictionary mapping group names to colors, or None if generation fails.
    """
    if data_df is None or data_df.empty or group_column_name not in data_df.columns:
        logger.warning(f"Cannot generate color map: DataFrame missing or column '{group_column_name}' not found.")
        return None

    # Get unique groups, ensuring 'complete' is last if present and included
    unique_groups_raw = data_df[group_column_name].unique()
    unique_groups_list: List[str] = []
    has_complete = 'complete' in unique_groups_raw

    for g_val in sorted([str(g) for g in unique_groups_raw if str(g) != 'complete']):
        unique_groups_list.append(g_val)
    if include_complete and has_complete:
        unique_groups_list.append('complete')

    if not unique_groups_list:
        logger.warning(f"No unique groups found in column '{group_column_name}' for color map generation.")
        return None

    try:
        # Generate a color palette using seaborn
        # 'husl' is good for categorical data with many levels
        # Can also use other palettes like "viridis", "tab10", "Set2", etc.
        palette_list = sns.color_palette("husl", n_colors=len(unique_groups_list))
        group_color_map = {group_name: color for group_name, color in zip(unique_groups_list, palette_list)}
        logger.debug(f"Generated color map for {len(unique_groups_list)} groups in '{group_column_name}'.")
        return group_color_map
    except Exception as e: # pragma: no cover
        logger.warning(f"Could not generate color palette for '{group_column_name}': {e}. Plot colors may be default.")
        return None


def _generate_plots_per_gene_colored_by_metadata(
    args: argparse.Namespace,
    combined_per_sequence_df_with_meta: pd.DataFrame, # Assumes 'Gene', 'Original_ID', and metadata_col_for_color are present
    all_ca_input_dfs: Dict[str, pd.DataFrame], # {gene_name: RSCU_wide_df}
    all_nucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]], # {gene: {seq_id: {nucl: freq}}}
    all_dinucl_freqs_per_seq_in_gene: Dict[str, Dict[str, Dict[str, float]]], # {gene: {seq_id: {dinucl: freq}}}
    metadata_col_for_color: str, # The actual metadata column name
    top_n_categories_map: Dict[str, str], # Maps original metadata values to topN/"Other"
    output_dir_path: Path, # Main output directory
    metadata_category_color_map: Optional[Dict[str, Any]] # Palette for the topN categories
) -> None:
    """
    Generates per-gene plots (ENC vs GC3, Neutrality, CA, Dinucleotide Abundance)
    where data points are colored or grouped by categories from a specified metadata column.
    Plots are saved into subdirectories under `output_dir_path / "images"`.

    Args:
        args: Parsed command-line arguments.
        combined_per_sequence_df_with_meta: DataFrame with all per-sequence metrics and merged metadata.
        all_ca_input_dfs: Dictionary of RSCU wide DataFrames for each gene, for CA.
        all_nucl_freqs_per_seq_in_gene: Per-sequence nucleotide frequencies.
        all_dinucl_freqs_per_seq_in_gene: Per-sequence dinucleotide frequencies.
        metadata_col_for_color: The original name of the metadata column used for coloring.
        top_n_categories_map: Mapping from original metadata values to "top N" or "Other".
        output_dir_path: The root directory for all outputs (e.g., "codon_analysis_results").
        metadata_category_color_map: Color palette for the (potentially grouped) metadata categories.
    """
    logger.info(f"Generating per-gene plots colored by metadata column '{metadata_col_for_color}'...")

    # Base directory for these specific metadata-colored plots, under the main "images" directory
    # e.g., output_dir_path / "images" / "metadataColName_per_gene_plots"
    metadata_plots_image_base_dir = output_dir_path / "images" / f"{utils.sanitize_filename(metadata_col_for_color)}_per_gene_plots"
    # Plotting functions will create subdirs under this if needed.

    # The column to use for actual hueing in plots (contains original values mapped to topN/"Other")
    plot_hue_col = f"{metadata_col_for_color}_topN_mapped" # Use a distinct name
    if metadata_col_for_color not in combined_per_sequence_df_with_meta.columns: # pragma: no cover (should be caught earlier)
        logger.error(f"Original metadata column '{metadata_col_for_color}' not found. Skipping per-gene metadata plots.")
        return
    # Create the mapping column if it doesn't exist
    combined_per_sequence_df_with_meta[plot_hue_col] = \
        combined_per_sequence_df_with_meta[metadata_col_for_color].astype(str).map(top_n_categories_map).fillna("Unknown")


    # Iterate through unique gene names present in the combined data
    # This includes 'complete' if it was analyzed.
    unique_genes_in_data = combined_per_sequence_df_with_meta['Gene'].unique()

    for gene_name_iter in unique_genes_in_data:
        # Directory for this specific gene's metadata-colored plots
        # e.g., .../images/metadataColName_per_gene_plots/GeneA/
        gene_specific_meta_plots_dir_abs = metadata_plots_image_base_dir / utils.sanitize_filename(gene_name_iter)
        # Plotting functions will create this directory if they save a file.

        logger.info(f"  Processing gene '{gene_name_iter}' for metadata-colored plots (output to: {gene_specific_meta_plots_dir_abs})")

        # Filter data for the current gene
        gene_specific_metrics_df = combined_per_sequence_df_with_meta[
            combined_per_sequence_df_with_meta['Gene'] == gene_name_iter
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if gene_specific_metrics_df.empty: # pragma: no cover
            logger.warning(f"No metric data for gene '{gene_name_iter}' after filtering. Skipping plots for this gene.")
            continue
        if plot_hue_col not in gene_specific_metrics_df.columns: # pragma: no cover (defensive)
            logger.error(f"Plot hue column '{plot_hue_col}' is missing from data for gene '{gene_name_iter}'. Skipping plots.")
            continue

        plot_title_prefix_for_gene = f"Gene {gene_name_iter}: "
        # Filename suffix to distinguish these plots
        filename_suffix_metadata_gene = f"_{utils.sanitize_filename(gene_name_iter)}_by_{utils.sanitize_filename(metadata_col_for_color)}"

        # --- Generate standard plots colored by metadata ---
        if not args.skip_plots:
            for fmt in args.plot_formats:
                # ENC vs GC3 Plot
                enc_plot_filepath = gene_specific_meta_plots_dir_abs / f"enc_vs_gc3_plot{filename_suffix_metadata_gene}.{fmt}"
                plotting.plot_enc_vs_gc3(
                    per_sequence_df=gene_specific_metrics_df,
                    output_filepath=str(enc_plot_filepath),
                    group_by_col=plot_hue_col, # Use the mapped hue column
                    palette=metadata_category_color_map,
                    plot_title_prefix=plot_title_prefix_for_gene
                )

                # Neutrality Plot
                neutrality_plot_filepath = gene_specific_meta_plots_dir_abs / f"neutrality_plot{filename_suffix_metadata_gene}.{fmt}"
                plotting.plot_neutrality(
                    per_sequence_df=gene_specific_metrics_df,
                    output_filepath=str(neutrality_plot_filepath),
                    group_by_col=plot_hue_col, # Use the mapped hue column
                    palette=metadata_category_color_map,
                    plot_title_prefix=plot_title_prefix_for_gene
                )

        # --- Per-Gene Correspondence Analysis (colored by metadata) ---
        rscu_df_for_this_gene: Optional[pd.DataFrame] = all_ca_input_dfs.get(gene_name_iter)
        if not args.skip_ca and PRINCE_AVAILABLE and rscu_df_for_this_gene is not None and not rscu_df_for_this_gene.empty:
            # Align RSCU data with metrics data (which contains the hue column)
            # The index of rscu_df_for_this_gene is 'geneName__originalID'
            # The index of gene_specific_metrics_df is default range index, but has 'ID' column like 'geneName__originalID'
            
            # Ensure gene_specific_metrics_df is indexed by 'ID' for merging/aligning
            temp_metrics_for_ca_hue = gene_specific_metrics_df.set_index('ID', drop=False) # Keep ID column if needed later
            
            # Find common indices (sequence IDs) between RSCU data and metrics data for this gene
            common_indices_for_ca = rscu_df_for_this_gene.index.intersection(temp_metrics_for_ca_hue.index)

            if not common_indices_for_ca.empty:
                rscu_data_for_ca_plot = rscu_df_for_this_gene.loc[common_indices_for_ca]
                # Get the hue series aligned with rscu_data_for_ca_plot
                hue_series_for_ca_plot = temp_metrics_for_ca_hue.loc[common_indices_for_ca, plot_hue_col]

                if not rscu_data_for_ca_plot.empty and rscu_data_for_ca_plot.shape[0] >= 2 and rscu_data_for_ca_plot.shape[1] >= 2:
                    ca_results_this_gene = analysis.perform_ca(rscu_data_for_ca_plot.copy(), n_components=args.ca_components)
                    if ca_results_this_gene:
                        for fmt in args.plot_formats:
                            ca_plot_filename = f"ca_biplot_comp{args.ca_dims[0]+1}v{args.ca_dims[1]+1}{filename_suffix_metadata_gene}.{fmt}"
                            ca_plot_filepath = gene_specific_meta_plots_dir_abs / ca_plot_filename
                            plotting.plot_ca(
                                ca_results=ca_results_this_gene,
                                ca_input_df=rscu_data_for_ca_plot,
                                output_filepath=str(ca_plot_filepath),
                                comp_x=args.ca_dims[0], comp_y=args.ca_dims[1],
                                groups=hue_series_for_ca_plot, # Series of metadata categories for hue
                                palette=metadata_category_color_map,
                                plot_title_prefix=plot_title_prefix_for_gene
                            )
                else: # pragma: no cover
                    logger.warning(f"Not enough data for CA for gene '{gene_name_iter}' (shape: {rscu_data_for_ca_plot.shape}) "
                                   "after filtering for metadata plot. Skipping CA plot.")
            else: # pragma: no cover
                logger.warning(f"No common sequences between RSCU data and metrics for CA plot of gene '{gene_name_iter}'. Skipping CA plot.")
        elif not args.skip_ca:
             logger.info(f"Skipping CA plot for gene '{gene_name_iter}' due to missing RSCU data or prince library.")


        # --- Per-Gene Dinucleotide Abundance Plot (colored by metadata) ---
        # This requires per-sequence nucleotide and dinucleotide frequencies for this gene.
        nucl_freqs_per_seq_this_gene = all_nucl_freqs_per_seq_in_gene.get(gene_name_iter)
        dinucl_freqs_per_seq_this_gene = all_dinucl_freqs_per_seq_in_gene.get(gene_name_iter)

        if not args.skip_plots and nucl_freqs_per_seq_this_gene and dinucl_freqs_per_seq_this_gene:
            per_sequence_oe_ratios_list_for_plot: List[Dict[str, Any]] = []
            # Iterate through original sequence IDs present in the metrics for this gene
            # 'Original_ID' column was added during aggregation of 'complete' or individual gene processing
            if 'Original_ID' not in gene_specific_metrics_df.columns: # pragma: no cover (defensive)
                logger.warning(f"'Original_ID' column missing for gene '{gene_name_iter}'. Cannot create dinucleotide plot by metadata.")
            else:
                for original_seq_id_val in gene_specific_metrics_df['Original_ID'].unique():
                    # Get the metadata category for this original_seq_id
                    # There might be multiple entries if Original_ID is not unique across 'ID' (gene__Original_ID)
                    # For simplicity, take the first metadata category found for this original_seq_id
                    seq_meta_rows = gene_specific_metrics_df[gene_specific_metrics_df['Original_ID'] == original_seq_id_val]
                    if seq_meta_rows.empty: continue # Should not happen if iterating unique IDs from the df

                    metadata_category_for_hue = seq_meta_rows.iloc[0][plot_hue_col]

                    # Get per-sequence frequencies for this original_seq_id
                    nucl_freqs_single_seq = nucl_freqs_per_seq_this_gene.get(str(original_seq_id_val))
                    dinucl_freqs_single_seq = dinucl_freqs_per_seq_this_gene.get(str(original_seq_id_val))

                    if nucl_freqs_single_seq and dinucl_freqs_single_seq:
                        oe_ratios_single_seq = analysis.calculate_relative_dinucleotide_abundance(
                            nucl_freqs_single_seq, dinucl_freqs_single_seq
                        )
                        for dinucl, ratio_val_dinucl in oe_ratios_single_seq.items():
                            per_sequence_oe_ratios_list_for_plot.append({
                                'SequenceID': original_seq_id_val, # For potential grouping, not strictly needed by plot func
                                'Dinucleotide': dinucl,
                                'RelativeAbundance': ratio_val_dinucl,
                                plot_hue_col: metadata_category_for_hue # This is the crucial column for hue
                            })
                if per_sequence_oe_ratios_list_for_plot:
                    per_sequence_oe_ratios_df_for_plot = pd.DataFrame(per_sequence_oe_ratios_list_for_plot)
                    for fmt in args.plot_formats:
                        dinucl_plot_filename = f"dinucl_abundance{filename_suffix_metadata_gene}.{fmt}"
                        dinucl_plot_filepath = gene_specific_meta_plots_dir_abs / dinucl_plot_filename
                        plotting.plot_per_gene_dinucleotide_abundance_by_metadata(
                            per_sequence_oe_ratios_df=per_sequence_oe_ratios_df_for_plot,
                            metadata_hue_col=plot_hue_col, # Column containing mapped categories
                            output_filepath=str(dinucl_plot_filepath),
                            palette=metadata_category_color_map,
                            gene_name=gene_name_iter, # For title
                        )
                else: # pragma: no cover
                    logger.warning(f"No per-sequence O/E ratios calculated for dinucleotide plot of gene '{gene_name_iter}'.")
        elif not args.skip_plots: # pragma: no cover
             logger.info(f"Skipping per-sequence dinucleotide plot for gene '{gene_name_iter}' "
                         "due to missing per-sequence frequency data.")


def _generate_all_combined_plots(
    args: argparse.Namespace,
    combined_per_sequence_df: Optional[pd.DataFrame], # Assumes 'Gene' column for grouping
    gene_color_map: Optional[Dict[str, Any]], # Palette for 'Gene' groups
    rel_abund_df_long: pd.DataFrame, # Long format: Gene, Dinucleotide, RelativeAbundance
    ca_results_combined: Optional[PrinceCA],
    combined_ca_input_df: Optional[pd.DataFrame], # RSCU wide, index 'gene__origID'
    gene_groups_for_ca_plotting: Optional[pd.Series], # Series with 'Gene' values, index matching combined_ca_input_df
    ca_row_coords_final: Optional[pd.DataFrame], # Row coordinates from combined CA
    reference_data_for_plot: Optional[pd.DataFrame], # For RSCU comparison plot
    output_dir_path: Path # Main output directory
) -> None:
    """
    Generates and saves all combined plots that summarize trends across all genes.
    Plots are saved into `output_dir_path / "images"`.

    Args:
        args: Parsed command-line arguments.
        combined_per_sequence_df: DataFrame of all per-sequence metrics.
        gene_color_map: Color palette for gene-based grouping.
        rel_abund_df_long: DataFrame of relative dinucleotide abundances (long format).
        ca_results_combined: Fitted prince.CA object for combined CA.
        combined_ca_input_df: Input DataFrame used for the combined CA.
        gene_groups_for_ca_plotting: Series mapping CA row indices to gene names.
        ca_row_coords_final: DataFrame of row coordinates from combined CA.
        reference_data_for_plot: DataFrame of reference codon usage data.
        output_dir_path: Main output directory path.
    """
    if args.skip_plots:
        logger.info("Skipping generation of all combined plots as requested (--skip_plots).")
        return

    logger.info("Generating combined plots (summarizing across all genes/sequences)...")
    # Base directory for all combined plot images
    output_images_dir_abs = output_dir_path / "images"
    # Plotting functions should handle parent directory creation if they save a file.
    # output_images_dir_abs.mkdir(parents=True, exist_ok=True) # Ensured by _ensure_output_subdirectories

    plot_formats_to_generate = args.plot_formats
    # Default values for CA plot parameters if not specified or invalid in args
    # Ensure ca_dims from args are valid indices for components.
    # Assuming ca_dims are 0-based indices [dim_x_idx, dim_y_idx].
    ca_dim_x_plot, ca_dim_y_plot = args.ca_dims[0], args.ca_dims[1] # from argparse
    n_ca_dims_for_variance_plot = 10 # Default for variance plot
    n_top_contributors_for_ca_plot = 10 # Default for contribution plots

    for fmt_ext in plot_formats_to_generate:
        logger.debug(f"Generating combined plots in format: .{fmt_ext}")
        try:
            if combined_per_sequence_df is not None and not combined_per_sequence_df.empty:
                # GC Means Barplot (grouped by Gene)
                gc_means_fname = f"gc_means_barplot_by_Gene.{fmt_ext}"
                plotting.plot_gc_means_barplot(
                    combined_per_sequence_df,
                    str(output_images_dir_abs / gc_means_fname),
                    group_by='Gene' # Explicitly group by 'Gene' column
                )

                # ENC vs GC3 Plot (grouped by Gene)
                enc_gc3_fname = f"enc_vs_gc3_plot_{utils.sanitize_filename('_grouped_by_gene')}.{fmt_ext}"
                plotting.plot_enc_vs_gc3(
                    combined_per_sequence_df,
                    output_filepath=str(output_images_dir_abs / enc_gc3_fname),
                    group_by_col='Gene',
                    palette=gene_color_map, # Use the gene-specific color map
                    plot_title_prefix="" # No prefix for combined plot
                )

                # Neutrality Plot (grouped by Gene)
                neutrality_fname = f"neutrality_plot_{utils.sanitize_filename('_grouped_by_gene')}.{fmt_ext}"
                plotting.plot_neutrality(
                    combined_per_sequence_df,
                    output_filepath=str(output_images_dir_abs / neutrality_fname),
                    group_by_col='Gene',
                    palette=gene_color_map, # Use the gene-specific color map
                    plot_title_prefix="" # No prefix
                )

                # Feature Correlation Heatmap
                features_for_main_correlation = [
                    'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'ENC', 'CAI', 'RCDI',
                    'Aromaticity', 'GRAVY', 'Length', 'TotalCodons', 'ProteinLength', 'Fop'
                ]
                available_features_corr = [
                    f for f in features_for_main_correlation if f in combined_per_sequence_df.columns
                ]
                if len(available_features_corr) >= 2:
                    corr_heatmap_fname = f"feature_correlation_heatmap_spearman.{fmt_ext}"
                    plotting.plot_correlation_heatmap(
                        combined_per_sequence_df,
                        available_features_corr,
                        str(output_images_dir_abs / corr_heatmap_fname),
                        method='spearman' # Default, can be parameterized if needed
                    )

            # Relative Dinucleotide Abundance Plot (lines per gene)
            if not rel_abund_df_long.empty:
                rel_dinucl_fname = f"relative_dinucleotide_abundance.{fmt_ext}"
                plotting.plot_relative_dinucleotide_abundance(
                    rel_abund_df_long,
                    str(output_images_dir_abs / rel_dinucl_fname),
                    palette=gene_color_map # Use gene_color_map if genes are hue
                )

            # Combined Correspondence Analysis (CA) Plots
            if ca_results_combined and combined_ca_input_df is not None and not combined_ca_input_df.empty:
                ca_plot_suffix = utils.sanitize_filename('_combined_by_gene')
                # CA Biplot
                ca_biplot_fname = f"ca_biplot_comp{ca_dim_x_plot+1}v{ca_dim_y_plot+1}_{ca_plot_suffix}.{fmt_ext}"
                plotting.plot_ca(
                    ca_results_combined, combined_ca_input_df,
                    str(output_images_dir_abs / ca_biplot_fname),
                    comp_x=ca_dim_x_plot, comp_y=ca_dim_y_plot,
                    groups=gene_groups_for_ca_plotting, # Series mapping index to Gene names
                    palette=gene_color_map, # Palette for Gene names
                    plot_title_prefix="Combined "
                )

                # CA Variance Explained Plot
                ca_var_fname = f"ca_variance_explained_top{n_ca_dims_for_variance_plot}.{fmt_ext}"
                plotting.plot_ca_variance(
                    ca_results_combined, n_ca_dims_for_variance_plot,
                    str(output_images_dir_abs / ca_var_fname)
                )

                # CA Contribution Plots (Dim 1 and Dim 2)
                if hasattr(ca_results_combined, 'column_contributions_') and \
                   not ca_results_combined.column_contributions_.empty:
                    max_dim_contrib = ca_results_combined.column_contributions_.shape[1]
                    if max_dim_contrib > 0:
                        ca_contrib1_fname = f"ca_contribution_dim1_top{n_top_contributors_for_ca_plot}.{fmt_ext}"
                        plotting.plot_ca_contribution(
                            ca_results_combined, 0, n_top_contributors_for_ca_plot,
                            str(output_images_dir_abs / ca_contrib1_fname)
                        )
                    if max_dim_contrib > 1:
                        ca_contrib2_fname = f"ca_contribution_dim2_top{n_top_contributors_for_ca_plot}.{fmt_ext}"
                        plotting.plot_ca_contribution(
                            ca_results_combined, 1, n_top_contributors_for_ca_plot,
                            str(output_images_dir_abs / ca_contrib2_fname)
                        )

            # CA Axes vs Features Correlation Heatmap
            if ca_row_coords_final is not None and not ca_row_coords_final.empty and \
               combined_per_sequence_df is not None and not combined_per_sequence_df.empty and \
               combined_ca_input_df is not None and not combined_ca_input_df.empty:

                # Ensure ca_row_coords_final has the specified dimensions
                max_available_dim_idx = ca_row_coords_final.shape[1] - 1
                if not (ca_dim_x_plot > max_available_dim_idx or ca_dim_y_plot > max_available_dim_idx or ca_dim_x_plot == ca_dim_y_plot):
                    # Select only the specified CA dimensions for the plot
                    ca_dims_for_corr_plot = ca_row_coords_final[[ca_dim_x_plot, ca_dim_y_plot]].copy()
                    ca_dims_for_corr_plot.columns = [f'CA_Dim{ca_dim_x_plot+1}', f'CA_Dim{ca_dim_y_plot+1}']

                    # Prepare metrics DataFrame (needs to be indexed by sequence ID)
                    metrics_df_for_corr = combined_per_sequence_df.copy()
                    if 'ID' in metrics_df_for_corr.columns:
                        if not metrics_df_for_corr['ID'].is_unique: # pragma: no cover (defensive)
                            metrics_df_for_corr.drop_duplicates(subset=['ID'], keep='first', inplace=True)
                        metrics_df_for_corr.set_index('ID', inplace=True)
                    else: # pragma: no cover
                        logger.warning("Cannot generate CA Axes vs Features correlation: 'ID' column missing in metrics.")
                        metrics_df_for_corr = pd.DataFrame() # Empty

                    # Prepare RSCU DataFrame (already indexed by 'gene__origID')
                    rscu_df_for_corr = combined_ca_input_df.copy() # Already cleaned and indexed

                    if not ca_dims_for_corr_plot.empty and not metrics_df_for_corr.empty and not rscu_df_for_corr.empty:
                        # Define features from metrics and RSCU to correlate
                        metric_features_list_corr = [
                            'Length', 'TotalCodons', 'GC', 'GC1', 'GC2', 'GC3', 'GC12',
                            'ENC', 'CAI', 'Fop', 'RCDI', 'ProteinLength', 'GRAVY', 'Aromaticity'
                        ]
                        available_metric_f_corr = [f for f in metric_features_list_corr if f in metrics_df_for_corr.columns]
                        available_rscu_c_corr = sorted([
                            col for col in rscu_df_for_corr.columns if len(col) == 3 and col.isupper()
                        ])
                        features_to_correlate_list = available_metric_f_corr + available_rscu_c_corr

                        if features_to_correlate_list:
                            ca_feat_corr_fname = f"ca_axes_feature_corr_{utils.sanitize_filename('Spearman')}.{fmt_ext}"
                            plotting.plot_ca_axes_feature_correlation(
                                ca_dims_df=ca_dims_for_corr_plot,
                                metrics_df=metrics_df_for_corr,
                                rscu_df=rscu_df_for_corr,
                                output_filepath=str(output_images_dir_abs / ca_feat_corr_fname),
                                features_to_correlate=features_to_correlate_list,
                                method_name="Spearman" # Default, can be parameterized
                            )
                else: # pragma: no cover (invalid CA dims requested)
                    logger.warning(f"Requested CA dimensions ({ca_dim_x_plot}, {ca_dim_y_plot}) are invalid "
                                   f"for available CA coordinates (max dim index: {max_available_dim_idx}). "
                                   "Skipping CA Axes vs Features correlation plot.")

            # RSCU Comparison Scatter Plot (if reference data is available)
            if reference_data_for_plot is not None and combined_per_sequence_df is not None:
                # Need an aggregate RSCU DataFrame for comparison (e.g., from 'complete' set or overall mean)
                # For simplicity, let's assume we want to compare the 'complete' set's aggregate RSCU if available
                # This part might need refinement based on what 'agg_usage_df_for_comp' should represent.
                # If `_analyze_complete_sequences_cli` returns `agg_usage_df_complete`, that could be used.
                # For now, this plot is commented out as `agg_usage_df_for_comp` is not clearly defined here.
                # agg_usage_df_for_comp = ... # This needs to be the aggregate RSCU of the sequences we want to compare
                # if agg_usage_df_for_comp is not None and not agg_usage_df_for_comp.empty:
                #    rscu_comp_fname = f"rscu_comparison_scatter.{fmt_ext}"
                #    plotting.plot_usage_comparison(
                #        agg_usage_df_for_comp,
                #        reference_data_for_plot,
                #        str(output_images_dir_abs / rscu_comp_fname)
                #    )
                pass


        except Exception as plot_err: # pragma: no cover
            logger.exception(f"Error during combined plot generation for format '.{fmt_ext}': {plot_err}")


def _save_main_output_tables(
    output_dir_path: Path, # Main output directory
    # combined_per_sequence_df is saved by _finalize_and_save_per_sequence_metrics
    mean_summary_df: Optional[pd.DataFrame],
    comparison_results_df: Optional[pd.DataFrame]
    # rel_abund_df_long is not saved as a separate main table here, but could be.
) -> None:
    """
    Saves the main summary output CSV tables (mean features, statistical comparisons)
    into the 'data' subdirectory of the main output path.

    Args:
        output_dir_path: Path object for the main output directory.
        mean_summary_df: DataFrame of mean features per gene.
        comparison_results_df: DataFrame of statistical comparison results.
    """
    data_subdir = output_dir_path / "data"
    # data_subdir.mkdir(parents=True, exist_ok=True) # Should be created by _ensure_output_subdirectories

    if mean_summary_df is not None and not mean_summary_df.empty:
        filepath = data_subdir / "mean_features_per_gene.csv"
        try:
            mean_summary_df.to_csv(filepath, index=False, float_format='%.4f')
            logger.info(f"Mean features per gene table saved: {filepath}")
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to save mean features table: {e}")


    if comparison_results_df is not None and not comparison_results_df.empty:
        filepath = data_subdir / "gene_comparison_stats.csv"
        try:
            comparison_results_df.to_csv(filepath, index=False, float_format='%.4g') # Use 'g' for p-values
            logger.info(f"Gene comparison statistics table saved: {filepath}")
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to save gene comparison statistics table: {e}")

# ------------------------------------------------------------------------------
# SUBCOMMAND HANDLERS
# ------------------------------------------------------------------------------

def handle_analyze_command(args: argparse.Namespace) -> None:
    """
    Orchestrates the 'analyze' subcommand workflow.
    This includes loading data, processing gene files, performing analyses,
    generating plots, and creating an HTML report.

    Args:
        args: Parsed command-line arguments for the 'analyze' subcommand.
    """
    logger.info(f"Starting 'analyze' command. Input directory: {args.directory}")
    logger.debug(f"Full 'analyze' arguments: {args}")

    # 1. Setup output directory and subdirectories
    output_dir_path = _setup_output_directory(args.output)
    _ensure_output_subdirectories(output_dir_path) # Create data/ and images/

    # 2. Load Reference Codon Usage Data (if specified)
    reference_weights, reference_data_for_plot = _load_reference_data(args)

    # 3. Load Metadata (if specified)
    # args.metadata is already a Path object from argparse type=Path
    metadata_df = _load_metadata(args.metadata, args.metadata_id_col, args.metadata_delimiter)

    # 4. Discover Gene Files and Extract Expected Gene Names
    gene_files, expected_gene_names = _get_gene_files_and_names(args.directory)

    # 5. Determine Number of Processes for Parallel Execution
    num_processes = _determine_num_processes(args.threads, len(gene_files))

    # 6. Run Gene File Analysis (in parallel or sequentially)
    # This now returns ProcessGeneFileResultType which includes per-sequence frequencies
    analyze_results_raw: List[Optional[ProcessGeneFileResultType]] = _run_gene_file_analysis_in_parallel(
        gene_files, args,
        reference_weights,
        expected_gene_names, # For context if worker needs it
        num_processes,
        output_dir_path # Pass main output dir for plot saving context
    )

    # 7. Collect and Aggregate Initial Results from Gene Processing
    (all_per_sequence_dfs, all_ca_input_dfs, successfully_processed_genes,
     _, sequences_by_original_id, # failed_genes_info is not used further here
     all_nucl_freqs_by_gene_agg, all_dinucl_freqs_by_gene_agg,
     all_nucl_freqs_per_seq_in_gene, all_dinucl_freqs_per_seq_in_gene
    ) = _collect_and_aggregate_results(analyze_results_raw, expected_gene_names)

    # 8. Analyze "Complete" (Concatenated) Sequences and Update Aggregate Data
    complete_analysis_results_tuple = (None, None, None, None, None, None, None) # Default empty tuple
    if sequences_by_original_id and successfully_processed_genes: # Ensure there's data to concat
        complete_analysis_results_tuple = _analyze_complete_sequences_cli(
            sequences_by_original_id, successfully_processed_genes, args,
            reference_weights, output_dir_path
        )
        _update_aggregate_data_with_complete_results(
            complete_analysis_results_tuple,
            all_per_sequence_dfs, all_ca_input_dfs,
            all_nucl_freqs_by_gene_agg, all_dinucl_freqs_by_gene_agg,
            all_nucl_freqs_per_seq_in_gene, all_dinucl_freqs_per_seq_in_gene
        )
    else:
        logger.info("Skipping 'complete' sequence analysis as no sequences_by_original_id or no successfully_processed_genes.")


    # 9. Finalize and Save the Main Per-Sequence Metrics Table
    # This table combines results from all individual genes and the 'complete' set.
    combined_per_sequence_df = _finalize_and_save_per_sequence_metrics(
        all_per_sequence_dfs, output_dir_path
    )
    if combined_per_sequence_df is None or combined_per_sequence_df.empty:
        logger.error("Fatal: Failed to produce the combined per-sequence metrics table. Cannot proceed. Exiting.")
        sys.exit(1)

    # 10. Merge Metadata with the Combined Per-Sequence Metrics
    combined_per_sequence_df_with_meta = combined_per_sequence_df.copy()
    if metadata_df is not None:
        logger.info("Merging metadata with analysis results...")
        if 'Original_ID' in combined_per_sequence_df_with_meta.columns:
            # Ensure merge keys are of the same type and clean
            combined_per_sequence_df_with_meta['Original_ID_str_for_merge'] = \
                combined_per_sequence_df_with_meta['Original_ID'].astype(str).str.strip()
            metadata_df.index = metadata_df.index.astype(str).str.strip()

            combined_per_sequence_df_with_meta = pd.merge(
                combined_per_sequence_df_with_meta, metadata_df,
                left_on='Original_ID_str_for_merge', right_index=True,
                how='left', suffixes=('', '_meta') # Add suffix if metadata has overlapping column names
            )
            combined_per_sequence_df_with_meta.drop(columns=['Original_ID_str_for_merge'], inplace=True, errors='ignore')
            logger.info("Metadata successfully merged with per-sequence metrics.")
        else:
            logger.warning("Could not merge metadata: 'Original_ID' column (original sequence ID) "
                           "is missing in the combined analysis results.")
    # Update the main CSV file with the merged metadata
    if metadata_df is not None and 'Original_ID' in combined_per_sequence_df.columns:
        _finalize_and_save_per_sequence_metrics( # Call again to overwrite with metadata
            [combined_per_sequence_df_with_meta], output_dir_path
        )


    # 11. Generate Plots Colored by Metadata (if specified)
    metadata_category_color_map: Optional[Dict[str, Any]] = None
    top_n_categories_map_for_report: Dict[str, str] = {}

    if args.color_by_metadata and metadata_df is not None:
        metadata_col_to_color_by = args.color_by_metadata
        if metadata_col_to_color_by not in combined_per_sequence_df_with_meta.columns:
            logger.error(f"Metadata column '{metadata_col_to_color_by}' specified for coloring not found "
                         "after merging with analysis results. Skipping metadata-colored plots.")
        else:
            # Ensure the metadata column is treated as string for proper category handling
            combined_per_sequence_df_with_meta[metadata_col_to_color_by] = \
                combined_per_sequence_df_with_meta[metadata_col_to_color_by].astype(str).fillna("Unknown")

            category_counts = combined_per_sequence_df_with_meta[metadata_col_to_color_by].value_counts()
            palette_categories_for_map: List[str]

            if len(category_counts) > args.metadata_max_categories:
                logger.info(f"Metadata column '{metadata_col_to_color_by}' has {len(category_counts)} unique categories. "
                            f"Limiting to top {args.metadata_max_categories} for plotting, others grouped as 'Other'.")
                top_categories_list = category_counts.nlargest(args.metadata_max_categories).index.tolist()

                # Create a mapping from all original categories to the top N or "Other"
                for cat_val in combined_per_sequence_df_with_meta[metadata_col_to_color_by].unique():
                    top_n_categories_map_for_report[cat_val] = cat_val if cat_val in top_categories_list else "Other"
                palette_categories_for_map = top_categories_list + (["Other"] if "Other" in top_n_categories_map_for_report.values() else [])
            else:
                for cat_val in combined_per_sequence_df_with_meta[metadata_col_to_color_by].unique():
                    top_n_categories_map_for_report[cat_val] = cat_val
                palette_categories_for_map = sorted(list(category_counts.index)) # Sort for consistent palette

            # Generate color map for these (potentially grouped) categories
            if palette_categories_for_map:
                palette_list_meta = sns.color_palette("husl", n_colors=len(palette_categories_for_map))
                metadata_category_color_map = {
                    cat: color for cat, color in zip(palette_categories_for_map, palette_list_meta)
                }

            _generate_plots_per_gene_colored_by_metadata(
                args, combined_per_sequence_df_with_meta,
                all_ca_input_dfs,
                all_nucl_freqs_per_seq_in_gene, all_dinucl_freqs_per_seq_in_gene,
                metadata_col_to_color_by, # Original column name
                top_n_categories_map_for_report, # Mapping to topN/"Other"
                output_dir_path,
                metadata_category_color_map # Palette for the mapped categories
            )

    # 12. Generate Summary Tables (Mean Features, Statistical Comparisons, Relative Dinucleotide Abundance)
    mean_summary_df, comparison_results_df, rel_abund_df_long = _generate_summary_tables_and_stats(
        combined_per_sequence_df_with_meta, # Use the metadata-merged DataFrame
        all_nucl_freqs_by_gene_agg, all_dinucl_freqs_by_gene_agg
    )

    # 13. Perform Combined Correspondence Analysis (CA) and Save Output Tables
    (combined_ca_input_df_final, ca_results_combined,
     gene_groups_for_ca_plotting, ca_row_coords_final
    ) = _perform_and_save_combined_ca(all_ca_input_dfs, output_dir_path, args)

    # 14. Prepare Standard Gene Color Palette for Combined Plots
    gene_color_map_standard = _generate_color_palette_for_groups(combined_per_sequence_df_with_meta, group_column_name='Gene')

    # 15. Generate All Combined Plots (summarizing across genes)
    _generate_all_combined_plots(
        args, combined_per_sequence_df_with_meta, gene_color_map_standard,
        rel_abund_df_long, ca_results_combined, combined_ca_input_df_final,
        gene_groups_for_ca_plotting, ca_row_coords_final,
        reference_data_for_plot,
        output_dir_path
    )

    # 16. Save Remaining Main Output Tables (mean summary, stats)
    _save_main_output_tables(
        output_dir_path,
        mean_summary_df, comparison_results_df
    )

    # 17. Generate HTML Report (if not disabled)
    if not args.no_html_report:
        if not reporting.JINJA2_AVAILABLE: # pragma: no cover
            logger.error("Cannot generate HTML report: Jinja2 library is not installed. "
                         "Please install it (e.g., 'pip install Jinja2').")
        else:
            logger.info("Preparing data and generating HTML report...")
            report_gen = reporting.HTMLReportGenerator(output_dir_path, vars(args))

            # Add summary statistics to the report
            num_processed_genes = len(successfully_processed_genes)
            total_valid_seqs = len(combined_per_sequence_df_with_meta)
            report_gen.add_summary_data(
                num_genes_processed=num_processed_genes,
                total_valid_sequences=total_valid_seqs
            )

            # Add tables to the report context
            report_gen.add_table(
                "per_sequence_metrics", combined_per_sequence_df_with_meta,
                table_csv_path_relative_to_outdir="data/per_sequence_metrics_all_genes.csv",
                display_in_html=False # Too large for direct display
            )
            # Gene sequence summary table (counts and lengths)
            gene_sequence_summary_df_for_report = None
            if 'Gene' in combined_per_sequence_df_with_meta.columns and \
               'Length' in combined_per_sequence_df_with_meta.columns:
                try:
                    gene_sequence_summary_df_for_report = combined_per_sequence_df_with_meta.groupby('Gene').agg(
                        num_sequences=('ID', 'count'),
                        mean_length=('Length', 'mean'),
                        min_length=('Length', 'min'),
                        max_length=('Length', 'max')
                    ).reset_index().rename(columns={
                        'Gene': 'Gene Name', 'num_sequences': 'Number of Sequences',
                        'mean_length': 'Mean Length (bp)', 'min_length': 'Min Length (bp)',
                        'max_length': 'Max Length (bp)'
                    })
                    # Format float columns
                    for col_fmt in ['Mean Length (bp)']:
                        if col_fmt in gene_sequence_summary_df_for_report.columns:
                             gene_sequence_summary_df_for_report[col_fmt] = \
                                 gene_sequence_summary_df_for_report[col_fmt].round(1)
                    # Save this table to data/ for linking
                    if gene_sequence_summary_df_for_report is not None and not gene_sequence_summary_df_for_report.empty:
                         summary_csv_path = output_dir_path / "data" / "gene_sequence_summary.csv"
                         gene_sequence_summary_df_for_report.to_csv(summary_csv_path, index=False, float_format='%.1f')
                         report_gen.add_table("gene_sequence_summary", gene_sequence_summary_df_for_report,
                                             table_csv_path_relative_to_outdir="data/gene_sequence_summary.csv")
                    else: # pragma: no cover
                         report_gen.add_table("gene_sequence_summary", None, None) # Add placeholder if empty
                except Exception as e: # pragma: no cover
                    logger.error(f"Could not generate gene sequence summary table for HTML report: {e}")
                    report_gen.add_table("gene_sequence_summary", None, None)


            report_gen.add_table("mean_features_per_gene", mean_summary_df,
                                 table_csv_path_relative_to_outdir="data/mean_features_per_gene.csv" if mean_summary_df is not None else None)
            report_gen.add_table("gene_comparison_stats", comparison_results_df,
                                 table_csv_path_relative_to_outdir="data/gene_comparison_stats.csv" if comparison_results_df is not None else None)

            # CA related tables
            ca_performed_for_report = ca_results_combined is not None and combined_ca_input_df_final is not None
            report_gen.set_ca_performed_status(ca_performed_for_report)
            if ca_performed_for_report:
                if ca_row_coords_final is not None:
                    report_gen.add_table("ca_combined_row_coordinates", ca_row_coords_final,
                                         table_csv_path_relative_to_outdir="data/ca_row_coordinates.csv",
                                         display_in_html=False, display_index=True)
                if hasattr(ca_results_combined, 'column_coordinates') and combined_ca_input_df_final is not None:
                    df_ca_col_coords = ca_results_combined.column_coordinates(combined_ca_input_df_final)
                    report_gen.add_table("ca_combined_col_coordinates", df_ca_col_coords,
                                         table_csv_path_relative_to_outdir="data/ca_col_coordinates.csv",
                                         display_in_html=False, display_index=True)
                if hasattr(ca_results_combined, 'column_contributions_'):
                    report_gen.add_table("ca_combined_col_contributions", ca_results_combined.column_contributions_,
                                         table_csv_path_relative_to_outdir="data/ca_col_contributions.csv",
                                         display_index=True)
                if hasattr(ca_results_combined, 'eigenvalues_summary'):
                    report_gen.add_table("ca_combined_eigenvalues", ca_results_combined.eigenvalues_summary,
                                         table_csv_path_relative_to_outdir="data/ca_eigenvalues.csv",
                                         display_index=True)
            if combined_ca_input_df_final is not None:
                 report_gen.add_table("per_sequence_rscu_wide", combined_ca_input_df_final,
                                      table_csv_path_relative_to_outdir="data/per_sequence_rscu_wide.csv",
                                      display_in_html=False, display_index=True) # Index is 'gene__origID'

            # Add paths to COMBINED plots for the report
            report_plot_format_ext = args.plot_formats[0] # Use the first specified format for the report
            def get_report_plot_path(base_filename_pattern: str, specific_format_ext: str = report_plot_format_ext) -> Optional[str]:
                # Path relative to output_dir_root, e.g., "images/plot_name.svg"
                filename_with_ext = f"{base_filename_pattern}.{specific_format_ext}"
                # Check if this exact file exists
                if (output_dir_path / "images" / filename_with_ext).exists():
                    return f"images/{filename_with_ext}"
                logger.debug(f"Report plot file not found: images/{filename_with_ext}")
                return None

            report_gen.add_plot("gc_means_barplot_by_Gene", get_report_plot_path("gc_means_barplot_by_Gene"))
            report_gen.add_plot("enc_vs_gc3_combined", get_report_plot_path(f"enc_vs_gc3_plot_{utils.sanitize_filename('_grouped_by_gene')}"))
            report_gen.add_plot("neutrality_plot_combined", get_report_plot_path(f"neutrality_plot_{utils.sanitize_filename('_grouped_by_gene')}"))
            report_gen.add_plot("relative_dinucleotide_abundance_combined", get_report_plot_path("relative_dinucleotide_abundance"))

            if ca_performed_for_report:
                ca_suffix_rep = utils.sanitize_filename('_combined_by_gene')
                report_gen.add_plot("ca_biplot_combined", get_report_plot_path(f"ca_biplot_comp{args.ca_dims[0]+1}v{args.ca_dims[1]+1}_{ca_suffix_rep}"))
                report_gen.add_plot("ca_variance_explained", get_report_plot_path(f"ca_variance_explained_top{10}"))
                report_gen.add_plot("ca_contribution_dim1", get_report_plot_path(f"ca_contribution_dim1_top{10}"))
                report_gen.add_plot("ca_contribution_dim2", get_report_plot_path(f"ca_contribution_dim2_top{10}"))

            report_gen.add_plot("feature_correlation_heatmap", get_report_plot_path(f"feature_correlation_heatmap_spearman"))
            report_gen.add_plot("ca_axes_feature_corr", get_report_plot_path(f"ca_axes_feature_corr_{utils.sanitize_filename('Spearman')}"))

            # Add paths to per-gene RSCU boxplots
            for gene_name_plot in list(successfully_processed_genes) + (["complete"] if "complete" in all_nucl_freqs_by_gene_agg else []):
                plot_fname_base_rscu = f"RSCU_boxplot_{utils.sanitize_filename(gene_name_plot)}"
                target_dict = report_gen.report_data["plot_paths"]["per_gene_rscu_boxplots"].setdefault(gene_name_plot, {})
                report_gen.add_plot(
                    plot_key="rscu_boxplot",
                    plot_path_relative_to_outdir=get_report_plot_path(plot_fname_base_rscu),
                    plot_dict_target=target_dict
                )

            # Add paths for per-gene plots colored by metadata
            if args.color_by_metadata and metadata_df is not None and metadata_category_color_map is not None:
                meta_col_report = args.color_by_metadata
                report_gen.report_data["metadata_info"]["column_used_for_coloring"] = meta_col_report
                report_gen.report_data["metadata_info"]["categories_shown"] = sorted(list(metadata_category_color_map.keys()))
                report_gen.report_data["metadata_info"]["other_category_used"] = "Other" in metadata_category_color_map

                meta_col_plot_data_for_report = report_gen.report_data["plot_paths"]["per_gene_metadata_plots"].setdefault(meta_col_report, {})
                meta_plots_image_subdir_rel = f"{utils.sanitize_filename(meta_col_report)}_per_gene_plots"

                genes_with_meta_plots_report = combined_per_sequence_df_with_meta['Gene'].unique()
                for gene_name_meta_plot_rep in genes_with_meta_plots_report:
                    gene_specific_meta_plot_subdir_rel = utils.sanitize_filename(gene_name_meta_plot_rep)
                    gene_plot_target_dict_rep = meta_col_plot_data_for_report.setdefault(gene_name_meta_plot_rep, {})
                    filename_suffix_meta_gene_rep = f"_{utils.sanitize_filename(gene_name_meta_plot_rep)}_by_{utils.sanitize_filename(meta_col_report)}"

                    plot_types_map_rep = {
                        "enc_vs_gc3": f"enc_vs_gc3_plot{filename_suffix_meta_gene_rep}",
                        "neutrality": f"neutrality_plot{filename_suffix_meta_gene_rep}",
                        "ca_biplot": f"ca_biplot_comp{args.ca_dims[0]+1}v{args.ca_dims[1]+1}{filename_suffix_meta_gene_rep}",
                        "dinucl_abundance": f"dinucl_abundance{filename_suffix_meta_gene_rep}"
                    }
                    for plot_key_rep, fname_base_rep in plot_types_map_rep.items():
                        # Full relative path for report: images/METAPLOT_SUBDIR/GENE_SUBDIR/plotname.fmt
                        plot_full_rel_path = Path("images") / meta_plots_image_subdir_rel / gene_specific_meta_plot_subdir_rel / f"{fname_base_rep}.{report_plot_format_ext}"
                        if (output_dir_path / plot_full_rel_path).exists():
                            report_gen.add_plot(plot_key_rep, str(plot_full_rel_path), plot_dict_target=gene_plot_target_dict_rep)
                        else: # pragma: no cover
                            report_gen.add_plot(plot_key_rep, None, plot_dict_target=gene_plot_target_dict_rep)
            else: # No metadata plots
                report_gen.report_data["metadata_info"]["column_used_for_coloring"] = None


            # Generate the actual HTML files
            report_gen.generate_report()
    else:
        logger.info("Skipping HTML report generation as per --no-html-report flag.")

    # 18. Create README for the output directory
    _create_output_readme(output_dir_path, args)

    logger.info(f"PyCodon Analyzer 'analyze' command finished successfully. Results in: {output_dir_path.resolve()}")


def process_analyze_gene_file(
    gene_filepath: str,
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    expected_gene_names: Set[str], # For context, not directly used by this worker currently
    output_dir_path_for_plots: Path # Main output directory (e.g., codon_analysis_results/)
) -> Optional[ProcessGeneFileResultType]:
    """
    Worker function to analyze a single gene FASTA file.
    This function is designed to be run in parallel by `_run_gene_file_analysis_in_parallel`.
    It reads sequences, cleans them, performs codon usage analysis, and prepares
    data for aggregation. Per-gene RSCU boxplots are saved directly by this worker
    into the `output_dir_path_for_plots / "images"` directory.

    Args:
        gene_filepath: Path to the gene FASTA file.
        args: Parsed command-line arguments.
        reference_weights: Pre-loaded reference codon weights.
        expected_gene_names: Set of all gene names being processed in the run (for context).
        output_dir_path_for_plots: The main output directory where "images" subdir exists.

    Returns:
        A ProcessGeneFileResultType tuple containing results for this gene,
        or None if the gene name is invalid or initial file reading fails.
    """
    # Setup matplotlib backend for non-interactive use in workers (if not already set globally)
    try:
        import matplotlib
        matplotlib.use('Agg') # Use a non-interactive backend
    except ImportError: # pragma: no cover (matplotlib should be a core dep)
        logging.getLogger(f"pycodon_analyzer.worker.{os.getpid()}").error(
            f"[Worker {os.getpid()}] Matplotlib not found. Cannot generate plots for this worker."
        )
    except Exception as backend_err: # pragma: no cover
        logging.getLogger(f"pycodon_analyzer.worker.{os.getpid()}").warning(
            f"[Worker {os.getpid()}] Could not set Matplotlib backend to 'Agg': {backend_err}"
        )

    # Get a logger specific to this worker process for better log tracking
    worker_logger = logging.getLogger(f"pycodon_analyzer.worker.{Path(gene_filepath).stem}")

    gene_name: Optional[str] = extract_gene_name_from_file(gene_filepath)
    if not gene_name: # Should not happen if _get_gene_files_and_names filters correctly
        worker_logger.error(f"Could not extract gene name from {Path(gene_filepath).name}. Skipping.")
        return None # Or a tuple indicating failure

    # Initialize return values
    per_sequence_df_gene: Optional[pd.DataFrame] = None
    ca_input_df_gene_wide: Optional[pd.DataFrame] = None # For CA, RSCU wide format
    nucl_freqs_gene_agg: Optional[Dict[str, float]] = None
    dinucl_freqs_gene_agg: Optional[Dict[str, float]] = None
    cleaned_seq_map_gene: Optional[Dict[str, Seq]] = None # {original_id: Bio.Seq object}
    per_seq_nucl_freqs_gene: Optional[Dict[str, Dict[str, float]]] = None
    per_seq_dinucl_freqs_gene: Optional[Dict[str, Dict[str, float]]] = None


    worker_logger.debug(f"Processing gene: {gene_name} (File: {Path(gene_filepath).name})")
    try:
        raw_sequences: List[SeqRecord] = io.read_fasta(gene_filepath)
        if not raw_sequences:
            worker_logger.info(f"Gene file {Path(gene_filepath).name} for '{gene_name}' is empty or unreadable.")
            return (gene_name, "empty file", None, None, None, None, None, None, None)

        cleaned_sequences: List[SeqRecord] = utils.clean_and_filter_sequences(
            raw_sequences, args.max_ambiguity
        )
        if not cleaned_sequences:
            worker_logger.info(f"No valid sequences remained after cleaning/filtering for gene '{gene_name}'.")
            return (gene_name, "no valid seqs", None, None, None, None, None, None, None)

        # Create a map of original_id to cleaned Bio.Seq object for this gene
        cleaned_seq_map_gene = {rec.id: rec.seq for rec in cleaned_sequences if rec.id}

        # Perform the core analysis for these cleaned sequences
        # run_full_analysis returns a tuple:
        # (agg_usage_df, per_sequence_df, overall_nucl_freqs, overall_dinucl_freqs,
        #  per_seq_nucl_freqs, per_seq_dinucl_freqs, _, _, ca_input_df_rscu_wide)
        analysis_results_tuple: FullAnalysisResultType = analysis.run_full_analysis(
            cleaned_sequences, args.genetic_code, reference_weights=reference_weights
        )
        # Unpack results for this specific gene
        agg_usage_df_gene, per_sequence_df_gene, nucl_freqs_gene_agg, dinucl_freqs_gene_agg, \
        per_seq_nucl_freqs_gene, per_seq_dinucl_freqs_gene, _, _, ca_input_df_gene_wide = analysis_results_tuple

        worker_logger.debug(f"Core analysis complete for gene '{gene_name}'.")

        # Generate and save per-gene RSCU boxplot if not skipping plots
        if not args.skip_plots and \
           agg_usage_df_gene is not None and not agg_usage_df_gene.empty and \
           ca_input_df_gene_wide is not None and not ca_input_df_gene_wide.empty:
            try:
                # RSCU data for boxplot needs to be in long format
                long_rscu_df_for_boxplot = ca_input_df_gene_wide.reset_index().rename(
                    columns={'index': 'SequenceID'} # Assuming index is SequenceID
                )
                long_rscu_df_for_boxplot = long_rscu_df_for_boxplot.melt(
                    id_vars=['SequenceID'], var_name='Codon', value_name='RSCU'
                )
                current_genetic_code_dict: Dict[str, str] = utils.get_genetic_code(args.genetic_code)
                long_rscu_df_for_boxplot['AminoAcid'] = long_rscu_df_for_boxplot['Codon'].map(current_genetic_code_dict.get)

                for fmt_ext_plot in args.plot_formats:
                    plot_filename_rscu = f"RSCU_boxplot_{utils.sanitize_filename(gene_name)}.{fmt_ext_plot}"
                    # Ensure plots are saved in the 'images' subdirectory of the main output path
                    rscu_boxplot_filepath_full = output_dir_path_for_plots / "images" / plot_filename_rscu
                    # The plotting function now takes the full output filepath
                    plotting.plot_rscu_boxplot_per_gene(
                        long_rscu_df=long_rscu_df_for_boxplot,
                        agg_rscu_df=agg_usage_df_gene, # Aggregate RSCU for this gene
                        gene_name=gene_name,          # For plot title
                        output_filepath=str(rscu_boxplot_filepath_full)
                    )
            except Exception as plot_err: # pragma: no cover
                worker_logger.error(f"Failed to prepare data or generate RSCU boxplot for gene '{gene_name}': {plot_err}")

        # Prepare per_sequence_df_gene for aggregation: prefix ID, add Gene column
        if per_sequence_df_gene is not None and not per_sequence_df_gene.empty:
            if 'ID' in per_sequence_df_gene.columns:
                # Store original ID before modifying for combined table
                per_sequence_df_gene['Original_ID'] = per_sequence_df_gene['ID']
                 # Prefix ID with gene_name to ensure uniqueness in combined table
                per_sequence_df_gene['ID'] = f"{gene_name}__" + per_sequence_df_gene['ID'].astype(str)
            per_sequence_df_gene['Gene'] = gene_name # Add Gene column

        # Prepare ca_input_df_gene_wide for aggregation: prefix index
        if ca_input_df_gene_wide is not None and not ca_input_df_gene_wide.empty:
            ca_input_df_gene_wide.index = f"{gene_name}__" + ca_input_df_gene_wide.index.astype(str)

        worker_logger.debug(f"Finished processing for gene: {gene_name}")
        return (gene_name, "success", per_sequence_df_gene, ca_input_df_gene_wide,
                nucl_freqs_gene_agg, dinucl_freqs_gene_agg, cleaned_seq_map_gene,
                per_seq_nucl_freqs_gene, per_seq_dinucl_freqs_gene)

    except FileNotFoundError: # pragma: no cover (should be caught by _get_gene_files_and_names)
        worker_logger.error(f"File not found during processing for gene '{gene_name}': {gene_filepath}")
        return (gene_name, "file not found error", None, None, None, None, None, None, None)
    except ValueError as ve: # pragma: no cover
        worker_logger.error(f"ValueError during processing of gene '{gene_name}': {ve}")
        return (gene_name, "value error", None, None, None, None, None, None, None)
    except Exception as e: # pragma: no cover
        worker_logger.exception(f"UNEXPECTED ERROR processing gene '{gene_name}' (file: {Path(gene_filepath).name}): {e}")
        return (gene_name, "exception", None, None, None, None, None, None, None)


def handle_extract_command(args: argparse.Namespace) -> None:
    """
    Orchestrates the 'extract' subcommand workflow.
    This involves parsing annotations, reading alignments, and extracting
    gene-specific alignments.

    Args:
        args: Parsed command-line arguments for the 'extract' subcommand.
    """
    logger.info(f"Starting 'extract' command. Annotations: {args.annotations}, "
                f"Alignment: {args.alignment}, Output Dir: {args.output_dir}")
    logger.debug(f"Full 'extract' arguments: {args}")

    # Input validation (Path objects from argparse should handle basic existence checks if type=Path used correctly)
    if not args.annotations.is_file():
        logger.error(f"Annotation file not found: {args.annotations}. Exiting.")
        sys.exit(1)
    if not args.alignment.is_file():
        logger.error(f"Alignment file not found: {args.alignment}. Exiting.")
        sys.exit(1)

    # Setup output directory
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{args.output_dir}' for extracted genes is ready.")
    except OSError as e:
        logger.error(f"Fatal: Error creating output directory '{args.output_dir}': {e}. Exiting.")
        sys.exit(1)

    try:
        # Call the main extraction function from the extraction module
        extraction.extract_gene_alignments_from_genome_msa(
            annotations_path=args.annotations,
            alignment_path=args.alignment,
            ref_id=args.ref_id,
            output_dir=args.output_dir
        )
        logger.info(f"'extract' command finished successfully. Extracted genes in: {args.output_dir.resolve()}")
    except FileNotFoundError as fnf_err: # Should be caught by earlier checks
        logger.error(f"Fatal extraction error (file not found): {fnf_err}")
        sys.exit(1)
    except ValueError as val_err: # For parsing or data integrity errors from extraction module
        logger.error(f"Fatal extraction error (value error): {val_err}")
        sys.exit(1)
    except Exception as e: # pragma: no cover
        logger.exception(f"Unexpected fatal error during 'extract' command: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the PyCodon Analyzer CLI.
    Sets up argument parsing, configures logging, and dispatches to subcommand handlers.
    """
    # --- Main Argument Parser ---
    parser = argparse.ArgumentParser(
        description="PyCodon Analyzer: A tool for codon usage analysis and gene alignment extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="For detailed help on a specific subcommand, run: pycodon_analyzer <subcommand> --help"
    )
    # Global arguments applicable to all subcommands
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase output verbosity to DEBUG level for console and file logs."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="pycodon_analyzer.log", # Default log file name
        help="Name for the log file. It will be created in the respective output directory "
             "of the 'analyze' or 'extract' command."
    )
    try:
        from . import __version__ as pkg_version
    except ImportError: # pragma: no cover
        pkg_version = "unknown"
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {pkg_version}'
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True, # Ensures a subcommand must be provided
        title="Available subcommands",
        description="Select one of the following actions:"
    )

    # --- Sub-parser for 'analyze' command ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze codon usage from pre-extracted gene alignment FASTA files.",
        description="Performs comprehensive codon usage analysis on a directory of "
                    "individual gene FASTA alignments (e.g., 'gene_GENENAME.fasta'). "
                    "Results include various metrics, plots, and an HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    analyze_parser.add_argument(
        "-d", "--directory",
        required=True,
        type=str, # Handled as Path later
        help="Path to the input directory containing gene alignment files "
             "(e.g., 'gene_XYZ.fasta')."
    )
    analyze_parser.add_argument(
        "-o", "--output",
        default="codon_analysis_results",
        type=str, # Handled as Path later
        help="Path to the output directory where analysis results will be saved."
    )
    analyze_parser.add_argument(
        "--genetic_code",
        type=int, default=1,
        help="NCBI genetic code ID to use for translation and analysis (Default: 1, Standard Code)."
    )
    analyze_parser.add_argument(
        "--ref", "--reference_usage_file", # Added alias
        dest="reference_usage_file", # Ensure correct dest
        type=str,
        default=DEFAULT_HUMAN_REF_PATH if DEFAULT_HUMAN_REF_PATH else "human", # Use evaluated default
        help="Reference codon usage table for metrics like CAI. Accepts 'human' (bundled), "
             "'none' (to skip reference-based metrics), or a path to a custom CSV/TSV file."
    )
    analyze_parser.add_argument(
        "--ref_delimiter",
        type=str,
        default=None,
        help="Delimiter for the custom reference codon usage file (e.g., ',', '\\t'). "
             "If not provided, attempts to auto-detect."
    )
    analyze_parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="Number of processor cores to use for parallel gene file analysis. "
             "Default is 1 (sequential). Use 0 or a negative value to utilize all available cores."
    )
    analyze_parser.add_argument(
        "--max_ambiguity",
        type=float,
        default=15.0,
        metavar="PERCENT",
        help="Maximum allowed percentage of ambiguous 'N' characters per sequence "
             "for it to be included in the analysis (0-100)."
    )
    analyze_parser.add_argument(
        "--plot_formats",
        nargs='+', # Allows multiple formats
        default=['svg'],
        choices=['svg', 'png', 'pdf', 'jpg', 'tiff'], # Added tiff
        type=str.lower,
        help="Output format(s) for generated plots (e.g., 'svg png pdf'). "
             "The first format is primarily used in the HTML report."
    )
    analyze_parser.add_argument(
        "--skip-plots", # Consistent hyphenation
        action='store_true',
        help="If set, disables the generation of all plot images."
    )
    analyze_parser.add_argument(
        "--ca_dims",
        nargs=2,
        type=int,
        default=[0, 1], # Corresponds to Dim1 and Dim2
        metavar=('X_DIM_IDX', 'Y_DIM_IDX'),
        help="Indices (0-based) of the Correspondence Analysis components to use for "
             "the combined CA biplot (e.g., '0 1' for Dimension 1 vs. Dimension 2)."
    )
    analyze_parser.add_argument(
        "--ca_components",
        type=int,
        default=10,
        metavar="N_COMP",
        help="Number of principal components to compute for Correspondence Analysis."
    )
    analyze_parser.add_argument(
        "--skip-ca", # Consistent hyphenation
        action='store_true',
        help="If set, skips the combined Correspondence Analysis."
    )
    # Metadata arguments for 'analyze'
    analyze_parser.add_argument(
        "--metadata",
        type=Path, # Use pathlib.Path for type conversion
        default=None,
        help="Optional path to a metadata file (CSV or TSV format) to associate with sequences. "
             "This enables stratified analysis and plotting."
    )
    analyze_parser.add_argument(
        "--metadata_id_col",
        type=str,
        default="seq_id",
        help="Name of the column in the metadata file that contains unique sequence identifiers "
             "matching those in the input FASTA files."
    )
    analyze_parser.add_argument(
        "--metadata_delimiter",
        type=str,
        default=None,
        help="Delimiter for the metadata file (e.g., ',', '\\t'). Auto-detects if not specified."
    )
    analyze_parser.add_argument(
        "--color_by_metadata",
        type=str,
        default=None,
        metavar="METADATA_COLUMN",
        help="If --metadata is provided, specify a categorical column name from the metadata "
             "to use for coloring points/lines in per-gene plots. These plots are saved "
             "in a dedicated subdirectory within 'images'."
    )
    analyze_parser.add_argument(
        "--metadata_max_categories",
        type=int,
        default=15, # Sensible default
        help="When using --color_by_metadata, if the specified column has more unique categories "
             "than this value, plots will show the top N most frequent categories and group the rest "
             "into an 'Other' category."
    )
    analyze_parser.add_argument(
        "--no-html-report",
        action="store_true",
        default=False, # Default is to generate the report
        help="If set, disables the generation of the final HTML report."
    )
    analyze_parser.set_defaults(func=handle_analyze_command)

    # --- Sub-parser for 'extract' command ---
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract individual gene alignments from a whole genome MSA.",
        description="Uses a reference annotation file (FASTA with GenBank-style feature tags) "
                    "and a whole genome multiple sequence alignment (MSA) to extract "
                    "alignments for each annotated gene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    extract_parser.add_argument(
        "-a", "--annotations",
        type=Path, # Use pathlib.Path
        required=True,
        help="Path to the reference gene annotation file. Expected format is multi-FASTA "
             "where sequence headers contain [gene=NAME] or [locus_tag=TAG] and "
             "[location=START..END] tags."
    )
    extract_parser.add_argument(
        "-g", "--alignment",
        type=Path, # Use pathlib.Path
        required=True,
        help="Path to the whole genome multiple sequence alignment file (FASTA format)."
    )
    extract_parser.add_argument(
        "-r", "--ref_id",
        type=str,
        required=True,
        help="Sequence ID of the reference genome as it appears in the alignment file. "
             "This sequence is used for coordinate mapping."
    )
    extract_parser.add_argument(
        "-o", "--output_dir", # Consistent naming with analyze's output
        type=Path, # Use pathlib.Path
        required=True,
        help="Output directory where extracted gene alignment FASTA files "
             "(e.g., 'gene_GENENAME.fasta') will be saved."
    )
    extract_parser.set_defaults(func=handle_extract_command)

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Configure Logging ---
    # Determine console log level based on --verbose flag
    console_log_level = logging.DEBUG if args.verbose else logging.INFO

    # Use RichHandler if available and stdout is a TTY, otherwise standard StreamHandler
    if RICH_AVAILABLE and sys.stderr.isatty(): # pragma: no cover
        console_handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_path=False, # Cleaner log messages
            markup=True,
            show_level=True, # Show level (INFO, DEBUG, etc.)
            log_time_format="[%X]" # Time format like [12:34:56]
        )
        # RichHandler handles its own formatting typically, but a basic formatter can be set
        # console_formatter = logging.Formatter("%(message)s") # For Rich, often keep it simple
        # console_handler.setFormatter(console_formatter)
    else: # pragma: no cover (fallback for no rich or non-TTY)
        if not RICH_AVAILABLE and args.verbose: # Notify if rich is missing but verbose is on
             print("INFO: 'rich' library not found. Console logging will be basic.", file=sys.stderr)
        console_handler = logging.StreamHandler(sys.stderr)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-7s] %(name)-20s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_log_level)

    # Configure the application's root logger
    # Clear existing handlers to prevent duplicate logging if main() is called multiple times (e.g., in tests)
    if logger.hasHandlers(): # pragma: no cover
        for h_exist in list(logger.handlers):
            logger.removeHandler(h_exist)
            h_exist.close()

    logger.addHandler(console_handler)
    # Set logger level to DEBUG to capture all messages for file log,
    # console handler will filter based on its own level (console_log_level).
    logger.setLevel(logging.DEBUG)
    logger.propagate = False # Prevent messages from going to the Python root logger

    # --- File Handler Setup (must happen after output directory is known/created) ---
    log_file_path_final: Optional[Path] = None
    if args.command == "analyze":
        # For 'analyze', output dir is args.output
        output_dir_for_log_setup = _setup_output_directory(args.output) # Ensure dir exists
        log_file_path_final = output_dir_for_log_setup / args.log_file
    elif args.command == "extract":
        # For 'extract', output dir is args.output_dir (already a Path object)
        try: # Ensure dir exists, though handle_extract_command also does this
            args.output_dir.mkdir(parents=True, exist_ok=True)
            log_file_path_final = args.output_dir / args.log_file
        except OSError as e: # pragma: no cover
            logger.error(f"Fatal: Could not create output directory '{args.output_dir}' for log file: {e}. Exiting.")
            sys.exit(1)


    if log_file_path_final:
        try:
            # Use RotatingFileHandler for better log management over time, though simple FileHandler is also fine.
            # For simplicity here, using FileHandler with mode 'w' to overwrite per run.
            file_handler = logging.FileHandler(log_file_path_final, mode='w', encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)-7s] %(name)-25s %(funcName)-20s L%(lineno)-4d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG) # Capture all levels (DEBUG and above) in the file
            logger.addHandler(file_handler)
            logger.info(f"Logging initialized. Console level: {logging.getLevelName(console_log_level)}, "
                        f"File log (DEBUG) at: {log_file_path_final.resolve()}")
        except Exception as e_log: # pragma: no cover
            logger.error(f"CRITICAL: Failed to set up file logging to {log_file_path_final}: {e_log}", exc_info=False)
            # Continue with console logging only.

    logger.info(f"PyCodon Analyzer v{pkg_version} - Executing command: '{args.command}'")
    if args.verbose: # Log full args only if verbose, to avoid cluttering INFO level
        logger.debug(f"Full command-line arguments: {vars(args)}")

    # --- Execute the Subcommand Function ---
    if hasattr(args, 'func') and callable(args.func):
        try:
            args.func(args)
        except SystemExit: # Let SystemExit propagate (e.g., from input validation errors)
            raise
        except Exception as e_runtime: # pragma: no cover
            logger.critical(f"A critical error occurred during execution of command '{args.command}'.")
            logger.critical("Please check the log file for more details if available.")
            # Log the full traceback to the logger (which includes file log if set up)
            logger.exception(f"Traceback for critical error in '{args.command}': {e_runtime}")
            sys.exit(1) # Exit with error status
    else: # pragma: no cover (should not happen if subparsers are required)
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__': # pragma: no cover
    main()