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
from typing import List, Dict, Optional, Tuple, Set, Any, Counter # <-- Import typing helpers
import seaborn as sns
from pathlib import Path

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

# Import necessary functions from local modules
# Assuming type checkers can find these or using forward references if needed
from . import io
from . import analysis
from . import plotting
from . import utils
from . import extraction
from .utils import load_reference_usage, get_genetic_code, clean_and_filter_sequences

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

def process_analyze_gene_file(
    gene_filepath: str,
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    expected_gene_names: Set[str]
) -> Optional[AnalyzeGeneResultType]:
    """
    Worker function for the 'analyze' subcommand.
    Reads a single gene FASTA file, cleans sequences, performs codon usage analysis,
    and optionally generates a per-gene RSCU boxplot.

    This function is designed to be called by multiprocessing.Pool.map or .imap.
    It sets the Matplotlib backend to 'Agg' to prevent GUI issues in worker processes.

    Args:
        gene_filepath (str): Absolute path to the gene_*.fasta file.
        args (argparse.Namespace): Parsed command-line arguments, providing access
                                   to settings like genetic_code, max_ambiguity,
                                   output directory, plot formats, and skip_plots.
        reference_weights (Optional[Dict[str, float]]): Pre-calculated reference codon
                                                       weights for CAI/Fop/RCDI.
                                                       None if no reference is used.
        expected_gene_names (Set[str]): A set of gene names expected to be processed,
                                        used for validating extracted gene names.

    Returns:
        Optional[AnalyzeGeneResultType]: A tuple containing the gene name, processing status,
                                         and various analysis result DataFrames/Dicts if successful.
                                         Returns None if the gene file is skipped (e.g., name mismatch).
                                         If an error occurs, status will indicate failure, and data
                                         elements in the tuple will be None.
    """
    # Set Matplotlib backend to non-interactive *within the worker process*
    # This is crucial for stability when plotting in parallel without a GUI.
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        # This should ideally not happen if matplotlib is a core dependency
        logging.getLogger("pycodon_analyzer.worker").error(f"[Worker {os.getpid()}] Matplotlib not found. Cannot generate plots.")
    except Exception as backend_err:
        logging.getLogger("pycodon_analyzer.worker").warning(f"[Worker {os.getpid()}] Could not set Matplotlib backend to 'Agg': {backend_err}")

    # Use a specific logger for worker messages for easier filtering if needed
    worker_logger = logging.getLogger(f"pycodon_analyzer.worker.{os.getpid()}")
    # pid_prefix = f"[Process {os.getpid()}]" # Alternative to logger name

    gene_name: Optional[str] = extract_gene_name_from_file(gene_filepath) # Use utils directly

    if not gene_name or gene_name not in expected_gene_names:
        # This file was not in the initial list of valid gene files, skip it silently
        # or with a very low-level debug message if necessary.
        return None

    worker_logger.debug(f"Starting processing for gene: {gene_name} (File: {os.path.basename(gene_filepath)})")

    # Initialize variables that will hold results from analysis
    agg_usage_df_gene: Optional[pd.DataFrame] = None
    per_sequence_df_gene: Optional[pd.DataFrame] = None
    nucl_freqs_gene: Optional[Dict[str, float]] = None
    dinucl_freqs_gene: Optional[Dict[str, float]] = None
    ca_input_df_gene: Optional[pd.DataFrame] = None # Holds RSCU per sequence for this gene
    cleaned_seq_map: Optional[Dict[str, Seq]] = None

    try:
        # 1. Read FASTA file
        worker_logger.debug(f"Reading FASTA file: {gene_filepath}")
        raw_sequences: List[SeqRecord] = io.read_fasta(gene_filepath) # io.read_fasta now handles its own FileNotFoundError/ValueError logging
        if not raw_sequences: # Should be caught by read_fasta if it raises ValueError for no sequences
            worker_logger.warning(f"No sequences found in {os.path.basename(gene_filepath)} for gene {gene_name}.")
            return (gene_name, "empty file", None, None, None, None, None)

        # 2. Clean and Filter Sequences
        worker_logger.debug(f"Cleaning and filtering {len(raw_sequences)} sequences for {gene_name} (Max N: {args.max_ambiguity}%)...")
        cleaned_sequences: List[SeqRecord] = utils.clean_and_filter_sequences(raw_sequences, args.max_ambiguity)

        if not cleaned_sequences:
            worker_logger.warning(f"No valid sequences remaining after cleaning/filtering for gene {gene_name}. Skipping analysis and plotting for this gene.")
            return (gene_name, "no valid seqs", None, None, None, None, None)
        cleaned_seq_map = {rec.id: rec.seq for rec in cleaned_sequences}
        worker_logger.debug(f"Retained {len(cleaned_sequences)} sequences after cleaning for {gene_name}.")

        # 3. Run Full Analysis (does not fit CA model internally)
        worker_logger.debug(f"Running core analysis for {len(cleaned_sequences)} sequences of gene {gene_name}...")
        analysis_results: tuple = analysis.run_full_analysis(
            cleaned_sequences,
            args.genetic_code,
            reference_weights=reference_weights # This is the Optional[Dict[str, float]]
            # No fit_ca_model argument passed to this version of run_full_analysis
        )
        # Unpack all results from run_full_analysis
        agg_usage_df_gene = analysis_results[0]      # Aggregate RSCU, Freq for this gene
        per_sequence_df_gene = analysis_results[1]   # Metrics per sequence for this gene
        nucl_freqs_gene = analysis_results[2]        # Nucleotide freqs for this gene
        dinucl_freqs_gene = analysis_results[3]      # Dinucleotide freqs for this gene
        # analysis_results[4] is None (placeholder for old reference_data)
        # analysis_results[5] is None (placeholder for CA results object)
        ca_input_df_gene = analysis_results[6]       # RSCU per sequence (wide format) for this gene

        worker_logger.debug(f"Core analysis complete for {gene_name}.")

        # --- 4. Generate Per-Gene RSCU Boxplot (if not skipped) ---
        if not args.skip_plots:
            worker_logger.debug(f"Preparing and generating RSCU boxplot for {gene_name}...")
            if agg_usage_df_gene is not None and not agg_usage_df_gene.empty and \
               ca_input_df_gene is not None and not ca_input_df_gene.empty:
                try:
                    # Prepare long format RSCU data from ca_input_df_gene for boxplotting
                    # The ca_input_df_gene has an index like 'SeqID' (from analyze_single_sequence)
                    # No gene prefix is on it yet.
                    long_rscu_df = ca_input_df_gene.reset_index().rename(columns={'index': 'SequenceID'})
                    long_rscu_df = long_rscu_df.melt(id_vars=['SequenceID'], var_name='Codon', value_name='RSCU')

                    current_genetic_code: Dict[str, str] = utils.get_genetic_code(args.genetic_code)
                    long_rscu_df['AminoAcid'] = long_rscu_df['Codon'].map(current_genetic_code.get)
                    long_rscu_df['Gene'] = gene_name # Add gene column for consistency if plot func uses it

                    # Call plotting function for each format
                    for fmt in args.plot_formats:
                        worker_logger.debug(f"Saving RSCU boxplot for {gene_name} as {fmt}...")
                        try:
                            plotting.plot_rscu_boxplot_per_gene(
                                long_rscu_df,       # Data for boxplot distributions
                                agg_usage_df_gene,  # Data for coloring labels (mean RSCU)
                                gene_name,
                                args.output,
                                fmt
                                # verbose argument removed from plotting functions
                            )
                        except Exception as plot_fmt_err:
                            worker_logger.error(f"Failed to generate/save RSCU boxplot for {gene_name} (format: {fmt}): {plot_fmt_err}")
                except Exception as plot_prep_err:
                    worker_logger.error(f"Failed to prepare data for RSCU boxplot for {gene_name}: {plot_prep_err}")
            else:
                worker_logger.warning(f"Skipping RSCU boxplot generation for {gene_name} due to missing analysis data (agg_usage_df or ca_input_df).")
        # --- End Plotting ---

        # 5. Prepare results for return to the main process
        # Modify per_sequence_df IDs to include gene name
        if per_sequence_df_gene is not None and not per_sequence_df_gene.empty:
            if 'ID' in per_sequence_df_gene.columns: # Should always be true
                 per_sequence_df_gene['Original_ID'] = per_sequence_df_gene['ID'] # Preserve original sequence ID
                 per_sequence_df_gene['ID'] = per_sequence_df_gene['ID'].astype(str) + "_" + gene_name # Create unique ID
            per_sequence_df_gene['Gene'] = gene_name
        else: # Handle case where no per-sequence metrics were generated
            per_sequence_df_gene = None # Ensure it's None if empty or None

        # Add gene prefix to ca_input_df_gene index for combined CA later
        if ca_input_df_gene is not None and not ca_input_df_gene.empty:
            ca_input_df_gene.index = f"{gene_name}__" + ca_input_df_gene.index.astype(str)
        else: # Handle case where no CA input data was generated
            ca_input_df_gene = None # Ensure it's None if empty or None


        worker_logger.debug(f"Finished processing for gene: {gene_name}")
        # Return tuple matching AnalyzeGeneResultType
        return (gene_name, "success", per_sequence_df_gene, ca_input_df_gene,
                nucl_freqs_gene, dinucl_freqs_gene, cleaned_seq_map)

    except FileNotFoundError: # Should be caught by io.read_fasta, but for safety
        worker_logger.error(f"File not found error for gene {gene_name}: {gene_filepath}")
        return (gene_name, "file not found error", None, None, None, None, None)
    except ValueError as ve: # Catch parsing errors or other value errors from analysis
        worker_logger.error(f"ValueError processing gene {gene_name}: {ve}")
        return (gene_name, "value error", None, None, None, None, None)
    except Exception as e:
        worker_logger.exception(f"UNEXPECTED ERROR processing gene {gene_name} (file {os.path.basename(gene_filepath)}): {e}")
        return (gene_name, "exception", None, None, None, None, None)

def handle_analyze_command(args: argparse.Namespace) -> None:
    """
    Handles the 'analyze' subcommand and its arguments.
    This function orchestrates the main codon usage analysis workflow, including:
    - Loading reference data.
    - Finding and processing gene files (potentially in parallel).
    - Collecting and aggregating results from individual gene analyses.
    - Analyzing concatenated "complete" sequences.
    - Calculating mean features, statistics, and dinucleotide abundances.
    - Performing combined Correspondence Analysis (CA).
    - Generating output tables and combined plots.

    Args:
        args (argparse.Namespace): Parsed command-line arguments specific
                                   to the 'analyze' subcommand. These include
                                   input directory, output directory, genetic code,
                                   reference file, threading options, filtering criteria,
                                   and plotting/CA skipping flags.
    """
    logger.info(f"Running 'analyze' command with input directory: {args.directory}")
    logger.info(f"Output will be saved to: {args.output}")

    # --- Argument Validation & Setup (mostly handled in main, but ensure paths are absolute if needed) ---
    # Output directory should have been created by the main CLI setup.
    # Convert relative paths from args to absolute if necessary, or ensure downstream functions handle it.
    # For instance, args.output should be an absolute path or robustly handled by plotting functions.

    # Setup output directory
    try:
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"Results will be saved to: {args.output}")
    except OSError as e:
        logger.error(f"Error creating output directory '{args.output}': {e}")
        sys.exit(1)

    # --- Load Reference Usage File ---
    reference_weights: Optional[Dict[str, float]] = None
    reference_data_for_plot: Optional[pd.DataFrame] = None # For plot_usage_comparison

    if args.reference_usage_file and args.reference_usage_file.lower() != 'none':
        ref_path_to_load: Optional[str] = None
        if args.reference_usage_file.lower() == 'human':
            ref_path_to_load = DEFAULT_HUMAN_REF_PATH # Uses the globally resolved path
            if not ref_path_to_load or not os.path.isfile(ref_path_to_load):
                 logger.warning("Default human reference file ('human') requested but not found or path invalid. Check installation or provide path.")
                 ref_path_to_load = None
        elif os.path.isfile(args.reference_usage_file):
            ref_path_to_load = args.reference_usage_file
        else:
            logger.warning(f"Specified reference file not found: {args.reference_usage_file}.")

        if ref_path_to_load:
            logger.info(f"Loading codon usage reference table: {ref_path_to_load}...")
            try:
                current_genetic_code: Dict[str, str] = utils.get_genetic_code(args.genetic_code)
                reference_data_for_plot = utils.load_reference_usage(ref_path_to_load, current_genetic_code, args.genetic_code)

                if reference_data_for_plot is not None and not reference_data_for_plot.empty:
                    if 'Weight' in reference_data_for_plot.columns:
                        reference_weights = reference_data_for_plot['Weight'].to_dict()
                        logger.info("Reference data loaded and weights extracted successfully.")
                    else:
                        logger.error("'Weight' column missing in loaded reference data. CAI/Fop/RCDI may fail or produce NaNs.")
                        reference_data_for_plot = None
                else:
                    logger.warning("Failed to process reference data after reading. CAI/Fop/RCDI will use NaN.")
                    reference_data_for_plot = None # Ensure it's None if processing failed
            except (NotImplementedError, ValueError) as ref_err: # Catch errors from get_genetic_code or load_reference_usage
                 logger.error(f"Error loading/processing reference file {ref_path_to_load}: {ref_err}")
                 reference_weights = None; reference_data_for_plot = None
            except Exception as load_err: # Catch other unexpected errors
                 logger.exception(f"Unexpected error loading reference file {ref_path_to_load}: {load_err}")
                 reference_weights = None; reference_data_for_plot = None
    else:
         logger.info("No reference file specified. Reference-based calculations (CAI/Fop/RCDI) will be skipped or result in NaN.")

    # --- Determine number of processes ---
    num_file_processes: int = args.threads
    if num_file_processes <= 0:
        if MP_AVAILABLE:
            try: 
                num_file_processes = os.cpu_count() or 1
            except NotImplementedError: 
                num_file_processes = 1
                logger.warning("Could not determine CPU count, using 1 process.")
        else: 
            num_file_processes = 1
    if num_file_processes > 1 and not MP_AVAILABLE:
        logger.warning(f"Requested {num_file_processes} processes, but multiprocessing is not available. Using 1 process.")
        num_file_processes = 1
    # Initial log, actual number might be capped by len(gene_files) later
    logger.info(f"User requested/configured {num_file_processes} process(es) for gene file analysis.")

    # --- Find gene alignment files ---
    logger.info(f"Searching for gene_*.fasta files in: {args.directory}")
    gene_files: List[str] = glob.glob(os.path.join(args.directory, "gene_*.*"))
    valid_extensions: Set[str] = {".fasta", ".fa", ".fna", ".fas", ".faa"} # Define valid extensions
    gene_files = sorted([f for f in gene_files if os.path.splitext(f)[1].lower() in valid_extensions])

    if not gene_files:
        logger.error(f"No gene alignment files (matching 'gene_*' with valid extensions) found in directory: {args.directory}")
        sys.exit(1)

    # Cap number of processes by the number of files
    original_requested_processes = num_file_processes
    num_file_processes = min(num_file_processes, len(gene_files))
    if num_file_processes < original_requested_processes and original_requested_processes > 1:
        logger.info(f"Adjusted number of processes to {num_file_processes} (number of files found).")
    logger.info(f"Found {len(gene_files)} potential gene files to process with {num_file_processes} process(es).")


    # --- Determine expected gene names ---
    expected_gene_names: Set[str] = set()
    for fpath in gene_files:
        gname = extract_gene_name_from_file(fpath)
        if gname: expected_gene_names.add(gname)
    if not expected_gene_names:
        logger.error("Could not extract any valid gene names from input filenames. Check naming pattern (gene_NAME.fasta).")
        sys.exit(1)
    logger.info(f"Expecting data for {len(expected_gene_names)} genes: {', '.join(sorted(list(expected_gene_names)))}")

    # --- Parallel/Sequential Processing of Gene Files ---
    processing_task = partial(process_analyze_gene_file, args=args, reference_weights=reference_weights, expected_gene_names=expected_gene_names)
    analyze_results_raw: List[Optional[AnalyzeGeneResultType]] = []

    progress_columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TextColumn("({task.completed}/{task.total} genes)"), TimeElapsedColumn(), TextColumn("<"), TimeRemainingColumn()]
    disable_rich = not sys.stderr.isatty() or not RICH_AVAILABLE

    with Progress(*progress_columns, transient=False, disable=disable_rich) as progress:
        analysis_task_id = progress.add_task("Analyzing Gene Files", total=len(gene_files))
        if num_file_processes > 1 and MP_AVAILABLE:
            try:
                with mp.Pool(processes=num_file_processes) as pool: # Removed initializer for now, pass queue directly
                    results_iterator = pool.imap(processing_task, gene_files)
                    for result in results_iterator: 
                        analyze_results_raw.append(result)
                        progress.update(analysis_task_id, advance=1)
            except Exception as pool_err:
                 logger.exception(f"Parallel analysis error: {pool_err}. Fallback to sequential.");
                 for gene_file in gene_files: 
                    analyze_results_raw.append(processing_task(gene_file))
                    progress.update(analysis_task_id, advance=1) # Manually update progress
        else:
             logger.info(f"Executing analysis sequentially (MP not available or threads=1).")
             for gene_file in gene_files: 
                analyze_results_raw.append(processing_task(gene_file))
                progress.update(analysis_task_id, advance=1)

    # --- Collect and Process Analysis Results ---
    logger.info("Collecting and aggregating analysis results...")
    all_per_sequence_dfs: List[pd.DataFrame] = []
    all_ca_input_dfs: Dict[str, pd.DataFrame] = {} # For combined CA
    processed_genes: Set[str] = set()
    failed_genes: List[str] = []
    sequences_by_original_id: Dict[str, Dict[str, Seq]] = {}
    all_nucl_freqs_by_gene: Dict[str, Dict[str, float]] = {}
    all_dinucl_freqs_by_gene: Dict[str, Dict[str, float]] = {}

    for result in analyze_results_raw:
        if result is None: 
            continue
        (gene_name_res, status, per_seq_df, ca_input_df,
         nucl_freqs, dinucl_freqs, cleaned_map) = result

        if status == "success":
            gene_name_str = str(gene_name_res)
            processed_genes.add(gene_name_str)
            if per_seq_df is not None: 
                all_per_sequence_dfs.append(per_seq_df)
            if ca_input_df is not None: 
                all_ca_input_dfs[gene_name_str] = ca_input_df
            if nucl_freqs: 
                all_nucl_freqs_by_gene[gene_name_str] = nucl_freqs
            if dinucl_freqs: 
                all_dinucl_freqs_by_gene[gene_name_str] = dinucl_freqs
            if cleaned_map:
                for seq_id, seq_obj in cleaned_map.items():
                    if seq_id not in sequences_by_original_id: 
                        sequences_by_original_id[seq_id] = {}
                    sequences_by_original_id[seq_id][gene_name_str] = seq_obj
        else:
            failed_genes.append(f"{gene_name_res or 'UnknownGene'} ({status})")

    # --- Post-processing Check ---
    successfully_processed_genes: Set[str] = processed_genes
    if not successfully_processed_genes:
         logger.error("No genes were successfully processed. Exiting.")
         sys.exit(1)
    elif len(successfully_processed_genes) < len(expected_gene_names):
         logger.warning(f"Processed {len(successfully_processed_genes)} genes out of {len(expected_gene_names)} expected.")
         if failed_genes: 
            logger.warning(f"  Failed genes/reasons: {'; '.join(failed_genes)}")

    # Per-gene RSCU boxplots are now generated in the worker process.

    # --- Analyze "Complete" Sequences ---
    logger.info("Analyzing concatenated 'complete' sequences...")
    complete_seq_records_to_analyze: List[SeqRecord] = []
    agg_usage_df_complete: Optional[pd.DataFrame] = None
    ca_input_df_complete_for_plot: Optional[pd.DataFrame] = None # For the 'complete' RSCU boxplot

    if sequences_by_original_id:
        max_ambiguity_pct_complete: float = args.max_ambiguity
        for original_id, gene_seq_map in sequences_by_original_id.items():
            if set(gene_seq_map.keys()) == successfully_processed_genes:
                try:
                    concatenated_seq_str = "".join(str(gene_seq_map[g_name]) for g_name in sorted(gene_seq_map.keys()))
                    if concatenated_seq_str and len(concatenated_seq_str) % 3 == 0:
                        n_count = concatenated_seq_str.count('N'); seq_len = len(concatenated_seq_str)
                        ambiguity_pct = (n_count / seq_len) * 100 if seq_len > 0 else 0
                        if ambiguity_pct <= max_ambiguity_pct_complete:
                            complete_record = SeqRecord(Seq(concatenated_seq_str), id=original_id, description=f"Concatenated {len(gene_seq_map)} genes")
                            complete_seq_records_to_analyze.append(complete_record)
                        else: 
                            logger.debug(f"Complete seq for {original_id} skipped (ambiguity {ambiguity_pct:.1f}% > {max_ambiguity_pct_complete}%)")
                    else: 
                        logger.debug(f"Complete seq for {original_id} skipped (invalid length or empty)")
                except Exception as concat_err: 
                    logger.warning(f"Error concatenating sequence for ID {original_id}: {concat_err}")

    if complete_seq_records_to_analyze:
        logger.info(f"Running analysis on {len(complete_seq_records_to_analyze)} valid 'complete' sequence records...")
        try:
            analysis_res_complete = analysis.run_full_analysis(complete_seq_records_to_analyze, 
                                                               args.genetic_code, 
                                                               reference_weights=reference_weights)
            agg_usage_df_complete = analysis_res_complete[0]
            per_sequence_df_complete = analysis_res_complete[1]
            nucl_freqs_complete = analysis_res_complete[2]
            dinucl_freqs_complete = analysis_res_complete[3]
            ca_input_df_complete_for_plot = analysis_res_complete[6] # For RSCU plot for 'complete'
            # For combined CA, use a copy and prefix index
            ca_input_df_complete_for_combined = analysis_res_complete[6].copy() if analysis_res_complete[6] is not None else None

            if nucl_freqs_complete: 
                all_nucl_freqs_by_gene['complete'] = nucl_freqs_complete
            if dinucl_freqs_complete: 
                all_dinucl_freqs_by_gene['complete'] = dinucl_freqs_complete
            if per_sequence_df_complete is not None and not per_sequence_df_complete.empty:
                if 'ID' in per_sequence_df_complete.columns: 
                    per_sequence_df_complete['Original_ID'] = per_sequence_df_complete['ID']
                    per_sequence_df_complete['ID'] = per_sequence_df_complete['ID'].astype(str) + "_complete"
                per_sequence_df_complete['Gene'] = 'complete'
                all_per_sequence_dfs.append(per_sequence_df_complete)
            if ca_input_df_complete_for_combined is not None and not ca_input_df_complete_for_combined.empty:
                ca_input_df_complete_for_combined.index = "complete__" + ca_input_df_complete_for_combined.index.astype(str)
                all_ca_input_dfs['complete'] = ca_input_df_complete_for_combined

            # Generate 'complete' RSCU boxplot sequentially here
            if not args.skip_plots:
                logger.info("Generating RSCU boxplot for 'complete' data...")
                if agg_usage_df_complete is not None and ca_input_df_complete_for_plot is not None:
                    try:
                        long_rscu_df_comp = ca_input_df_complete_for_plot.reset_index().rename(columns={'index': 'SequenceID'})
                        long_rscu_df_comp = long_rscu_df_comp.melt(id_vars=['SequenceID'], 
                                                                   var_name='Codon', 
                                                                   value_name='RSCU')
                        current_gc_dict = utils.get_genetic_code(args.genetic_code)
                        long_rscu_df_comp['AminoAcid'] = long_rscu_df_comp['Codon'].map(current_gc_dict.get)
                        long_rscu_df_comp['Gene'] = 'complete'
                        for fmt in args.plot_formats:
                            plotting.plot_rscu_boxplot_per_gene(long_rscu_df_comp, 
                                                                agg_usage_df_complete, 
                                                                'complete', 
                                                                args.output, 
                                                                fmt)
                    except Exception as comp_plot_err: 
                        logger.error(f"Failed to generate 'complete' RSCU boxplot: {comp_plot_err}")
                else: 
                    logger.warning("Cannot generate 'complete' RSCU boxplot due to missing analysis data.")
        except Exception as e: 
            logger.exception(f"Error during 'complete' sequence analysis: {e}")
    else: 
        logger.info("No valid 'complete' sequences to analyze.")

    # --- Final Combination and Saving ---
    logger.info("Combining final results...")
    if not all_per_sequence_dfs: 
        logger.error("No per-sequence results collected. Exiting.")
        sys.exit(1)
    combined_per_sequence_df: Optional[pd.DataFrame] = None
    try:
        combined_per_sequence_df = pd.concat(all_per_sequence_dfs, ignore_index=True)
        per_seq_filepath = os.path.join(args.output, "per_sequence_metrics_all_genes.csv")
        combined_per_sequence_df.to_csv(per_seq_filepath, index=False, float_format='%.5f')
        logger.info(f"Combined metrics table saved: {per_seq_filepath}")
    except Exception as concat_err: 
        logger.exception(f"Error concatenating/saving per-sequence results: {concat_err}"); sys.exit(1)

    # --- Calculate and Save Mean Features, Stats, Dinucl Abund ---
    mean_summary_df: Optional[pd.DataFrame] = None
    # --- Calculate and Save Mean Features per Gene ---
    if combined_per_sequence_df is not None:
        logger.info("Calculating mean features per gene...")
        mean_features_list: List[str] = [ # Defined list of features
            'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'RCDI', 'ENC', 'CAI',
            'Aromaticity', 'GRAVY']
        available_mean_features = [f for f in mean_features_list if f in combined_per_sequence_df.columns]
        missing_mean_features = [f for f in mean_features_list if f not in available_mean_features]
        if missing_mean_features:
            logger.warning(f"Cannot calculate mean for missing features: {', '.join(missing_mean_features)}")

        mean_summary_df: Optional[pd.DataFrame] = None
        if 'Gene' not in combined_per_sequence_df.columns:
            logger.error("'Gene' column missing in combined data, cannot calculate mean features per gene.")
        elif not available_mean_features:
            logger.warning("No available features found to calculate means.")
        else:
            try:
                temp_df_for_mean = combined_per_sequence_df.copy()
                for col in available_mean_features:
                    temp_df_for_mean[col] = pd.to_numeric(temp_df_for_mean[col], errors='coerce')

                mean_summary_df = temp_df_for_mean.groupby('Gene')[available_mean_features].mean(numeric_only=True).reset_index()
                # Rename Aromaticity for clarity if desired
                # if 'Aromaticity' in mean_summary_df.columns:
                #    mean_summary_df.rename(columns={'Aromaticity': 'Aromaticity_pct'}, inplace=True)

                # Save the mean summary table
                if mean_summary_df is not None and not mean_summary_df.empty:
                    mean_summary_filepath = os.path.join(args.output, "mean_features_per_gene.csv")
                    mean_summary_df.to_csv(mean_summary_filepath, index=False, float_format='%.4f')
                    logger.info(f"Mean features per gene saved to: {mean_summary_filepath}")

            except Exception as mean_err:
                logger.exception(f"Error calculating mean features per gene: {mean_err}")

    # --- Perform and Save Statistical Comparisons ---
    if combined_per_sequence_df is not None:
        logger.info("Performing statistical comparison between genes...")
        features_to_compare: List[str] = [ # Define features again or reuse list
            'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'RCDI', 'ENC', 'CAI', 'Aromaticity', 'GRAVY']
        try:
            comparison_results_df = analysis.compare_features_between_genes(
                combined_per_sequence_df, features=features_to_compare, method='kruskal')

            if comparison_results_df is not None and not comparison_results_df.empty:
                comparison_filepath = os.path.join(args.output, "gene_comparison_stats.csv")
                comparison_results_df.to_csv(comparison_filepath, index=False, float_format='%.4g')
                logger.info(f"Gene comparison statistics saved to: {comparison_filepath}")
            else:
                logger.info("No statistical comparison results generated (check data and groups).")
        except Exception as stat_err:
                logger.exception(f"Error during statistical comparison: {stat_err}")


    # --- Calculate Relative Dinucleotide Abundance ---
    logger.info("Calculating relative dinucleotide abundances...")
    all_rel_abund_data: List[Dict[str, Any]] = []
    genes_for_dinucl = sorted(list(set(all_nucl_freqs_by_gene.keys()) & set(all_dinucl_freqs_by_gene.keys())))
    if not genes_for_dinucl:
        logger.warning("Missing nucleotide or dinucleotide frequencies for all genes. Cannot calculate relative abundance.")
        rel_abund_df = pd.DataFrame()
    else:
        for gene_name in genes_for_dinucl:
                nucl_freqs = all_nucl_freqs_by_gene.get(gene_name)
                dinucl_freqs = all_dinucl_freqs_by_gene.get(gene_name)
                if nucl_freqs and dinucl_freqs: # Check both exist
                    try:
                        rel_abund = analysis.calculate_relative_dinucleotide_abundance(nucl_freqs, dinucl_freqs)
                        for dinucl, ratio in rel_abund.items():
                                all_rel_abund_data.append({'Gene': gene_name, 'Dinucleotide': dinucl, 'RelativeAbundance': ratio})
                    except Exception as e:
                            logger.warning(f"Could not calculate relative dinucleotide abundance for '{gene_name}': {e}")

        if not all_rel_abund_data:
                logger.warning("No relative dinucleotide abundance data generated.")
                rel_abund_df = pd.DataFrame()
        else:
                rel_abund_df = pd.DataFrame(all_rel_abund_data)


    # --- Preparation for combined plots ---
    # Initialize variables if some steps fail
    combined_ca_input_df: Optional[pd.DataFrame] = None
    ca_results_combined: Optional[analysis.PrinceCA] = None # type: ignore
    gene_groups_for_plotting: Optional[pd.Series] = None
    gene_color_map: Optional[Dict[str, Any]] = None
    
    # 1. Determine all unique groups/genes presents in final data.
    # Use combined_per_sequence_df which should have 'Gene' column
    all_plot_groups: List[str] = []
    if combined_per_sequence_df is not None and 'Gene' in combined_per_sequence_df.columns:
        # Sort for a coherent order, put 'complete' at the end if present
        unique_genes = sorted([g for g in combined_per_sequence_df['Gene'].unique() if g != 'complete'])
        if 'complete' in combined_per_sequence_df['Gene'].unique():
            unique_genes.append('complete')
        all_plot_groups = unique_genes
        logger.info(f"Found {len(all_plot_groups)} unique groups for consistent plotting: {', '.join(all_plot_groups)}")

        # 2. Generate color palette and color mapping 
        if all_plot_groups:
            try:
                # Use 'husl' to get a large number of distinct colors
                palette_list = sns.color_palette("husl", n_colors=len(all_plot_groups))
                gene_color_map = {gene: color for gene, color in zip(all_plot_groups, palette_list)}
                logger.debug(f"Generated color map for groups: {gene_color_map}")
            except Exception as palette_err:
                logger.warning(f"Could not generate custom color palette: {palette_err}. Falling back to defaults.")
                gene_color_map = None # Use by default if error
        else:
            logger.warning("No groups found to create color map.")

    else:
        logger.warning("Could not determine unique genes from combined data. Plots might lack consistent colors.")


    # --- Combined Correspondence Analysis ---
    combined_ca_input_df: Optional[pd.DataFrame] = None
    ca_results_combined: Optional[Any] = None # Type depends on prince.CA object
    gene_groups_for_plotting: Optional[pd.Series] = None
    ca_row_coords: Optional[pd.DataFrame] = None

    if not args.skip_ca and all_ca_input_dfs:
        logger.info("Performing combined Correspondence Analysis...")
        try:
            if all_ca_input_dfs:
                combined_ca_input_df = pd.concat(all_ca_input_dfs.values())
                # Clean combined data before CA
                combined_ca_input_df.fillna(0.0, inplace=True)
                combined_ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True)

                if not combined_ca_input_df.empty:
                    # Create grouping data (Series with index matching combined_ca_input_df)
                    group_data = combined_ca_input_df.index.str.split('__', n=1).str[0]
                    gene_groups_for_plotting = pd.Series(
                        data=group_data, index=combined_ca_input_df.index, name='Gene')

                    # Perform CA
                    ca_results_combined = analysis.perform_ca(combined_ca_input_df)
                    if ca_results_combined:
                        logger.info("Combined Correspondence Analysis complete.")
                        try:
                            ca_row_coords = ca_results_combined.row_coordinates(combined_ca_input_df)
                            logger.debug(f"Extracted CA row coordinates with shape: {ca_row_coords.shape}")
                        except Exception as coord_err:
                            logger.error(f"Could not extract row coordinates from CA results: {coord_err}")
                            ca_row_coords = None
                    else:
                        logger.warning("Combined CA calculation failed or produced no result.")
                        combined_ca_input_df = None # Reset df if CA fails
                else:
                    logger.warning("Combined CA input data is empty after concatenation.")
                    combined_ca_input_df = None
            else:
                logger.warning("No data collected for combined CA.")

        except Exception as ca_err:
            logger.exception(f"Error during combined Correspondence Analysis: {ca_err}")
            ca_results_combined = None
            combined_ca_input_df = None
    elif args.skip_ca:
            logger.info("Skipping combined Correspondence Analysis as requested.")
    else:
        logger.info("Skipping combined Correspondence Analysis as no input data was available.")

    # --- Save CA Details ---
    if ca_results_combined is not None and combined_ca_input_df is not None:
        logger.info("Saving CA details (coordinates, contributions, eigenvalues)...")
        try:
            output_dir = args.output # Use variable for clarity
            ca_results_combined.row_coordinates(combined_ca_input_df).to_csv(
                os.path.join(output_dir, "ca_row_coordinates.csv"), float_format='%.5f')
            ca_results_combined.column_coordinates(combined_ca_input_df).to_csv(
                os.path.join(output_dir, "ca_col_coordinates.csv"), float_format='%.5f')
            # Check for attribute existence before accessing
            if hasattr(ca_results_combined, 'column_contributions_'):
                ca_results_combined.column_contributions_.to_csv(
                    os.path.join(output_dir, "ca_col_contributions.csv"), float_format='%.5f')
            if hasattr(ca_results_combined, 'eigenvalues_summary'):
                ca_results_combined.eigenvalues_summary.to_csv(
                    os.path.join(output_dir, "ca_eigenvalues.csv"), float_format='%.5f')
            logger.info("CA details saved.")
        except AttributeError as ae:
            logger.error(f"Could not access all attributes from CA results object to save details: {ae}")
        except Exception as ca_save_err:
            logger.exception(f"Error saving CA details: {ca_save_err}")


    # --- Save Per-Sequence RSCU (Wide Format) ---
    if combined_ca_input_df is not None and not combined_ca_input_df.empty:
        logger.info("Saving per-sequence RSCU values (wide format)...")
        rscu_wide_filepath = os.path.join(args.output, "per_sequence_rscu_wide.csv")
        try:
            combined_ca_input_df.to_csv(rscu_wide_filepath, float_format='%.4f')
            logger.info(f"Per-sequence RSCU table saved to: {rscu_wide_filepath}")
        except Exception as rscu_save_err:
            logger.exception(f"Error saving per-sequence RSCU table: {rscu_save_err}")

    # --- Correlation CA Axes vs Features ---
    if ca_row_coords is not None and combined_per_sequence_df is not None and combined_ca_input_df is not None:
        logger.info("Preparing data for CA Axes vs Features correlation heatmap...")
        try:
            # 1. Select Dim1, Dim2 from CA results
            if ca_row_coords.shape[1] >= 2:
                    ca_dims_to_merge = ca_row_coords[[0, 1]].copy()
                    ca_dims_to_merge.columns = ['CA_Dim1', 'CA_Dim2']
            else:
                    logger.warning("CA result has fewer than 2 dimensions. Cannot create correlation heatmap.")
                    raise ValueError("Insufficient CA dimensions") # Raise error to skip this section

            # 2. Prepare merge key for metrics dataframe
            df_metrics = combined_per_sequence_df.copy()
            # Assuming 'Original_ID' and 'Gene' columns exist from previous steps
            if 'Original_ID' not in df_metrics.columns or 'Gene' not in df_metrics.columns:
                    raise KeyError("Missing 'Original_ID' or 'Gene' column in combined metrics for merging.")
            def create_merge_key(row): return f"{row['Gene']}__{row['Original_ID']}"
            df_metrics['merge_key'] = df_metrics.apply(create_merge_key, axis=1)
            df_metrics.set_index('merge_key', inplace=True, verify_integrity=True) # Ensure unique keys

            # 3. Merge CA dims, metrics, and RSCU values
            # Check index compatibility before merge
            if not ca_dims_to_merge.index.isin(df_metrics.index).all():
                logger.warning("Index mismatch between CA coordinates and metrics dataframe. Merge might lose data.")
            if not ca_dims_to_merge.index.isin(combined_ca_input_df.index).all():
                    logger.warning("Index mismatch between CA coordinates and RSCU dataframe. Merge might lose data.")

            merged_df_1 = pd.merge(df_metrics, ca_dims_to_merge, left_index=True, right_index=True, how='inner')
            # Merge RSCU data (combined_ca_input_df already has the right index type)
            merged_df = pd.merge(merged_df_1, combined_ca_input_df, left_index=True, right_index=True, how='inner')

            if merged_df.empty:
                logger.warning("Merged DataFrame for CA-Feature correlation is empty. Skipping heatmap.")
            else:
                logger.info(f"Merged data for correlation created with shape: {merged_df.shape}")

                # 4. Define features and calculate correlations
                metric_features = ['Length', 'TotalCodons', 'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'ENC', 'CAI', 'Fop', 'RCDI', 'ProteinLength', 'GRAVY', 'Aromaticity']
                metric_features = [f for f in metric_features if f in merged_df.columns] # Filter existing
                rscu_columns = sorted([col for col in combined_ca_input_df.columns if len(col) == 3 and col == col.upper()]) # Ensure sorted order
                features_to_correlate = metric_features + rscu_columns
                logger.info(f"Calculating Spearman correlations for CA Axes vs {len(features_to_correlate)} features...")

                corr_coeffs_dim1 = {}
                corr_pvals_dim1 = {}
                corr_coeffs_dim2 = {}
                corr_pvals_dim2 = {}

                if not SCIPY_AVAILABLE:
                    logger.warning("Scipy not installed. Cannot calculate p-values for correlations. Heatmap will show coefficients only.")
                    # Fallback to pandas correlation
                    corr_matrix_full = merged_df[['CA_Dim1', 'CA_Dim2'] + features_to_correlate].corr(method='spearman')
                    corr_matrix_subset = corr_matrix_full.loc[['CA_Dim1', 'CA_Dim2'], features_to_correlate]
                    # Create dummy NaN p-value matrix
                    pval_matrix_subset = pd.DataFrame(np.nan, index=corr_matrix_subset.index, columns=corr_matrix_subset.columns)
                else:
                    # Calculate pairwise with scipy to get p-values
                    ca_dim1_data = merged_df['CA_Dim1']
                    ca_dim2_data = merged_df['CA_Dim2']
                    for feature in features_to_correlate:
                        feature_data = merged_df[feature]
                        # Perform correlation only if there's variance and enough common non-NaN data points
                        common_mask = ca_dim1_data.notna() & ca_dim2_data.notna() & feature_data.notna()
                        n_common = common_mask.sum()
                        if n_common < 3 or feature_data[common_mask].nunique() <= 1 or ca_dim1_data[common_mask].nunique() <= 1 or ca_dim2_data[common_mask].nunique() <= 1:
                            corr_coeffs_dim1[feature], corr_pvals_dim1[feature] = np.nan, np.nan
                            corr_coeffs_dim2[feature], corr_pvals_dim2[feature] = np.nan, np.nan
                            continue
                        try:
                            corr1, pval1 = scipy_stats.spearmanr(ca_dim1_data[common_mask], feature_data[common_mask]) # nan_policy='omit' not needed due to mask
                            corr2, pval2 = scipy_stats.spearmanr(ca_dim2_data[common_mask], feature_data[common_mask])
                            corr_coeffs_dim1[feature], corr_pvals_dim1[feature] = corr1, pval1
                            corr_coeffs_dim2[feature], corr_pvals_dim2[feature] = corr2, pval2
                        except ValueError as spe_err: # Handle potential errors like all NaNs after filtering
                            logger.warning(f"Could not calculate Spearman correlation for feature '{feature}': {spe_err}")
                            corr_coeffs_dim1[feature], corr_pvals_dim1[feature] = np.nan, np.nan
                            corr_coeffs_dim2[feature], corr_pvals_dim2[feature] = np.nan, np.nan

                    # Create DataFrames from results
                    corr_matrix_subset = pd.DataFrame({'CA_Dim1': corr_coeffs_dim1, 'CA_Dim2': corr_coeffs_dim2}).T
                    pval_matrix_subset = pd.DataFrame({'CA_Dim1': corr_pvals_dim1, 'CA_Dim2': corr_pvals_dim2}).T

                # 5. Call plotting function
                if not args.skip_plots and not corr_matrix_subset.empty:
                    for fmt in args.plot_formats:
                        try:
                            plotting.plot_ca_axes_feature_correlation(
                                corr_df=corr_matrix_subset, pval_df=pval_matrix_subset,
                                output_dir=args.output, file_format=fmt,
                                significance_threshold=0.05, method_name="Spearman")
                        except Exception as plot_err:
                            logger.error(f"Failed CA-Feature correlation heatmap ({fmt}): {plot_err}")

        except (KeyError, ValueError) as merge_err:
            logger.error(f"Error merging data for CA-Feature correlation (check IDs/columns): {merge_err}")
        except Exception as e:
            logger.exception(f"Unexpected error preparing data or plotting CA-Feature correlation: {e}")
    else:
        logger.info("Skipping CA Axes vs Features correlation heatmap: Missing CA results or combined metrics.")


    # --- Generate COMBINED Plots ---
    if not args.skip_plots:
        logger.info("Generating combined plots...")
        n_ca_dims_variance: int = 10
        n_ca_contrib_top: int = 10
        output_dir: str = args.output # Use variable

        for fmt in args.plot_formats:
            logger.debug(f"Generating combined plots in format: {fmt}")
            try:
                # Plot functions require the combined data calculated above
                if combined_per_sequence_df is not None:
                    plotting.plot_gc_means_barplot(combined_per_sequence_df, output_dir, fmt, 'Gene')
                    plotting.plot_enc_vs_gc3(combined_per_sequence_df, output_dir, fmt, 'Gene', palette=gene_color_map)
                    plotting.plot_neutrality(combined_per_sequence_df, output_dir, fmt, 'Gene', palette=gene_color_map)

                    # Correlation Heatmap
                    features_for_corr: List[str] = [ # Define features for correlation
                        'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'ENC', 'CAI', 'RCDI',
                        'Aromaticity', 'GRAVY', 'Length', 'TotalCodons']
                    available_corr_features = [f for f in features_for_corr if f in combined_per_sequence_df.columns]
                    if len(available_corr_features) > 1:
                        plotting.plot_correlation_heatmap(
                            combined_per_sequence_df, available_corr_features, output_dir, fmt, 'spearman')
                    else:
                        logger.warning("Not enough features available in combined data for correlation heatmap.")

                # Relative Dinucleotide Abundance Plot
                if not rel_abund_df.empty:
                    plotting.plot_relative_dinucleotide_abundance(rel_abund_df, output_dir, fmt)

                # CA Plots (only if CA was successful)
                if ca_results_combined is not None and combined_ca_input_df is not None:
                    plotting.plot_ca(ca_results_combined, combined_ca_input_df, output_dir, fmt,
                                    args.ca_dims[0], args.ca_dims[1],
                                    groups=gene_groups_for_plotting,
                                    palette=gene_color_map,
                                    filename_suffix="_combined_by_gene")
                    
                    plotting.plot_ca_variance(ca_results_combined, n_ca_dims_variance, output_dir, fmt)
                    # Plot contributions for Dim 1 and Dim 2
                    if hasattr(ca_results_combined, 'column_contributions_') and ca_results_combined.column_contributions_.shape[1] > 0:
                            plotting.plot_ca_contribution(ca_results_combined, 0, n_ca_contrib_top, output_dir, fmt)
                    if hasattr(ca_results_combined, 'column_contributions_') and ca_results_combined.column_contributions_.shape[1] > 1:
                            plotting.plot_ca_contribution(ca_results_combined, 1, n_ca_contrib_top, output_dir, fmt)

                elif not args.skip_ca:
                    logger.warning("Skipping CA-related plots as CA calculation was skipped, failed, or produced no results.")

            except Exception as plot_err:
                # Log error for specific format but continue with others
                logger.exception(f"Error occurred during combined plot generation for format '{fmt}': {plot_err}")
    else:
        logger.info("Skipping combined plot generation as requested.")

    logger.info("PyCodon Analyzer run finished successfully.")

def handle_extract_command(args: argparse.Namespace) -> None:
    """Handles the 'extract' subcommand and its arguments."""
    logger.info(f"Running 'extract' command. Annotations: {args.annotations}, Alignment: {args.alignment}, Output: {args.output_dir}")
    try:
        # Call the main extraction function from the extraction module
        extraction.extract_gene_alignments_from_genome_msa(
            annotations_path=args.annotations,
            alignment_path=args.alignment,
            ref_id=args.ref_id,
            output_dir=args.output_dir
        )
        logger.info("'extract' command finished successfully.")
    except FileNotFoundError as fnf_err:
        logger.error(f"Extraction error: {fnf_err}")
        sys.exit(1)
    except ValueError as val_err: # For parsing or data integrity errors
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
    # Add version argument
    # Assuming __version__ is defined in your package's __init__.py
    try: 
        from . import __version__ as pkg_version
    except ImportError: 
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
                                type=str, 
                                help="Path to input directory with gene_*.fasta files.")
    analyze_parser.add_argument("-o", 
                                "--output", 
                                default="codon_analysis_results", 
                                type=str, help="Path to output directory for analysis results.")
    analyze_parser.add_argument("--genetic_code", 
                                type=int, default=1, 
                                help="NCBI genetic code ID.")
    analyze_parser.add_argument("--ref", 
                                dest="reference_usage_file", 
                                type=str, 
                                default=DEFAULT_HUMAN_REF_PATH if DEFAULT_HUMAN_REF_PATH else "human", 
                                help="Reference codon usage table ('human', 'none', or path).")
    analyze_parser.add_argument("-t", 
                                "--threads", 
                                type=int, 
                                default=1, 
                                help="Number of processes for parallel gene file analysis.")
    analyze_parser.add_argument("--max_ambiguity", 
                                type=float, 
                                default=15.0, 
                                help="Max allowed 'N' percentage per sequence.")
    analyze_parser.add_argument("--plot_formats", 
                                nargs='+', 
                                default=['png'], 
                                type=str, 
                                help="Output format(s) for plots.")
    analyze_parser.add_argument("--skip_plots", 
                                action='store_true', 
                                help="Disable all plot generation.")
    analyze_parser.add_argument("--ca_dims", 
                                nargs=2, 
                                type=int, 
                                default=[0, 1], 
                                metavar=('X', 'Y'), 
                                help="Components for combined CA plot.")
    analyze_parser.add_argument("--skip_ca", 
                                action='store_true', 
                                help="Skip combined Correspondence Analysis.")
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

    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Only configure logging if not in a test environment that handles it
    if RICH_AVAILABLE:
        handler_to_use: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=True, show_level=True, log_time_format="[%X]"
        )
        # Pour RichHandler, le formateur est souvent plus simple car Rich gère les détails
        formatter = logging.Formatter("%(message)s", datefmt="[%X]") # Le nom du logger sera ajouté par RichHandler
        handler_to_use.setFormatter(formatter)
    else:
        # Fallback pour les tests ou si Rich n'est pas là
        handler_to_use = logging.StreamHandler(sys.stderr) # Écrit sur stderr
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler_to_use.setFormatter(formatter)

    # Configurer le logger spécifique de l'application
    # NE PAS configurer le root logger directement avec basicConfig si caplog doit fonctionner de manière prévisible
    logger.setLevel(log_level)
    
    # S'assurer qu'il n'y a pas de handlers dupliqués si main() est appelé plusieurs fois (peu probable ici)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler_to_use)

    # Empêcher les messages de remonter au logger racine si on gère tout ici
    #logger.propagate = False # Essayez avec et sans cette ligne

    # Les tests avec caplog s'attachent généralement au root logger.
    # Si on ne configure que notre logger "pycodon_analyzer", il faut que caplog soit configuré pour l'écouter.
    # Par défaut, caplog écoute le root logger. Si notre logger "pycodon_analyzer" propage
    # ses messages au root (ce qui est le comportement par défaut si propagate=True),
    # ET si le root logger a un niveau qui permet aux messages de passer, caplog devrait les voir.

    # Pour s'assurer que caplog voit quelque chose, configurons aussi le root logger
    # de manière minimale SANS RichHandler pour que caplog puisse s'y fier.
    # Ou, mieux, dans les tests, on dira à caplog d'écouter "pycodon_analyzer".

    # Le logger spécifique de l'application utilisera cette configuration racine
    #logger = logging.getLogger("pycodon_analyzer")
    # logger.setLevel(log_level) # Not strictly necessary if root is set and propagate is True

    logger.info(f"PyCodon Analyzer - Command: {args.command}") # This should be from "pycodon_analyzer"
    if args.verbose:
        logger.debug(f"Full arguments: {args}")


    # --- Execute the function associated with the subcommand ---
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help() # Should not be reached if command is required

if __name__ == '__main__':
    main()