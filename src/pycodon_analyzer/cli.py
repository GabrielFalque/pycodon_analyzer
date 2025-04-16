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

import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Import necessary functions from local modules
# Assuming type checkers can find these or using forward references if needed
from . import io
from . import analysis
from . import plotting
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


# --- Helper Function to process a single file (with type hints and logging) ---
# Define a type alias for the complex return tuple for clarity
GeneResultType = Tuple[
    Optional[str],                  # gene_name
    Optional[str],                  # status
    Optional[pd.DataFrame],         # per_sequence_df
    Optional[pd.DataFrame],         # ca_input_df (still needed for combined CA)
    Optional[Dict[str, float]],     # nucl_freqs
    Optional[Dict[str, float]],     # dinucl_freqs
    Optional[Dict[str, Seq]]       # cleaned_seq_map
]

def process_gene_file(gene_filepath: str,
                      args: argparse.Namespace,
                      reference_weights: Optional[Dict[str, float]],
                      expected_gene_names: Set[str]) -> Optional[GeneResultType]:
    """
    Reads, cleans, and analyzes sequences from a single gene FASTA file.
    Calls analysis.run_full_analysis (which is now internally sequential).

    This function is designed to be called in parallel for multiple files.

    Args:
        gene_filepath (str): Path to the gene FASTA file.
        args (argparse.Namespace): Command line arguments.
        reference_weights (Optional[Dict[str, float]]): Pre-calculated reference codon weights.
        expected_gene_names (Set[str]): Set of expected gene names (from filenames).

    Returns:
        Optional[GeneResultType]: A tuple containing results for this gene,
                                  or None if the file should be skipped.
                                  The tuple contains status and result DataFrames/Dicts.
    """
   # --- Set Matplotlib backend to non-interactive *within the worker* ---
    # This is the crucial part for parallel plotting stability
    try:
        import matplotlib
        matplotlib.use('Agg')
        # Optional: Import pyplot here if needed by plotting functions (often not directly)
        # import matplotlib.pyplot as plt
        # logger.debug(f"[Worker {os.getpid()}] Matplotlib backend set to Agg.") # Debug log
    except ImportError:
        logger.error(f"[Worker {os.getpid()}] Matplotlib not found. Cannot generate plots in this worker.")
        # Decide behavior: proceed without plotting or return error? Let's proceed.
    except Exception as backend_err:
         logger.warning(f"[Worker {os.getpid()}] Could not set Matplotlib backend to Agg: {backend_err}")

    pid_prefix = f"[Process {os.getpid()}]" # Keep for clarity in logs
    gene_name = extract_gene_name_from_file(gene_filepath)

    if not gene_name or gene_name not in expected_gene_names: return None

    logger.info(f"{pid_prefix} Starting processing for gene: {gene_name} (File: {os.path.basename(gene_filepath)})")
    # Define variables needed for plotting outside the analysis try block
    agg_usage_df_gene: Optional[pd.DataFrame] = None
    ca_input_df_gene: Optional[pd.DataFrame] = None

    try:
        # 1. Read FASTA
        raw_sequences: List[SeqRecord] = io.read_fasta(gene_filepath)
        if not raw_sequences: return (gene_name, "empty file", None, None, None, None, None)

        # 2. Clean
        cleaned_sequences: List[SeqRecord] = clean_and_filter_sequences(raw_sequences, args.max_ambiguity)
        if not cleaned_sequences:
             logger.warning(f"{pid_prefix} No valid sequences after cleaning for gene {gene_name}. Skipping.")
             return (gene_name, "no valid seqs", None, None, None, None, None)
        cleaned_seq_map: Dict[str, Seq] = {seq_record.id: seq_record.seq for seq_record in cleaned_sequences}

        # 3. Analyze (does not fit CA model)
        analysis_results: tuple = analysis.run_full_analysis(
            cleaned_sequences,
            args.genetic_code,
            reference_weights=reference_weights
            # No fit_ca_model argument
        )
        # Unpack needed results
        agg_usage_df_gene = analysis_results[0] # Needed for plotting
        per_sequence_df_gene: Optional[pd.DataFrame] = analysis_results[1]
        nucl_freqs_gene: Optional[Dict[str, float]] = analysis_results[2]
        dinucl_freqs_gene: Optional[Dict[str, float]] = analysis_results[3]
        ca_input_df_gene = analysis_results[6] # Needed for plotting & combined CA

        # --- 4. Generate Per-Gene Plot (if not skipped) ---
        if not args.skip_plots:
            logger.debug(f"{pid_prefix} Preparing and generating RSCU boxplot for {gene_name}...")
            if agg_usage_df_gene is not None and not agg_usage_df_gene.empty and \
               ca_input_df_gene is not None and not ca_input_df_gene.empty:
                try:
                    # Prepare long format RSCU data from ca_input_df
                    temp_ca_input = ca_input_df_gene.copy()
                    # Remove gene prefix before melting (index should be Gene__SeqID)
                    temp_ca_input.index = temp_ca_input.index.str.split('__', n=1).str[1]
                    long_rscu_df = temp_ca_input.reset_index().rename(columns={'index': 'SequenceID'})
                    long_rscu_df = long_rscu_df.melt(id_vars=['SequenceID'], var_name='Codon', value_name='RSCU')
                    gc_dict = get_genetic_code(args.genetic_code)
                    long_rscu_df['AminoAcid'] = long_rscu_df['Codon'].map(gc_dict.get)
                    long_rscu_df['Gene'] = gene_name

                    # Call plotting function for each format
                    for fmt in args.plot_formats:
                        logger.debug(f"{pid_prefix} Saving RSCU boxplot for {gene_name} as {fmt}...")
                        try:
                            # Call the plotting function directly
                            plotting.plot_rscu_boxplot_per_gene(
                                long_rscu_df, agg_usage_df_gene, gene_name, args.output, fmt
                            )
                        except Exception as plot_fmt_err:
                            # Log error specific to plotting/saving this format
                            logger.error(f"{pid_prefix} Failed to generate/save RSCU boxplot for {gene_name} (format: {fmt}): {plot_fmt_err}")
                            # Optionally log traceback: logger.exception(...)

                except Exception as plot_prep_err:
                    # Log error during plot data preparation
                    logger.error(f"{pid_prefix} Failed to prepare data for RSCU boxplot for {gene_name}: {plot_prep_err}")
            else:
                logger.warning(f"{pid_prefix} Skipping RSCU boxplot generation for {gene_name} due to missing analysis data.")
        # --- End Plotting ---

        # 5. Prepare results for return
        # Modify per_sequence_df IDs if it exists
        if per_sequence_df_gene is not None and not per_sequence_df_gene.empty:
            if 'ID' in per_sequence_df_gene.columns:
                 per_sequence_df_gene['Original_ID'] = per_sequence_df_gene['ID']
                 per_sequence_df_gene['ID'] = per_sequence_df_gene['ID'].astype(str) + "_" + gene_name
            per_sequence_df_gene['Gene'] = gene_name

        # Ensure ca_input_df still has the prefix for combined analysis
        if ca_input_df_gene is not None and not ca_input_df_gene.empty:
             # Check if NOT ALL indices start with the prefix before adding it
             # (assuming it should have been added during run_full_analysis if needed)
             index_as_str = ca_input_df_gene.index.astype(str)
             if not index_as_str.str.startswith(f"{gene_name}__").all():
                  logger.warning(f"{pid_prefix} Adding missing gene prefix to ca_input_df index for {gene_name}") # Log if this happens
                  ca_input_df_gene.index = f"{gene_name}__" + index_as_str

        logger.info(f"{pid_prefix} Finished processing for gene: {gene_name}")
        # Return tuple matching GeneResultType (no agg_usage_df needed back)
        return (gene_name, "success", per_sequence_df_gene, ca_input_df_gene,
                nucl_freqs_gene, dinucl_freqs_gene, cleaned_seq_map)

    # --- Error Handling within worker ---
    except FileNotFoundError: # Should be caught by read_fasta now
        logger.error(f"{pid_prefix} File not found error for gene {gene_name}: {gene_filepath}")
        return (gene_name, "file not found error", None, None, None, None, None)
    except ValueError as ve: # Catch parsing errors or others
        logger.error(f"{pid_prefix} ValueError processing gene {gene_name}: {ve}")
        return (gene_name, "value error", None, None, None, None, None)
    except Exception as e:
        logger.exception(f"{pid_prefix} UNEXPECTED ERROR processing gene {gene_name} (file {os.path.basename(gene_filepath)}): {e}")
        return (gene_name, "exception", None, None, None, None, None)


# --- Main function ---
def main() -> None:
    """Main function to parse arguments and run the analysis workflow."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Analyze codon usage and sequence properties from gene alignment files within a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add all arguments with type hints where applicable
    parser.add_argument(
        "-d", "--directory", required=True, type=str,
        help="Path to the input directory containing gene alignment FASTA files (e.g., gene_ORF1ab.fasta)."
    )
    parser.add_argument(
        "-o", "--output", default="codon_analysis_results", type=str,
        help="Path to the output directory for results."
    )
    parser.add_argument(
        "--genetic_code", type=int, default=1,
        help="NCBI genetic code ID."
    )
    parser.add_argument(
        "--ref", "--reference_usage", dest="reference_usage_file", type=str,
        default=DEFAULT_HUMAN_REF_PATH if DEFAULT_HUMAN_REF_PATH else "human",
        help="Path to reference codon usage table (CSV/TSV) for CAI/Fop/RCDI. "
             "Use 'human' for bundled default, 'none' to disable."
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=1,
        help="Number of processor threads for parallel processing of GENE FILES. "
             "0 or negative uses all available cores."
    )
    parser.add_argument(
        "--max_ambiguity", type=float, default=15.0,
        help="Maximum allowed percentage of ambiguous bases ('N') per sequence after cleaning (0-100)."
    )
    parser.add_argument(
        "--plot_formats", nargs='+', default=['png'], type=str,
        help="Format(s) for saving plots (e.g., png, svg, pdf)."
    )
    parser.add_argument(
        "--skip_plots", action='store_true',
        help="Do not generate any plots."
    )
    parser.add_argument(
        "--ca_dims", nargs=2, type=int, default=[0, 1], metavar=('X', 'Y'),
        help="Components (0-indexed) to plot for Correspondence Analysis (e.g., 0 1 for Comp1 vs Comp2)."
    )
    parser.add_argument(
        "--skip_ca", action='store_true',
        help="Skip Correspondence Analysis calculation and plotting."
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        help="Add verbosity (more messages, logs DEBUG level)."
    )
    args: argparse.Namespace = parser.parse_args()

    # --- Basic Logging Setup ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        # stream=sys.stderr # Log to stderr by default
    )
    # Mute overly verbose libraries if necessary (optional)
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # logging.getLogger("Bio").setLevel(logging.WARNING)

    logger.info("Starting PyCodon Analyzer run.")
    logger.debug(f"Arguments received: {args}")

    # --- Argument Validation & Setup ---
    if not os.path.isdir(args.directory):
        logger.error(f"Input directory not found: {args.directory}")
        sys.exit(1)
    if not (0 <= args.max_ambiguity <= 100):
         logger.error("--max_ambiguity must be between 0 and 100.")
         sys.exit(1)
    if not args.ca_dims or len(args.ca_dims) != 2 or not all(isinstance(d, int) and d >= 0 for d in args.ca_dims):
        logger.error("--ca_dims requires two non-negative integers (e.g., 0 1).")
        sys.exit(1)

    # Setup output directory
    try:
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"Results will be saved to: {args.output}")
    except OSError as e:
        logger.error(f"Error creating output directory '{args.output}': {e}")
        sys.exit(1)

    # --- Load Reference Usage File (with logging) ---
    reference_data: Optional[pd.DataFrame] = None
    reference_weights: Optional[Dict[str, float]] = None # Keep Optional for clarity

    if args.reference_usage_file and args.reference_usage_file.lower() != 'none':
        ref_path_to_load: Optional[str] = None
        if args.reference_usage_file.lower() == 'human':
             if DEFAULT_HUMAN_REF_PATH and os.path.isfile(DEFAULT_HUMAN_REF_PATH):
                 ref_path_to_load = DEFAULT_HUMAN_REF_PATH
             else:
                 logger.warning("Default human reference file requested ('human') but not found or path invalid. Cannot load reference.")
        elif os.path.isfile(args.reference_usage_file):
            ref_path_to_load = args.reference_usage_file
        else:
            logger.warning(f"Specified reference file not found: {args.reference_usage_file}. Cannot load reference.")

        if ref_path_to_load:
             logger.info(f"Loading codon usage reference table: {ref_path_to_load}...")
             try:
                # Get genetic code dict needed by load_reference_usage
                genetic_code_dict: Dict[str, str] = get_genetic_code(args.genetic_code)
                reference_data = load_reference_usage(ref_path_to_load, genetic_code_dict, args.genetic_code)

                if reference_data is not None and not reference_data.empty:
                     # Extract weights if loading succeeded
                     if 'Weight' in reference_data.columns:
                        reference_weights = reference_data['Weight'].to_dict()
                        logger.info("Reference data loaded and weights extracted successfully.")
                     else:
                         logger.error("'Weight' column missing in loaded reference data. Cannot use for CAI/Fop/RCDI.")
                         reference_data = None # Treat as if loading failed for weights
                else:
                     # load_reference_usage logs warnings if processing fails
                     logger.warning("Failed to process reference data after reading. CAI/Fop/RCDI will use NaN.")
                     reference_data = None # Ensure it's None if processing failed
             except NotImplementedError as nie:
                  logger.error(f"Error loading reference: {nie}")
                  sys.exit(1)
             except ValueError as ve: # Catch value errors from load_reference_usage
                 logger.error(f"Error processing reference file {ref_path_to_load}: {ve}")
             except Exception as load_err: # Catch other unexpected errors
                  logger.exception(f"Unexpected error loading reference file {ref_path_to_load}: {load_err}")
                  # Ensure weights are None if loading fails catastrophically
                  reference_weights = None
                  reference_data = None
    else:
         logger.info("No reference file specified (--ref set to 'none' or invalid). Skipping reference-based calculations (CAI/Fop/RCDI).")
    # --- End reference loading ---

    # --- Determine number of processes (with logging) ---
    num_file_processes: int = args.threads
    if num_file_processes <= 0:
        if MP_AVAILABLE:
            try:
                num_file_processes = os.cpu_count() or 1
                logger.info(f"Using {num_file_processes} processes for parallel gene file analysis (all available cores).")
            except NotImplementedError:
                num_file_processes = 1
                logger.warning("Could not determine number of cores, using 1 process.")
        else:
            num_file_processes = 1
            logger.warning("Multiprocessing not available, using 1 process.")
    elif num_file_processes == 1:
         logger.info("Using 1 process (sequential gene file analysis).")
    else:
         if not MP_AVAILABLE:
             logger.warning(f"Requested {num_file_processes} processes, but multiprocessing is not available. Using 1 process.")
             num_file_processes = 1
         else:
            logger.info(f"Using {num_file_processes} processes for parallel gene file analysis.")

    # --- Start Workflow ---
    try:
        # --- Find gene alignment files ---
        logger.info(f"Searching for gene files in directory: {args.directory}")
        search_pattern = os.path.join(args.directory, "gene_*.*")
        gene_files: List[str] = glob.glob(search_pattern)
        valid_extensions: Set[str] = {".fasta", ".fa", ".fna", ".fas", ".faa"}
        gene_files = sorted([f for f in gene_files if os.path.splitext(f)[1].lower() in valid_extensions])

        if not gene_files:
            logger.error(f"No gene alignment files (matching 'gene_*.(fasta|fa|...)') found in directory: {args.directory}")
            sys.exit(1)

        # Adjust number of processes if fewer files than requested threads
        original_requested_threads = args.threads
        num_file_processes = min(num_file_processes, len(gene_files))
        if num_file_processes != args.threads and original_requested_threads > 1 :
             logger.debug(f"Adjusted to {num_file_processes} processes as only {len(gene_files)} files were found.")

        logger.info(f"Found {len(gene_files)} potential gene alignment files to process.")

        # --- Determine expected gene names ---
        expected_gene_names: Set[str] = set()
        for fpath in gene_files:
             gname = extract_gene_name_from_file(fpath)
             if gname:
                  expected_gene_names.add(gname)
        if not expected_gene_names:
             logger.error("Could not extract any valid gene names from input filenames. Check naming pattern.")
             sys.exit(1)
        logger.info(f"Expecting data for {len(expected_gene_names)} genes: {', '.join(sorted(list(expected_gene_names)))}")

        # --- Parallel Processing of Gene Files ---
        logger.info(f"Processing {len(gene_files)} gene files using {num_file_processes} processes...")

        # Create partial function with fixed arguments
        processing_task = partial(process_gene_file, # Pass args which includes skip_plots now
                                  args=args,
                                  reference_weights=reference_weights,
                                  expected_gene_names=expected_gene_names)
        gene_results_raw: List[Optional[GeneResultType]] = []
        # Use multiprocessing if > 1 process and module available
        if num_file_processes > 1 and MP_AVAILABLE:
            try:
                with mp.Pool(processes=num_file_processes) as pool:
                    gene_results_raw = pool.map(processing_task, gene_files)
            except Exception as pool_err:
                 logger.exception(f"Error during parallel gene processing: {pool_err}. Falling back to sequential.")
                 # Sequential fallback
                 gene_results_raw = [processing_task(f) for f in gene_files]
        else:
             # Sequential processing
             if num_file_processes > 1 and not MP_AVAILABLE:
                  logger.info("Executing sequentially as multiprocessing is unavailable.")
             else:
                  logger.info("Executing sequentially.")
             gene_results_raw = [processing_task(f) for f in gene_files]

        # --- Collect and Process Results ---
        logger.info("Collecting and aggregating results...")
        all_per_sequence_dfs: List[pd.DataFrame] = []
        all_ca_input_dfs: Dict[str, pd.DataFrame] = {}
        processed_genes: Set[str] = set()
        failed_genes: List[str] = []
        sequences_by_original_id: Dict[str, Dict[str, Seq]] = {}
        all_nucl_freqs_by_gene: Dict[str, Dict[str, float]] = {}
        all_dinucl_freqs_by_gene: Dict[str, Dict[str, float]] = {}
        gene_plot_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {} # {gene: (long_rscu, agg_usage)}

        for result in gene_results_raw:
            if result is None: continue # Skip ignored files

            # Unpack tuple safely
            (gene_name_res, status, per_seq_df, ca_input_df,
             nucl_freqs, dinucl_freqs, cleaned_map) = result
            
            if status == "success":
                processed_genes.add(str(gene_name_res)) # Ensure string
                # Append/add results if they are not None
                if per_seq_df is not None: 
                    all_per_sequence_dfs.append(per_seq_df)
                if ca_input_df is not None: 
                    all_ca_input_dfs[str(gene_name_res)] = ca_input_df
                if nucl_freqs: 
                    all_nucl_freqs_by_gene[str(gene_name_res)] = nucl_freqs
                if dinucl_freqs: 
                    all_dinucl_freqs_by_gene[str(gene_name_res)] = dinucl_freqs

                # Aggregate cleaned sequences
                if cleaned_map:
                    for seq_id, seq_obj in cleaned_map.items():
                        if seq_id not in sequences_by_original_id: sequences_by_original_id[seq_id] = {}
                        sequences_by_original_id[seq_id][str(gene_name_res)] = seq_obj
                        
            else:
                # Add failed gene/reason
                failed_genes.append(f"{gene_name_res} ({status})")

        # --- Post-processing Check ---
        successfully_processed_genes: Set[str] = processed_genes
        if not successfully_processed_genes:
             logger.error("No genes were successfully processed. Exiting.")
             sys.exit(1)
        elif len(successfully_processed_genes) < len(expected_gene_names):
             logger.warning(f"Processed {len(successfully_processed_genes)} genes out of {len(expected_gene_names)} expected.")
             if failed_genes:
                  logger.warning(f"  Failed genes/reasons: {'; '.join(failed_genes)}")


        # --- Generate per-gene plots ---
        if not args.skip_plots:
            logger.info("Generating per-gene RSCU boxplots...")
            if gene_plot_data:
                for gene_name, plot_data_tuple in gene_plot_data.items():
                    long_df, agg_df = plot_data_tuple
                    if long_df is not None and not long_df.empty and agg_df is not None and not agg_df.empty:
                         for fmt in args.plot_formats:
                             try:
                                 plotting.plot_rscu_boxplot_per_gene(
                                     long_df, agg_df, gene_name, args.output, fmt)
                             except Exception as gene_plot_err:
                                 logger.error(f"Error generating RSCU boxplot for {gene_name} (format {fmt}): {gene_plot_err}")
                                 # Consider logging traceback here with logger.exception if needed for debug
                    else:
                         logger.debug(f"Skipping RSCU boxplot for {gene_name} due to missing plot data.")
            else:
                 logger.info("No data available to generate per-gene plots.")


        # --- Analyze "Complete" Sequences ---
        logger.info("Analyzing concatenated 'complete' sequences...")
        complete_seq_records_to_analyze: List[SeqRecord] = []
        if sequences_by_original_id:
             max_ambiguity_pct_complete: float = args.max_ambiguity
             for original_id, gene_seq_map in sequences_by_original_id.items():
                 if set(gene_seq_map.keys()) == successfully_processed_genes:
                     try:
                        concatenated_seq_str = "".join(str(gene_seq_map[g_name]) for g_name in sorted(gene_seq_map.keys()))
                        if concatenated_seq_str and len(concatenated_seq_str) % 3 == 0:
                            n_count = concatenated_seq_str.count('N')
                            seq_len = len(concatenated_seq_str)
                            ambiguity_pct = (n_count / seq_len) * 100 if seq_len > 0 else 0
                            if ambiguity_pct <= max_ambiguity_pct_complete:
                                complete_record = SeqRecord(Seq(concatenated_seq_str), id=original_id, description=f"Concatenated {len(gene_seq_map)} genes")
                                complete_seq_records_to_analyze.append(complete_record)
                            else: logger.debug(f"Complete sequence for {original_id} skipped (ambiguity {ambiguity_pct:.1f}% > {max_ambiguity_pct_complete}%)")
                        else: logger.debug(f"Complete sequence for {original_id} skipped (invalid length or empty)")
                     except Exception as concat_err:
                         logger.warning(f"Error concatenating sequence for ID {original_id}: {concat_err}")
                 # else: logger.debug(f"Complete sequence for {original_id} skipped (missing some processed genes)")


        if complete_seq_records_to_analyze:
            logger.info(f"Running analysis on {len(complete_seq_records_to_analyze)} valid 'complete' sequence records...")
            try:
                # Run SEQUENTIAL analysis on complete sequences
                agg_usage_df_complete, per_sequence_df_complete, \
                nucl_freqs_complete, dinucl_freqs_complete, \
                _, _, ca_input_df_complete = analysis.run_full_analysis(
                    complete_seq_records_to_analyze,
                    args.genetic_code,
                    reference_weights=reference_weights
                )

                # Store "complete" results
                if nucl_freqs_complete: all_nucl_freqs_by_gene['complete'] = nucl_freqs_complete
                if dinucl_freqs_complete: all_dinucl_freqs_by_gene['complete'] = dinucl_freqs_complete

                # Prepare data for 'complete' plot
                if ca_input_df_complete is not None and agg_usage_df_complete is not None:
                     try:
                         long_rscu_df_comp = ca_input_df_complete.reset_index().rename(columns={'index': 'SequenceID'})
                         long_rscu_df_comp = long_rscu_df_comp.melt(id_vars=['SequenceID'], var_name='Codon', value_name='RSCU')
                         gc_dict = get_genetic_code(args.genetic_code)
                         long_rscu_df_comp['AminoAcid'] = long_rscu_df_comp['Codon'].map(gc_dict.get)
                         long_rscu_df_comp['Gene'] = 'complete'
                         gene_plot_data['complete'] = (long_rscu_df_comp, agg_usage_df_complete)
                     except Exception as plot_prep_err:
                          logger.warning(f"Could not prepare plot data for 'complete': {plot_prep_err}")

                # Add results to main aggregates
                if per_sequence_df_complete is not None and not per_sequence_df_complete.empty:
                    if 'ID' in per_sequence_df_complete.columns:
                         per_sequence_df_complete['Original_ID'] = per_sequence_df_complete['ID']
                         per_sequence_df_complete['ID'] = per_sequence_df_complete['ID'].astype(str) + "_complete"
                    per_sequence_df_complete['Gene'] = 'complete'
                    all_per_sequence_dfs.append(per_sequence_df_complete)

                if ca_input_df_complete is not None and not ca_input_df_complete.empty:
                    ca_input_df_complete.index = "complete__" + ca_input_df_complete.index.astype(str)
                    all_ca_input_dfs['complete'] = ca_input_df_complete

            except Exception as e:
                logger.exception(f"Error during 'complete' sequence analysis: {e}")
        else:
            logger.info("No valid 'complete' sequences found to analyze.")


        # --- Generate 'complete' plot AFTER its analysis ---
        if not args.skip_plots and 'complete' in gene_plot_data:
             logger.info("Generating RSCU boxplot for 'complete' data...")
             plot_data_tuple_comp = gene_plot_data.get('complete')
             if plot_data_tuple_comp:
                 long_df_comp, agg_df_comp = plot_data_tuple_comp
                 if long_df_comp is not None and agg_df_comp is not None:
                     for fmt in args.plot_formats:
                         try:
                             plotting.plot_rscu_boxplot_per_gene(long_df_comp, agg_df_comp, 'complete', args.output, fmt)
                         except Exception as comp_plot_err:
                             logger.error(f"Error generating RSCU boxplot for 'complete' (format {fmt}): {comp_plot_err}")
                 else:
                     logger.warning("Invalid plot data for 'complete', boxplot skipped.")
             else:
                  logger.warning("No plot data found for 'complete'.")


        # --- Final Combination and Saving of Results ---
        logger.info("Combining final results...")
        if not all_per_sequence_dfs:
             logger.error("No per-sequence results were collected. Cannot create combined tables or plots. Exiting.")
             sys.exit(1) # Exit if no data to combine

        combined_per_sequence_df: Optional[pd.DataFrame] = None
        try:
            combined_per_sequence_df = pd.concat(all_per_sequence_dfs, ignore_index=True)
            per_seq_filepath = os.path.join(args.output, "per_sequence_metrics_all_genes.csv")
            combined_per_sequence_df.to_csv(per_seq_filepath, index=False, float_format='%.5f')
            logger.info(f"Combined per-sequence metrics table saved to: {per_seq_filepath}")
        except pd.errors.InvalidIndexError as concat_err: # More specific error
             logger.exception(f"Error concatenating per-sequence results (potential index issue): {concat_err}")
             sys.exit(1)
        except Exception as save_err: # General saving error
            logger.exception(f"Error saving combined metrics table to '{per_seq_filepath}': {save_err}")
            # Decide if we can continue without this table

        # --- Calculate and Save Mean Features per Gene ---
        if combined_per_sequence_df is not None:
            logger.info("Calculating mean features per gene...")
            # ... (existing logic, replace print with logger.warning/info) ...
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
        ca_results_combined: Optional[analysis.PrinceCA] = None
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

    # --- Global Exception Handling ---
    except FileNotFoundError as e:
        logger.error(f"File not found error during execution: {e}")
        sys.exit(1)
    except ValueError as e: # Catch ValueErrors raised (e.g., from analysis functions)
        logger.exception(f"Data error during execution: {e}")
        sys.exit(1)
    except ImportError as e: # Catch missing dependencies
         logger.error(f"Import Error: {e}. Please ensure all dependencies are installed.")
         sys.exit(1)
    except NotImplementedError as e: # Catch features not implemented
         logger.error(f"Feature Error: {e}")
         sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (KeyboardInterrupt). Exiting.")
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors
        logger.exception(f"An unexpected error occurred in the main workflow: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Entry point if script is run directly
    main()