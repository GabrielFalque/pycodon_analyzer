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
    num_processes: int
) -> List[Optional[AnalyzeGeneResultType]]:
    """Runs process_analyze_gene_file in parallel or sequentially."""
    processing_task = partial(process_analyze_gene_file, args=args, reference_weights=reference_weights, expected_gene_names=expected_gene_names)
    results_raw: List[Optional[AnalyzeGeneResultType]] = []

    progress_columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TextColumn("({task.completed}/{task.total} genes)"), TimeElapsedColumn(), TextColumn("<"), TimeRemainingColumn()]
    disable_rich = not sys.stderr.isatty() or not RICH_AVAILABLE

    with Progress(*progress_columns, transient=False, disable=disable_rich) as progress:
        analysis_task_id = progress.add_task("Analyzing Gene Files", total=len(gene_files))
        if num_processes > 1 and MP_AVAILABLE:
            try:
                with mp.Pool(processes=num_processes) as pool:
                    for result in pool.imap(processing_task, gene_files):
                        results_raw.append(result)
                        progress.update(analysis_task_id, advance=1)
            except Exception as pool_err:
                logger.exception(f"Parallel analysis error: {pool_err}. No results collected from parallel run.")
                results_raw = [None] * len(gene_files) # Ensure list of correct size with failures
        else:
            for gene_file in gene_files:
                results_raw.append(processing_task(gene_file))
                progress.update(analysis_task_id, advance=1)
    return results_raw

def _collect_and_aggregate_results(
    analyze_results_raw: List[Optional[AnalyzeGeneResultType]],
    expected_gene_names: Set[str]
) -> Tuple[List[pd.DataFrame], Dict[str, pd.DataFrame], Set[str], List[str], Dict[str, Dict[str, Seq]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Collects results from individual gene analyses."""
    all_per_sequence_dfs: List[pd.DataFrame] = []
    all_ca_input_dfs: Dict[str, pd.DataFrame] = {}
    processed_genes: Set[str] = set()
    failed_genes_info: List[str] = []
    sequences_by_original_id: Dict[str, Dict[str, Seq]] = {}
    all_nucl_freqs_by_gene: Dict[str, Dict[str, float]] = {}
    all_dinucl_freqs_by_gene: Dict[str, Dict[str, float]] = {}

    for result in analyze_results_raw:
        if result is None: continue
        gene_name_res, status, per_seq_df, ca_input_df, nucl_freqs, dinucl_freqs, cleaned_map = result
        if status == "success" and gene_name_res:
            processed_genes.add(gene_name_res)
            if per_seq_df is not None: all_per_sequence_dfs.append(per_seq_df)
            if ca_input_df is not None: all_ca_input_dfs[gene_name_res] = ca_input_df
            if nucl_freqs: all_nucl_freqs_by_gene[gene_name_res] = nucl_freqs
            if dinucl_freqs: all_dinucl_freqs_by_gene[gene_name_res] = dinucl_freqs
            if cleaned_map:
                for seq_id, seq_obj in cleaned_map.items():
                    sequences_by_original_id.setdefault(seq_id, {})[gene_name_res] = seq_obj
        elif gene_name_res:
            failed_genes_info.append(f"{gene_name_res} ({status})")

    if not processed_genes:
        logger.error("No genes were successfully processed. Exiting.")
        sys.exit(1)
    if len(processed_genes) < len(expected_gene_names):
        logger.warning(f"Processed {len(processed_genes)} genes out of {len(expected_gene_names)} expected.")
        if failed_genes_info: logger.warning(f"  Failed genes/reasons: {'; '.join(failed_genes_info)}")
    
    return (all_per_sequence_dfs, all_ca_input_dfs, processed_genes, failed_genes_info,
            sequences_by_original_id, all_nucl_freqs_by_gene, all_dinucl_freqs_by_gene)

def _analyze_complete_sequences_cli(
    sequences_by_original_id: Dict[str, Dict[str, Seq]],
    successfully_processed_genes: Set[str],
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """Analyzes concatenated 'complete' sequences."""
    logger.info("Analyzing concatenated 'complete' sequences...")
    complete_seq_records: List[SeqRecord] = []
    max_ambiguity_pct_complete = args.max_ambiguity

    for original_id, gene_seq_map in sequences_by_original_id.items():
        if set(gene_seq_map.keys()) == successfully_processed_genes:
            try:
                # Ensure gene names are sorted for consistent concatenation order
                concat_str = "".join(str(gene_seq_map[g_name]) for g_name in sorted(gene_seq_map.keys()))
                if concat_str and len(concat_str) % 3 == 0:
                    n_count, seq_len = concat_str.count('N'), len(concat_str)
                    ambiguity = (n_count / seq_len) * 100 if seq_len > 0 else 0
                    if ambiguity <= max_ambiguity_pct_complete:
                        complete_seq_records.append(SeqRecord(Seq(concat_str), id=original_id, description=f"Concatenated {len(gene_seq_map)} genes"))
                    else:
                        logger.debug(f"Complete seq for {original_id} skipped (ambiguity {ambiguity:.1f}% > {max_ambiguity_pct_complete}%)")
            except Exception as e:
                logger.warning(f"Error concatenating sequence for ID {original_id}: {e}")

    if not complete_seq_records:
        logger.info("No valid 'complete' sequences to analyze.")
        return None, None, None, None, None

    logger.info(f"Running analysis on {len(complete_seq_records)} 'complete' sequence records...")
    try:
        res_comp = analysis.run_full_analysis(complete_seq_records, args.genetic_code, reference_weights)
        # agg_usage_df_complete, per_sequence_df_complete, nucl_freqs_complete, dinucl_freqs_complete, _, _, ca_input_df_complete
        return res_comp[0], res_comp[1], res_comp[2], res_comp[3], res_comp[6]
    except Exception as e:
        logger.exception(f"Error during 'complete' sequence analysis: {e}")
        return None, None, None, None, None

def _update_aggregate_data_with_complete_results(
    complete_results: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[pd.DataFrame]],
    all_per_sequence_dfs: List[pd.DataFrame],
    all_ca_input_dfs: Dict[str, pd.DataFrame],
    all_nucl_freqs_by_gene: Dict[str, Dict[str, float]],
    all_dinucl_freqs_by_gene: Dict[str, Dict[str, float]],
    args: argparse.Namespace # For plot formats and output dir
) -> None:
    """Updates the main data collections with results from 'complete' sequence analysis and plots for 'complete'."""
    agg_usage_df_complete, per_seq_df_complete, nucl_freqs_complete, dinucl_freqs_complete, ca_input_df_complete_plot = complete_results

    if nucl_freqs_complete: all_nucl_freqs_by_gene['complete'] = nucl_freqs_complete
    if dinucl_freqs_complete: all_dinucl_freqs_by_gene['complete'] = dinucl_freqs_complete
    
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

        # Plot RSCU for 'complete'
        if not args.skip_plots and agg_usage_df_complete is not None:
            logger.info("Generating RSCU boxplot for 'complete' data...")
            try:
                long_rscu_df_comp = ca_input_df_complete_plot.reset_index().rename(columns={'index': 'SequenceID'})
                long_rscu_df_comp = long_rscu_df_comp.melt(id_vars=['SequenceID'], var_name='Codon', value_name='RSCU')
                current_gc_dict = utils.get_genetic_code(args.genetic_code)
                long_rscu_df_comp['AminoAcid'] = long_rscu_df_comp['Codon'].map(current_gc_dict.get)
                for fmt in args.plot_formats:
                    plotting.plot_rscu_boxplot_per_gene(long_rscu_df_comp, agg_usage_df_complete, 'complete', args.output, fmt)
            except Exception as e:
                logger.error(f"Failed to generate 'complete' RSCU boxplot: {e}")
        elif not args.skip_plots:
             logger.warning("Cannot generate 'complete' RSCU boxplot due to missing aggregate usage data for complete set.")

def _finalize_and_save_per_sequence_metrics(all_per_sequence_dfs: List[pd.DataFrame], output_dir_path: Path) -> Optional[pd.DataFrame]:
    """Combines and saves the per-sequence metrics from all genes and the 'complete' set."""
    if not all_per_sequence_dfs:
        logger.error("No per-sequence results collected. Cannot save combined metrics.")
        return None
    try:
        combined_df = pd.concat(all_per_sequence_dfs, ignore_index=True)
        filepath = output_dir_path / "per_sequence_metrics_all_genes.csv"
        combined_df.to_csv(filepath, index=False, float_format='%.5f')
        logger.info(f"Combined per-sequence metrics saved: {filepath}")
        return combined_df
    except Exception as e:
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
    output_dir_path: Path,
    args: argparse.Namespace # For skip_ca and ca_dims
) -> Tuple[Optional[pd.DataFrame], Optional[PrinceCA], Optional[pd.Series], Optional[pd.DataFrame]]:
    """Performs combined CA and saves detailed CA results."""
    if args.skip_ca:
        logger.info("Skipping combined Correspondence Analysis as requested.")
        return None, None, None, None
    if not all_ca_input_dfs:
        logger.info("Skipping combined CA: No CA input data available from gene analyses.")
        return None, None, None, None

    logger.info("Performing combined Correspondence Analysis...")
    combined_ca_input_df: Optional[pd.DataFrame] = None
    ca_results_combined: Optional[PrinceCA] = None
    gene_groups_for_plotting: Optional[pd.Series] = None
    ca_row_coords_df: Optional[pd.DataFrame] = None

    try:
        combined_ca_input_df = pd.concat(all_ca_input_dfs.values())
        combined_ca_input_df.fillna(0.0, inplace=True) # Should already be 0.0 from run_full_analysis
        combined_ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True)

        if not combined_ca_input_df.empty:
            group_data = combined_ca_input_df.index.str.split('__', n=1, expand=True)
            gene_groups_for_plotting = pd.Series(data=group_data.get_level_values(0), index=combined_ca_input_df.index, name='Gene')
            
            ca_results_combined = analysis.perform_ca(combined_ca_input_df)
            if ca_results_combined:
                logger.info("Combined CA complete. Saving details...")
                ca_row_coords_df = ca_results_combined.row_coordinates(combined_ca_input_df)
                ca_row_coords_df.to_csv(output_dir_path / "ca_row_coordinates.csv", float_format='%.5f')
                ca_results_combined.column_coordinates(combined_ca_input_df).to_csv(output_dir_path / "ca_col_coordinates.csv", float_format='%.5f')
                if hasattr(ca_results_combined, 'column_contributions_'):
                    ca_results_combined.column_contributions_.to_csv(output_dir_path / "ca_col_contributions.csv", float_format='%.5f')
                if hasattr(ca_results_combined, 'eigenvalues_summary'):
                    ca_results_combined.eigenvalues_summary.to_csv(output_dir_path / "ca_eigenvalues.csv", float_format='%.5f')
                
                # Save per-sequence RSCU wide format
                rscu_wide_path = output_dir_path / "per_sequence_rscu_wide.csv"
                combined_ca_input_df.to_csv(rscu_wide_path, float_format='%.4f')
                logger.info(f"Per-sequence RSCU (wide) saved: {rscu_wide_path}")
            else:
                logger.warning("Combined CA fitting failed or produced no result.")
                combined_ca_input_df = None # Reset if CA failed
        else:
            logger.warning("Combined CA input data is empty. Skipping CA.")
            combined_ca_input_df = None
    except Exception as e:
        logger.exception(f"Error during combined CA: {e}")
        combined_ca_input_df, ca_results_combined, gene_groups_for_plotting, ca_row_coords_df = None, None, None, None
    
    return combined_ca_input_df, ca_results_combined, gene_groups_for_plotting, ca_row_coords_df

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

def _generate_all_combined_plots(
    args: argparse.Namespace,
    combined_per_sequence_df: Optional[pd.DataFrame],
    gene_color_map: Optional[Dict[str, Any]],
    rel_abund_df: pd.DataFrame,
    ca_results_combined: Optional[PrinceCA],
    combined_ca_input_df: Optional[pd.DataFrame], # This is the RSCU wide format data
    gene_groups_for_plotting: Optional[pd.Series],
    ca_row_coords: Optional[pd.DataFrame], # Output of ca_results.row_coordinates()
    reference_data_for_plot: Optional[pd.DataFrame] # For RSCU comparison plot
) -> None:
    """Generates all combined plots."""
    if args.skip_plots:
        logger.info("Skipping combined plot generation as requested.")
        return

    logger.info("Generating combined plots...")
    output_dir_path = Path(args.output)
    plot_formats = args.plot_formats
    n_ca_dims_variance, n_ca_contrib_top = 10, 10

    for fmt in plot_formats:
        logger.debug(f"Generating combined plots in format: {fmt}")
        try:
            if combined_per_sequence_df is not None:
                plotting.plot_gc_means_barplot(combined_per_sequence_df, str(output_dir_path), fmt, 'Gene')
                plotting.plot_enc_vs_gc3(combined_per_sequence_df, str(output_dir_path), fmt, 'Gene', palette=gene_color_map)
                plotting.plot_neutrality(combined_per_sequence_df, str(output_dir_path), fmt, 'Gene', palette=gene_color_map)
                
                features_for_corr = ['GC', 'GC1', 'GC2', 'GC3', 'GC12', 'ENC', 'CAI', 'RCDI', 'Aromaticity', 'GRAVY', 'Length', 'TotalCodons']
                available_corr_feat = [f for f in features_for_corr if f in combined_per_sequence_df.columns]
                if len(available_corr_feat) > 1:
                    plotting.plot_correlation_heatmap(combined_per_sequence_df, available_corr_feat, str(output_dir_path), fmt)
            
            if not rel_abund_df.empty:
                plotting.plot_relative_dinucleotide_abundance(rel_abund_df, str(output_dir_path), fmt, palette=gene_color_map)

            if ca_results_combined and combined_ca_input_df is not None:
                plotting.plot_ca(ca_results_combined, combined_ca_input_df, str(output_dir_path), fmt,
                                args.ca_dims[0], args.ca_dims[1], groups=gene_groups_for_plotting,
                                palette=gene_color_map, filename_suffix="_combined_by_gene")
                plotting.plot_ca_variance(ca_results_combined, n_ca_dims_variance, str(output_dir_path), fmt)
                if hasattr(ca_results_combined, 'column_contributions_'):
                    if ca_results_combined.column_contributions_.shape[1] > 0:
                         plotting.plot_ca_contribution(ca_results_combined, 0, n_ca_contrib_top, str(output_dir_path), fmt)
                    if ca_results_combined.column_contributions_.shape[1] > 1:
                         plotting.plot_ca_contribution(ca_results_combined, 1, n_ca_contrib_top, str(output_dir_path), fmt)
            
            # CA Axes vs Features Correlation Plot (using the new plotting function call structure)
            if ca_row_coords is not None and combined_per_sequence_df is not None and combined_ca_input_df is not None:
                dim_x_idx, dim_y_idx = args.ca_dims[0], args.ca_dims[1]
                max_available_dim = ca_row_coords.shape[1] - 1
                ca_dims_prepared_df_for_plot: Optional[pd.DataFrame] = None

                if not (dim_x_idx > max_available_dim or dim_y_idx > max_available_dim or dim_x_idx == dim_y_idx):
                    ca_dims_prepared_df_for_plot = ca_row_coords[[dim_x_idx, dim_y_idx]].copy()
                    ca_dims_prepared_df_for_plot.columns = [f'CA_Dim{dim_x_idx+1}', f'CA_Dim{dim_y_idx+1}']
                    if not ca_dims_prepared_df_for_plot.index.is_unique:
                        ca_dims_prepared_df_for_plot = ca_dims_prepared_df_for_plot[~ca_dims_prepared_df_for_plot.index.duplicated(keep='first')]

                metrics_prepared_df_for_plot: Optional[pd.DataFrame] = None
                if 'ID' in combined_per_sequence_df.columns:
                    metrics_prepared_df_for_plot = combined_per_sequence_df.copy()
                    if not metrics_prepared_df_for_plot['ID'].is_unique:
                        metrics_prepared_df_for_plot.drop_duplicates(subset=['ID'], keep='first', inplace=True)
                    metrics_prepared_df_for_plot.set_index('ID', inplace=True)
                
                rscu_prepared_df_for_plot = combined_ca_input_df.copy() # Already indexed correctly
                if not rscu_prepared_df_for_plot.index.is_unique:
                     rscu_prepared_df_for_plot = rscu_prepared_df_for_plot[~rscu_prepared_df_for_plot.index.duplicated(keep='first')]


                if ca_dims_prepared_df_for_plot is not None and metrics_prepared_df_for_plot is not None:
                    metric_features = ['Length', 'TotalCodons', 'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'ENC', 'CAI', 'Fop', 'RCDI', 'ProteinLength', 'GRAVY', 'Aromaticity']
                    available_metric_f = [f for f in metric_features if f in metrics_prepared_df_for_plot.columns]
                    available_rscu_c = sorted([col for col in rscu_prepared_df_for_plot.columns if len(col) == 3 and col.isupper()])
                    features_corr_plot = available_metric_f + available_rscu_c
                    
                    if features_corr_plot:
                        plotting.plot_ca_axes_feature_correlation(
                            ca_dims_df=ca_dims_prepared_df_for_plot,
                            metrics_df=metrics_prepared_df_for_plot,
                            rscu_df=rscu_prepared_df_for_plot,
                            output_dir=str(output_dir_path),
                            file_format=fmt,
                            features_to_correlate=features_corr_plot
                        )
            # Add plot for RSCU comparison if reference data is available
            if reference_data_for_plot is not None and combined_per_sequence_df is not None:
                # Need an aggregate usage df from combined_per_sequence_df or similar for observed RSCU
                # This might require recalculating aggregate RSCU from all sequences if not readily available.
                # For simplicity, this plot might be better if `agg_usage_df_complete` or a similar
                # overall aggregate is passed or calculated here.
                # Let's assume we'd use agg_usage_df_complete if available.
                # This part needs careful data sourcing from the main flow.
                pass # Placeholder - RSCU comparison plotting needs careful data plumbing

        except Exception as plot_err:
            logger.exception(f"Error during combined plot generation for format '{fmt}': {plot_err}")

def _save_main_output_tables(
    output_dir_path: Path,
    combined_per_sequence_df: Optional[pd.DataFrame],
    mean_summary_df: Optional[pd.DataFrame],
    comparison_results_df: Optional[pd.DataFrame]
    # combined_ca_input_df (RSCU wide) is now saved in _perform_and_save_combined_ca
) -> None:
    """Saves the main output CSV tables."""
    if combined_per_sequence_df is not None and not combined_per_sequence_df.empty:
        # This is already saved by _finalize_and_save_per_sequence_metrics
        pass
    if mean_summary_df is not None and not mean_summary_df.empty:
        filepath = output_dir_path / "mean_features_per_gene.csv"
        mean_summary_df.to_csv(filepath, index=False, float_format='%.4f')
        logger.info(f"Mean features per gene saved: {filepath}")
    if comparison_results_df is not None and not comparison_results_df.empty:
        filepath = output_dir_path / "gene_comparison_stats.csv"
        comparison_results_df.to_csv(filepath, index=False, float_format='%.4g')
        logger.info(f"Gene comparison statistics saved: {filepath}")


# --- Main Command Handler Refactored ---

def handle_analyze_command(args: argparse.Namespace) -> None:
    logger.info(f"Running 'analyze' command with input directory: {args.directory}")

    # 1. Setup output directory (deferred from original, now done early by this helper)
    output_dir_path = _setup_output_directory(args.output)

    # 2. Load Reference Data
    reference_weights, reference_data_for_plot = _load_reference_data(args)

    # 3. Get Gene Files and Expected Names
    gene_files, expected_gene_names = _get_gene_files_and_names(args.directory)

    # 4. Determine Number of Processes
    num_processes = _determine_num_processes(args.threads, len(gene_files))

    # 5. Run Gene File Analysis (Parallel/Sequential)
    analyze_results_raw = _run_gene_file_analysis_in_parallel(
        gene_files, args, reference_weights, expected_gene_names, num_processes
    )

    # 6. Collect and Aggregate Initial Results
    (all_per_sequence_dfs, all_ca_input_dfs, successfully_processed_genes, _,
     sequences_by_original_id, all_nucl_freqs_by_gene, all_dinucl_freqs_by_gene) = \
        _collect_and_aggregate_results(analyze_results_raw, expected_gene_names)

    # 7. Analyze "Complete" Sequences and Update Aggregates
    if sequences_by_original_id:
        complete_results = _analyze_complete_sequences_cli(
            sequences_by_original_id, successfully_processed_genes, args, reference_weights
        )
        if complete_results[0] is not None: # Check if agg_usage_df_complete is not None
            _update_aggregate_data_with_complete_results(
                complete_results, all_per_sequence_dfs, all_ca_input_dfs,
                all_nucl_freqs_by_gene, all_dinucl_freqs_by_gene, args
            )

    # 8. Finalize and Save Per-Sequence Metrics (main combined table)
    combined_per_sequence_df = _finalize_and_save_per_sequence_metrics(all_per_sequence_dfs, output_dir_path)
    if combined_per_sequence_df is None:
        logger.error("Failed to produce combined per-sequence metrics. Further analysis may be impacted. Exiting.")
        sys.exit(1) # Critical if this df is needed by subsequent steps

    # 9. Generate Summary Tables (Means, Stats) and Relative Dinucleotide Abundance Table
    mean_summary_df, comparison_results_df, rel_abund_df = _generate_summary_tables_and_stats(
        combined_per_sequence_df, all_nucl_freqs_by_gene, all_dinucl_freqs_by_gene, output_dir_path
    )

    # 10. Perform Combined CA and Save Details
    (combined_ca_input_df_final, ca_results_combined, 
     gene_groups_for_plotting, ca_row_coords) = _perform_and_save_combined_ca(
        all_ca_input_dfs, output_dir_path, args
    )

    # 11. Prepare Color Palette for Plotting
    gene_color_map = _generate_color_palette_for_groups(combined_per_sequence_df)
    
    # 12. Generate All Combined Plots
    _generate_all_combined_plots(
        args, combined_per_sequence_df, gene_color_map, rel_abund_df,
        ca_results_combined, combined_ca_input_df_final, gene_groups_for_plotting,
        ca_row_coords, reference_data_for_plot
    )
    
    # 13. Save Remaining Output Tables
    _save_main_output_tables(output_dir_path, combined_per_sequence_df, mean_summary_df, comparison_results_df)

    logger.info("PyCodon Analyzer 'analyze' command finished successfully.")


def process_analyze_gene_file(
    gene_filepath: str,
    args: argparse.Namespace,
    reference_weights: Optional[Dict[str, float]],
    expected_gene_names: Set[str]
) -> Optional[AnalyzeGeneResultType]: # type: ignore
    """
    Worker function for the 'analyze' subcommand.
    (Content of this function remains the same as in your provided cli.py)
    """
    # ... (existing implementation of process_analyze_gene_file)
    # (Ensure it correctly uses the 'ID' format f"{gene_name}__" + original_id
    # for the per_sequence_df_gene['ID'] column as per previous corrections)
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logging.getLogger("pycodon_analyzer.worker").error(f"[Worker {os.getpid()}] Matplotlib not found.")
    except Exception as backend_err:
        logging.getLogger("pycodon_analyzer.worker").warning(f"[Worker {os.getpid()}] Could not set Matplotlib backend: {backend_err}")

    worker_logger = logging.getLogger(f"pycodon_analyzer.worker.{os.getpid()}")
    gene_name: Optional[str] = extract_gene_name_from_file(gene_filepath)

    if not gene_name or gene_name not in expected_gene_names:
        return None

    worker_logger.debug(f"Starting processing for gene: {gene_name} (File: {os.path.basename(gene_filepath)})")
    agg_usage_df_gene: Optional[pd.DataFrame] = None
    per_sequence_df_gene: Optional[pd.DataFrame] = None
    nucl_freqs_gene: Optional[Dict[str, float]] = None
    dinucl_freqs_gene: Optional[Dict[str, float]] = None
    ca_input_df_gene: Optional[pd.DataFrame] = None
    cleaned_seq_map: Optional[Dict[str, Seq]] = None

    try:
        raw_sequences: List[SeqRecord] = io.read_fasta(gene_filepath)
        if not raw_sequences:
            return (gene_name, "empty file", None, None, None, None, None)

        cleaned_sequences: List[SeqRecord] = utils.clean_and_filter_sequences(raw_sequences, args.max_ambiguity)
        if not cleaned_sequences:
            return (gene_name, "no valid seqs", None, None, None, None, None)
        cleaned_seq_map = {rec.id: rec.seq for rec in cleaned_sequences}

        analysis_results_tuple: tuple = analysis.run_full_analysis(
            cleaned_sequences, args.genetic_code, reference_weights=reference_weights
        )
        agg_usage_df_gene = analysis_results_tuple[0]
        per_sequence_df_gene = analysis_results_tuple[1]
        nucl_freqs_gene = analysis_results_tuple[2]
        dinucl_freqs_gene = analysis_results_tuple[3]
        ca_input_df_gene = analysis_results_tuple[6]

        if not args.skip_plots and agg_usage_df_gene is not None and ca_input_df_gene is not None:
            try:
                long_rscu_df = ca_input_df_gene.reset_index().rename(columns={'index': 'SequenceID'})
                long_rscu_df = long_rscu_df.melt(id_vars=['SequenceID'], var_name='Codon', value_name='RSCU')
                current_genetic_code: Dict[str, str] = utils.get_genetic_code(args.genetic_code)
                long_rscu_df['AminoAcid'] = long_rscu_df['Codon'].map(current_genetic_code.get)
                for fmt in args.plot_formats:
                    plotting.plot_rscu_boxplot_per_gene(long_rscu_df, agg_usage_df_gene, gene_name, args.output, fmt)
            except Exception as plot_prep_err:
                worker_logger.error(f"Failed to prepare/plot RSCU boxplot for {gene_name}: {plot_prep_err}")

        if per_sequence_df_gene is not None and not per_sequence_df_gene.empty:
            if 'ID' in per_sequence_df_gene.columns:
                 per_sequence_df_gene['Original_ID'] = per_sequence_df_gene['ID']
                 per_sequence_df_gene['ID'] = f"{gene_name}__" + per_sequence_df_gene['ID'].astype(str) # Standardized ID
            per_sequence_df_gene['Gene'] = gene_name
        if ca_input_df_gene is not None and not ca_input_df_gene.empty:
            ca_input_df_gene.index = f"{gene_name}__" + ca_input_df_gene.index.astype(str) # Standardized index

        return (gene_name, "success", per_sequence_df_gene, ca_input_df_gene,
                nucl_freqs_gene, dinucl_freqs_gene, cleaned_seq_map)
    except FileNotFoundError:
        return (gene_name, "file not found error", None, None, None, None, None)
    except ValueError as ve:
        return (gene_name, "value error", None, None, None, None, None)
    except Exception as e:
        worker_logger.exception(f"UNEXPECTED ERROR processing gene {gene_name}: {e}")
        return (gene_name, "exception", None, None, None, None, None)


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