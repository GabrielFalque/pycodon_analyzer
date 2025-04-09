# src/pycodon_analyzer/cli.py

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
import pandas as pd # type: ignore
import numpy as np # type: ignore
import glob # To find files matching a pattern
import re   # To extract gene name from filename
import traceback # For detailed error printing
from Bio.SeqRecord import SeqRecord # type: ignore
from Bio.Seq import Seq # type: ignore

# Import necessary functions from local modules
from . import io
from . import analysis
from . import plotting
from .utils import load_reference_usage, get_genetic_code, clean_and_filter_sequences

# --- Find default reference path ---
try:
    # Use files() API for modern importlib.resources (preferred)
    from importlib.resources import files as pkg_resources_files
except ImportError:
    # Fallback for Python < 3.9
    try:
        from importlib_resources import files as pkg_resources_files # type: ignore
    except ImportError:
        pkg_resources_files = None # Indicate API is unavailable

DEFAULT_REF_FILENAME = "human_codon_usage.csv"
DEFAULT_HUMAN_REF_PATH = None # Initialize

print(f"\nAttempting to find default reference file: {DEFAULT_REF_FILENAME}")

# --- Use ONLY importlib.resources ---
if pkg_resources_files:
    try:
        # Construct path using the recommended API
        ref_path_obj = pkg_resources_files('pycodon_analyzer').joinpath('data').joinpath(DEFAULT_REF_FILENAME)
        print(f"  Checking path object: {ref_path_obj}") # Debug print
        if ref_path_obj.is_file():
             DEFAULT_HUMAN_REF_PATH = str(ref_path_obj)
             print(f"  SUCCESS: Found via importlib.resources: {DEFAULT_HUMAN_REF_PATH}")
        else:
             print(f"  INFO: Path object found but is not a file (Package installed, but data file missing?): {ref_path_obj}")
    except (ModuleNotFoundError, Exception) as pkg_err:
         print(f"  INFO: importlib.resources failed to find package/file: {type(pkg_err).__name__}: {pkg_err}")
         print("        (This usually means the package is not installed correctly or data was excluded)")
else:
    print("  ERROR: Cannot use importlib.resources.files() (is importlib_resources installed for Python < 3.9?). Cannot find default reference file reliably.")


# --- Final Check ---
if not DEFAULT_HUMAN_REF_PATH:
    print(f"\nCRITICAL WARNING: Default reference file '{DEFAULT_REF_FILENAME}' could NOT be found.")
    print("                 CAI/Fop/RCDI calculations will fail unless a valid path is provided via --ref.")
    print("                 Ensure the package is correctly installed including data files.")
else:
     print(f"\nDefault Reference Path found: {DEFAULT_HUMAN_REF_PATH}")


# --- Helper function to extract gene name from filename ---
def extract_gene_name_from_file(filename):
    """
    Extracts gene name from filenames like 'gene_XYZ.fasta' or 'gene_ABC.fa'.
    Returns the extracted name (e.g., 'XYZ') or None if pattern doesn't match.
    """
    base = os.path.basename(filename)
    # Match pattern 'gene_' followed by name, ending with common FASTA extensions
    # Allow more characters in gene name, including hyphens
    match = re.match(r'gene_([\w\-.]+)\.(fasta|fa|fna|fas|faa)$', base, re.IGNORECASE)
    # old regex : r'gene_([\w\-.]+?)\.(fasta|fa|fna|fas|faa)$'
    
    if match:
        return match.group(1) # Return the captured gene name part
    else:
        print(f"Warning: Could not extract gene name from filename pattern 'gene_*.(fa|fasta|...)' for '{base}'. Skipping file.", file=sys.stderr)
        return None

# --- Main function ---
def main():
    """Main function to parse arguments and run the analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze codon usage and sequence properties from gene alignment files within a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- INPUT ARGUMENT ---
    parser.add_argument(
        "-d", "--directory",
        required=True,
        help="Path to the input directory containing gene alignment FASTA files "
             "(e.g., gene_ORF1ab.fasta, gene_S.fasta)."
    )

    # --- Output Arguments ---
    parser.add_argument(
        "-o", "--output", default="codon_analysis_results",
        help="Path to the output directory for results."
    )

    # --- Analysis Arguments ---
    parser.add_argument(
        "--genetic_code", type=int, default=1,
        help="NCBI genetic code ID."
    )
    parser.add_argument(
        "--ref", "--reference_usage", dest="reference_usage_file",
        default=DEFAULT_HUMAN_REF_PATH if DEFAULT_HUMAN_REF_PATH else "human",
        help="Path to reference codon usage table (CSV/TSV) for CAI/Fop/RCDI/comparison. "
             "Use 'human' for bundled default, 'none' to disable reference use."
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=1,
        help="Number of processor threads for parallel per-sequence analysis within each gene/complete set. "
             "0 or negative uses all available cores."
    )
    parser.add_argument(
        "--max_ambiguity", type=float, default=15.0,
        help="Maximum allowed percentage of ambiguous bases ('N') per sequence after cleaning (0-100)."
    )


    # --- Plotting Arguments ---
    parser.add_argument(
        "--plot_formats", nargs='+', default=['png'],
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
        help="Add verbosity."
    )


    args = parser.parse_args()

    # --- Argument Validation and Setup ---
    if not os.path.isdir(args.directory):
        print(f"Error: Input directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)
    if args.max_ambiguity < 0 or args.max_ambiguity > 100:
         print(f"Error: --max_ambiguity must be between 0 and 100.", file=sys.stderr)
         sys.exit(1)

    # Setup output directory
    try:
        os.makedirs(args.output, exist_ok=True)
        if args.verbose :
            print(f"Results will be saved to: {args.output}")
    except OSError as e:
        print(f"Error creating output directory '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Resolve and LOAD codon reference file ONCE ---
    reference_data = None
    reference_weights = {} # Initialize as empty dict

    if args.reference_usage_file and args.reference_usage_file.lower() != 'none':
        ref_path_to_load = None
        if args.reference_usage_file.lower() == 'human':
             if DEFAULT_HUMAN_REF_PATH and os.path.isfile(DEFAULT_HUMAN_REF_PATH):
                 ref_path_to_load = DEFAULT_HUMAN_REF_PATH
             else:
                 print("Warning: Default human reference file requested ('human') but not found or path invalid. Cannot load reference.", file=sys.stderr)
        elif os.path.isfile(args.reference_usage_file):
            ref_path_to_load = args.reference_usage_file
        else:
            print(f"Warning: Specified reference file not found: {args.reference_usage_file}. Cannot load reference.", file=sys.stderr)

        if ref_path_to_load:
             if args.verbose :
                print(f"Loading codon usage reference table: {ref_path_to_load}...")
             try:
                # Get genetic code dict needed by load_reference_usage
                genetic_code_dict = get_genetic_code(args.genetic_code)
                reference_data = load_reference_usage(ref_path_to_load, genetic_code_dict, args.genetic_code)

                if reference_data is not None and not reference_data.empty:
                     # Extract weights if loading succeeded
                     reference_weights = reference_data['Weight'].to_dict()
                     if args.verbose :
                        print("Reference data loaded and weights calculated successfully.")
                
                else:
                     # load_reference_usage prints warnings if processing fails
                     print("Warning: Failed to load or process reference data after reading. CAI/Fop/RCDI calculations will use NaN.", file=sys.stderr)
                     reference_data = None # Ensure it's None if processing failed
                     reference_weights = {}
             except NotImplementedError as nie: # Catch error for unsupported genetic codes
                  print(f"Error loading reference: {nie}", file=sys.stderr)
                  sys.exit(1)
             except Exception as load_err:
                  print(f"Error loading reference file {ref_path_to_load}: {load_err}", file=sys.stderr)
                  reference_data = None
                  reference_weights = {}
    else:
         print("No reference file specified (--ref set to 'none' or invalid). Skipping reference-based calculations (CAI/Fop/RCDI).")
    # --- End reference loading ---


    # Determine number of threads
    num_threads = args.threads
    if num_threads <= 0:
        try:
            num_threads = os.cpu_count() or 1
            print(f"Using {num_threads} threads (all available).")
        except NotImplementedError:
            num_threads = 1
            print("Could not determine number of cores, using 1 thread.")
    elif num_threads == 1:
         print("Using 1 thread (sequential processing).")
    else:
         print(f"Using {num_threads} threads.")


    # --- Run Workflow ---
    try:
        # --- Find gene alignment files ---
        search_pattern = os.path.join(args.directory, "gene_*.*")
        gene_files = glob.glob(search_pattern)
        valid_extensions = {".fasta", ".fa", ".fna", ".fas", ".faa"}
        gene_files = sorted([f for f in gene_files if os.path.splitext(f)[1].lower() in valid_extensions]) # Sort for consistent processing order

        if not gene_files:
            print(f"Error: No gene alignment files (matching 'gene_*.(fasta|fa|...)') found in directory: {args.directory}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(gene_files)} potential gene alignment files to process.")

        # --- Determine the set of expected gene names ---

        expected_gene_names = set()
        for fpath in gene_files:
             gname = extract_gene_name_from_file(fpath)
             if gname:
                  expected_gene_names.add(gname)

        if not expected_gene_names:
             sys.exit("Error: Could not extract any valid gene names from input files.")
        print(f"Expecting data for {len(expected_gene_names)} genes: {', '.join(sorted(list(expected_gene_names)))}")

        # --- Loop through genes: Read, CLEAN, Analyze, COLLECT ---
        all_per_sequence_dfs = []
        all_ca_input_dfs = {}
        processed_genes = set()
        failed_genes = []
        sequences_by_original_id = {} # Store CLEANED sequences {orig_id -> {gene_name: Bio.Seq}}
        # --- Dictionaries to store frequencies per gene ---
        all_nucl_freqs_by_gene = {}
        all_dinucl_freqs_by_gene = {}

        print("\nCleaning, filtering and analyzing of gene sequences...")
        for gene_filepath in gene_files:
            gene_name = extract_gene_name_from_file(gene_filepath)
            # Ensure we only process genes identified initially
            if not gene_name or gene_name not in expected_gene_names:
                continue # Skip if name extraction failed or wasn't in initial list

            print(f"  Processing gene: {gene_name} (File: {os.path.basename(gene_filepath)})")
            try:
                # Read sequences
                raw_sequences = io.read_fasta(gene_filepath, args.verbose)
                if not raw_sequences:
                    print(f"    Warning: No sequences found in file {os.path.basename(gene_filepath)}. Skipping gene {gene_name}.")
                    failed_genes.append(gene_name + " (empty file)")
                    continue

                # Clean and Filter Sequences
                if args.verbose :
                    print(f"    Cleaning and filtering {len(raw_sequences)} sequences for {gene_name} (Max N: {args.max_ambiguity}%)...")
                cleaned_sequences = clean_and_filter_sequences(raw_sequences, args.max_ambiguity)

                if not cleaned_sequences:
                    print(f"    Warning: No sequences remaining after cleaning/filtering for gene {gene_name}. Skipping analysis.")
                    # Consider if this is a failure or just no valid data
                    failed_genes.append(gene_name + " (no valid seqs)")
                    continue

                # Store CLEANED sequences
                for seq_record in cleaned_sequences:
                    original_id = seq_record.id
                    if original_id not in sequences_by_original_id:
                        sequences_by_original_id[original_id] = {}
                    sequences_by_original_id[original_id][gene_name] = seq_record.seq

                # Run analysis on the CLEANED gene alignment
                agg_usage_df_gene, per_sequence_df_gene, \
                nucl_freqs_gene, dinucl_freqs_gene, \
                _, ca_results_gene_obj, ca_input_df_gene = analysis.run_full_analysis(
                    cleaned_sequences,
                    args.genetic_code,
                    reference_weights=reference_weights,
                    num_threads=num_threads,
                    perform_ca=False
                )

                # --- Store frequencies ---
                if nucl_freqs_gene:
                    all_nucl_freqs_by_gene[gene_name] = nucl_freqs_gene
                if dinucl_freqs_gene:
                    all_dinucl_freqs_by_gene[gene_name] = dinucl_freqs_gene
                # ---

                # Store results (modifying ID in per_sequence_df)
                if per_sequence_df_gene is not None and not per_sequence_df_gene.empty:
                    # Keep original ID if needed, modify main ID column
                    per_sequence_df_gene['Original_ID'] = per_sequence_df_gene['ID']
                    per_sequence_df_gene['ID'] = per_sequence_df_gene['ID'].astype(str) + "_" + gene_name
                    per_sequence_df_gene['Gene'] = gene_name
                    all_per_sequence_dfs.append(per_sequence_df_gene)
                else:
                    print(f"    Note: No per-sequence metrics generated for gene {gene_name}.", file=sys.stderr)


                if ca_input_df_gene is not None and not ca_input_df_gene.empty:
                    # Prepend gene name to sequence IDs (index) for unique combined CA index
                    ca_input_df_gene.index = f"{gene_name}__" + ca_input_df_gene.index.astype(str)
                    all_ca_input_dfs[gene_name] = ca_input_df_gene
                # else: # Optional debug/info
                    # print(f"    Note: No data generated for CA input for gene {gene_name}.")

                # --- Call RSCU distribution plot for this gene ---
                # NOT USED ANYMORE
                """
                if not args.skip_plots and agg_usage_df_gene is not None and not agg_usage_df_gene.empty:
                    for fmt in args.plot_formats:
                        try:
                            plotting.plot_rscu_distribution_per_gene(
                                agg_usage_df_gene, gene_name, args.output, fmt, args.verbose
                            )
                        except Exception as gene_plot_err:
                            print(f"    Error generating RSCU distribution plot for {gene_name} (format {fmt}): {gene_plot_err}", file=sys.stderr)
                """

                # --- Call RSCU BOXPLOT plot for this gene ---
                if not args.skip_plots and ca_input_df_gene is not None and not ca_input_df_gene.empty and agg_usage_df_gene is not None and not agg_usage_df_gene.empty:
                    try:
                        # Melt the wide ca_input_df to long format for boxplots
                        long_rscu_df = ca_input_df_gene.reset_index().rename(columns={'index': 'SequenceID'})
                        long_rscu_df = long_rscu_df.melt(
                            id_vars=['SequenceID'], var_name='Codon', value_name='RSCU'
                        )
                        # Add AminoAcid column
                        gc_dict = get_genetic_code(args.genetic_code)
                        long_rscu_df['AminoAcid'] = long_rscu_df['Codon'].map(gc_dict.get)
                        long_rscu_df['Gene'] = gene_name # Not strictly needed by plot func but good for consistency

                        # Call the plotting function for each format, passing BOTH dfs
                        for fmt in args.plot_formats:
                            plotting.plot_rscu_boxplot_per_gene(
                                long_rscu_df,     # Data for boxplot distributions
                                agg_usage_df_gene,# Data for label coloring (means)
                                gene_name,
                                args.output,
                                fmt,
                                args.verbose
                            )
                    except Exception as gene_plot_err:
                        print(f"    Error preparing/generating RSCU boxplot for {gene_name}: {gene_plot_err}", file=sys.stderr)
                        # traceback.print_exc()

                processed_genes.add(gene_name)

            except Exception as e:
                print(f"    ERROR processing gene {gene_name} from file {os.path.basename(gene_filepath)}: {e}", file=sys.stderr)
                traceback.print_exc() # Print detailed traceback for gene-specific errors
                failed_genes.append(gene_name + " (exception)")

        # --- Update the set of actually processed genes in case some failed ---
        # This ensures we only require genes that could actually be processed
        successfully_processed_genes = processed_genes
        if len(successfully_processed_genes) < len(expected_gene_names):
             print(f"Warning: Only processed {len(successfully_processed_genes)} out of {len(expected_gene_names)} expected genes due to errors or lack of valid sequences.", file=sys.stderr)
             if not successfully_processed_genes:
                  sys.exit("Error: No genes successfully processed.")

        # --- Analyze "Complete" Sequences ---
        print("\nAnalyzing concatenated 'complete' sequences...")
        complete_seq_records_to_analyze = [] # Validated records ready for analysis

        if sequences_by_original_id:
            max_ambiguity_pct_complete = args.max_ambiguity # Use same threshold

            for original_id, gene_seq_map in sequences_by_original_id.items():
                # --- Check if this ID has data for ALL successfully processed genes ---
                if set(gene_seq_map.keys()) != successfully_processed_genes:
                     # print(f"    Skipping 'complete' for {original_id}: Missing data for one or more processed genes.") # Verbose option
                     continue
                
                # print(f"    Concatenating {len(gene_seq_map)} cleaned genes for {original_id}...") # Reduce verbosity
                concatenated_seq_str = ""
                # Concatenate in alphabetical order of gene names for consistency
                for gene_name in sorted(gene_seq_map.keys()):
                     concatenated_seq_str += str(gene_seq_map[gene_name])

                if not concatenated_seq_str:
                    # print(f"    Warning: Concatenation resulted in empty sequence for {original_id}.") # Reduce verbosity
                    continue

                # Validate concatenated sequence
                if len(concatenated_seq_str) % 3 != 0:
                     print(f"    Warning: Concatenated sequence for {original_id} length ({len(concatenated_seq_str)}) "
                           f"is not multiple of 3. Skipping 'complete' analysis for this ID.", file=sys.stderr)
                     continue

                n_count = concatenated_seq_str.count('N')
                seq_len = len(concatenated_seq_str)
                ambiguity_pct = (n_count / seq_len) * 100 if seq_len > 0 else 0
                if ambiguity_pct > max_ambiguity_pct_complete:
                     print(f"    Warning: Concatenated sequence for {original_id} ambiguity ({ambiguity_pct:.1f}%) "
                           f"exceeds threshold ({max_ambiguity_pct_complete}%). Skipping 'complete' analysis for this ID.", file=sys.stderr)
                     continue

                # If valid, create record for analysis
                complete_seq = Seq(concatenated_seq_str)
                complete_record = SeqRecord(
                    complete_seq,
                    id=original_id, # Use original ID for analysis input
                    description=f"Concatenated {len(gene_seq_map)} genes for {original_id} [cleaned]"
                )
                complete_seq_records_to_analyze.append(complete_record)


            if complete_seq_records_to_analyze:
                print(f"Running analysis on {len(complete_seq_records_to_analyze)} valid 'complete' sequence records...")
                try:
                    # Run analysis on the list of valid complete sequences
                    agg_usage_df_complete, per_sequence_df_complete, \
                    nucl_freqs_complete, dinucl_freqs_complete, \
                    _, _, ca_input_df_complete = analysis.run_full_analysis(
                        complete_seq_records_to_analyze,
                        args.genetic_code,
                        reference_weights=reference_weights,
                        num_threads=num_threads,
                        perform_ca=False
                    )

                    # --- Store "complete" frequencies ---
                    if nucl_freqs_complete:
                        all_nucl_freqs_by_gene['complete'] = nucl_freqs_complete
                    if dinucl_freqs_complete:
                        all_dinucl_freqs_by_gene['complete'] = dinucl_freqs_complete

                    # --- Call RSCU distribution plot for "complete" ---
                    # NOT USED ANYMORE
                    """
                    if not args.skip_plots and agg_usage_df_complete is not None and not agg_usage_df_complete.empty:
                        for fmt in args.plot_formats:
                            try:
                                plotting.plot_rscu_distribution_per_gene(
                                    agg_usage_df_complete, 'complete', args.output, fmt, args.verbose
                                )
                            except Exception as complete_plot_err:
                                print(f"    Error generating RSCU distribution plot for 'complete' (format {fmt}): {complete_plot_err}", file=sys.stderr)
                    """

                    # --- Call RSCU boxplot for "complete" ---
                    if not args.skip_plots and ca_input_df_complete is not None and not ca_input_df_complete.empty and agg_usage_df_complete is not None and not agg_usage_df_complete.empty:
                        try:
                            # Melt the ca_input_df for "complete"
                            long_rscu_df_comp = ca_input_df_complete.reset_index().rename(columns={'index': 'SequenceID'})
                            long_rscu_df_comp['SequenceID'] = long_rscu_df_comp['SequenceID'].str.split('__', n=1).str[1]
                            long_rscu_df_comp = long_rscu_df_comp.melt(
                                id_vars=['SequenceID'], var_name='Codon', value_name='RSCU'
                            )
                            gc_dict = get_genetic_code(args.genetic_code)
                            long_rscu_df_comp['AminoAcid'] = long_rscu_df_comp['Codon'].map(gc_dict.get)
                            long_rscu_df_comp['Gene'] = 'complete'

                            # Call plot function passing both DFs
                            for fmt in args.plot_formats:
                                plotting.plot_rscu_boxplot_per_gene(
                                    long_rscu_df_comp,         # Data for boxplots
                                    agg_usage_df_complete,     # Data for labels
                                    'complete', args.output, fmt, args.verbose
                                )
                        except Exception as complete_plot_err:
                            print(f"    Error preparing/generating RSCU boxplot for 'complete': {complete_plot_err}", file=sys.stderr)
                                
                    # Add results to combined lists
                    if per_sequence_df_complete is not None and not per_sequence_df_complete.empty:
                        # Modify ID and add Gene column
                        per_sequence_df_complete['Original_ID'] = per_sequence_df_complete['ID']
                        per_sequence_df_complete['ID'] = per_sequence_df_complete['ID'].astype(str) + "_complete"
                        per_sequence_df_complete['Gene'] = 'complete' # Assign pseudo-gene name
                        all_per_sequence_dfs.append(per_sequence_df_complete)
                    else:
                        print("    Note: No per-sequence metrics generated for 'complete' sequences.", file=sys.stderr)


                    if ca_input_df_complete is not None and not ca_input_df_complete.empty:
                        # Prepend 'complete__' to index for CA uniqueness
                        ca_input_df_complete.index = "complete__" + ca_input_df_complete.index.astype(str)
                        all_ca_input_dfs['complete'] = ca_input_df_complete # Add to dict
                    # else: # Optional debug/info
                        # print("    Note: No data generated for CA input for 'complete' sequences.")

                except Exception as e:
                    print(f"    ERROR during 'complete' sequence analysis: {e}", file=sys.stderr)
                    traceback.print_exc()
            else:
                print("No valid 'complete' sequences to analyze after concatenation and filtering.")
        else:
            print("No sequences collected to generate 'complete' sequences.")
        # --- End "Complete" Sequence Analysis ---


        # --- Combine results ---
        print("Combining results...")
        if not all_per_sequence_dfs:
             sys.exit("Error: No per-sequence results were generated to combine.")

        try:
            combined_per_sequence_df = pd.concat(all_per_sequence_dfs, ignore_index=True)
        except Exception as concat_err:
             print(f"Error concatenating per-sequence results: {concat_err}", file=sys.stderr)
             sys.exit(1)


        # Save combined per-sequence table
        per_seq_filepath = os.path.join(args.output, "per_sequence_metrics_all_genes.csv")
        try:
            combined_per_sequence_df.to_csv(per_seq_filepath, index=False)
            if args.verbose :
                print(f"Combined per-sequence metrics table saved to: {per_seq_filepath}")
        except Exception as save_err:
            print(f"Error saving combined metrics table to '{per_seq_filepath}': {save_err}", file=sys.stderr)


        # --- Calculate and Save Mean Features per Gene ---
        print("Calculating mean features per gene...")
        mean_features = [
            'GC', 'GC1', 'GC2', 'GC3', 'GC12', # Case sensitive based on DataFrame columns
            'RCDI', 'ENC', 'CAI',
            'Aromaticity', # The actual column name
            'GRAVY'        # The actual column name
        ]
        # Ensure all requested columns exist, warn if not
        available_features = [f for f in mean_features if f in combined_per_sequence_df.columns]
        missing_features = [f for f in mean_features if f not in available_features]
        if missing_features:
            print(f"Warning: Cannot calculate mean for missing features: {', '.join(missing_features)}", file=sys.stderr)

        if 'Gene' not in combined_per_sequence_df.columns:
            print("Error: 'Gene' column missing in combined data, cannot calculate mean features per gene.", file=sys.stderr)
            mean_summary_df = pd.DataFrame() # Create empty df
        elif not available_features:
            print("Warning: No available features found to calculate means.", file=sys.stderr)
            mean_summary_df = pd.DataFrame({'Gene': combined_per_sequence_df['Gene'].unique()}) # Df with just gene names
        else:
            try:
                # Convert relevant columns to numeric, coercing errors
                for col in available_features:
                    combined_per_sequence_df[col] = pd.to_numeric(combined_per_sequence_df[col], errors='coerce')

                # Group by Gene and calculate mean, dropping NA values during mean calc
                mean_summary_df = combined_per_sequence_df.groupby('Gene')[available_features].mean(numeric_only=True) # numeric_only handles potential non-numeric columns safely
                mean_summary_df = mean_summary_df.reset_index() # Make 'Gene' a column again

                # Optional: Rename columns for clarity in output file
                mean_summary_df.rename(columns={'Aromaticity': 'Aromaticity_pct'}, inplace=True)

            except Exception as mean_err:
                print(f"Error calculating mean features per gene: {mean_err}", file=sys.stderr)
                mean_summary_df = pd.DataFrame() # Create empty df on error


        # Save the mean summary table if not empty
        if not mean_summary_df.empty:
            mean_summary_filepath = os.path.join(args.output, "mean_features_per_gene.csv")
            try:
                mean_summary_df.to_csv(mean_summary_filepath, index=False, float_format='%.4f') # Format float output
                if args.verbose :
                    print(f"Mean features per gene saved to: {mean_summary_filepath}")
            except Exception as save_err:
                print(f"Error saving mean features table to '{mean_summary_filepath}': {save_err}", file=sys.stderr)
        # --- End Mean Features ---


        # --- Perform and Save Statistical Comparisons ---
        print("Performing statistical comparison between genes...")
        # Define features to compare (must match columns in combined_per_sequence_df)
        features_to_compare = [
            'GC', 'GC1', 'GC2', 'GC3', 'GC12',
            'RCDI', 'ENC', 'CAI',
            'Aromaticity', 'GRAVY'
            ]
        comparison_results_df = analysis.compare_features_between_genes(
            combined_per_sequence_df,
            features=features_to_compare,
            method='kruskal' # Or use args.stat_test if added
            )

        if comparison_results_df is not None and not comparison_results_df.empty:
            comparison_filepath = os.path.join(args.output, "gene_comparison_stats.csv")
            try:
                comparison_results_df.to_csv(comparison_filepath, index=False, float_format='%.4g') # Use general format
                if args.verbose:
                    print(f"Gene comparison statistics saved to: {comparison_filepath}")
            except Exception as save_err:
                print(f"Error saving gene comparison stats to '{comparison_filepath}': {save_err}", file=sys.stderr)
        else:
            print("No statistical comparison results generated.")
        # --- End Stat Comparisons ---


        # --- Calculate Relative Dinucleotide Abundance ---
        print("Calculating relative dinucleotide abundances...")
        all_rel_abund_data = []
        # Get the list of genes for which we have both sets of frequencies
        genes_to_process = sorted(list(set(all_nucl_freqs_by_gene.keys()) & set(all_dinucl_freqs_by_gene.keys())))

        for gene_name in genes_to_process:
             nucl_freqs = all_nucl_freqs_by_gene[gene_name]
             dinucl_freqs = all_dinucl_freqs_by_gene[gene_name]
             try:
                 rel_abund = analysis.calculate_relative_dinucleotide_abundance(nucl_freqs, dinucl_freqs)
                 # Convert to long format for plotting
                 for dinucl, ratio in rel_abund.items():
                      all_rel_abund_data.append({'Gene': gene_name, 'Dinucleotide': dinucl, 'RelativeAbundance': ratio})
             except Exception as e:
                  print(f"Warning: Could not calculate relative dinucleotide abundance for '{gene_name}': {e}", file=sys.stderr)

        if not all_rel_abund_data:
             print("Warning: No relative dinucleotide abundance data generated.", file=sys.stderr)
             rel_abund_df = pd.DataFrame() # Empty df
        else:
             rel_abund_df = pd.DataFrame(all_rel_abund_data)
        # --- End Calculation ---

        # --- Combined Correspondence Analysis ---
        combined_ca_input_df = None
        ca_results_combined = None
        gene_groups_for_plotting = None # Series mapping combined index to gene name

        if not args.skip_ca and all_ca_input_dfs:
            print("Performing combined Correspondence Analysis...")
            try:
                if all_ca_input_dfs: # Check dictionary is not empty
                    combined_ca_input_df = pd.concat(all_ca_input_dfs.values())
                    combined_ca_input_df.fillna(0.0, inplace=True)
                    combined_ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True) # Replace Inf with 0

                    if not combined_ca_input_df.empty:
                        # Create grouping data by splitting the index (e.g., "GeneName__SeqID")
                        group_data = combined_ca_input_df.index.str.split('__', n=1).str[0]

                        # Ensure the index of the Series matches the DataFrame's index
                        gene_groups_for_plotting = pd.Series(
                            data=group_data,               # The gene names extracted
                            index=combined_ca_input_df.index, # Use the original combined index
                            name='Gene'                    # Name the Series (used for legend title)
                        )

                        # Perform CA using the dedicated function
                        ca_results_combined = analysis.perform_ca(combined_ca_input_df)
                        if ca_results_combined:
                            print("Combined Correspondence Analysis complete.")
                        else:
                            # perform_ca prints warnings/errors
                            print("Warning: Combined CA calculation failed or produced no result.")
                            combined_ca_input_df = None # Reset df if CA fails
                    else:
                        print("Warning: Combined CA input data is empty after concatenation.")
                        combined_ca_input_df = None
                else:
                    print("Warning: No data collected for combined CA.")

            except Exception as e:
                print(f"Error during combined Correspondence Analysis: {e}", file=sys.stderr)
                traceback.print_exc()
                ca_results_combined = None
                combined_ca_input_df = None # Reset df on error
        else:
             print("Skipping combined Correspondence Analysis as requested or no data available.")


        # --- Save CA Details ---
        if ca_results_combined is not None and combined_ca_input_df is not None:
            print("Saving CA details (coordinates, contributions, eigenvalues)...")
            try:
                # Row Coordinates (Sequences/Genes)
                ca_rows = ca_results_combined.row_coordinates(combined_ca_input_df)
                ca_rows.to_csv(os.path.join(args.output, "ca_row_coordinates.csv"), float_format='%.5f')

                # Column Coordinates (Codons)
                ca_cols = ca_results_combined.column_coordinates(combined_ca_input_df)
                ca_cols.to_csv(os.path.join(args.output, "ca_col_coordinates.csv"), float_format='%.5f')

                # Column Contributions
                ca_contrib = ca_results_combined.column_contributions_
                ca_contrib.to_csv(os.path.join(args.output, "ca_col_contributions.csv"), float_format='%.5f')

                # Eigenvalues / Variance Explained
                ca_eigen = ca_results_combined.eigenvalues_summary
                ca_eigen.to_csv(os.path.join(args.output, "ca_eigenvalues.csv"), float_format='%.5f')

                print("CA details saved.")
            except AttributeError as ae:
                print(f"Warning: Could not access all attributes from CA results object to save details: {ae}", file=sys.stderr)
            except Exception as ca_save_err:
                print(f"Error saving CA details: {ca_save_err}", file=sys.stderr)
        # --- End Save CA Details ---

        # --- Save Per-Sequence RSCU (Wide Format) ---
        if combined_ca_input_df is not None and not combined_ca_input_df.empty:
            if args.verbose:
                print("Saving per-sequence RSCU values (wide format)...")
            rscu_wide_filepath = os.path.join(args.output, "per_sequence_rscu_wide.csv")
            try:
                # The index already contains Gene__SeqID or complete__SeqID
                combined_ca_input_df.to_csv(rscu_wide_filepath, float_format='%.4f')
                if args.verbose:
                    print(f"Per-sequence RSCU table saved to: {rscu_wide_filepath}")
            except Exception as rscu_save_err:
                print(f"Error saving per-sequence RSCU table: {rscu_save_err}", file=sys.stderr)
        # --- End Save Per-Sequence RSCU ---


        # --- Generate COMBINED Plots ---
        if not args.skip_plots:
            print("Generating combined plots...")
            # Define number of dims/contributors for CA plots
            n_ca_dims_variance = 10
            n_ca_contrib_top = 10

            for fmt in args.plot_formats:
                try:
                    # --- GC Plot Call ---
                    plotting.plot_gc_means_barplot(combined_per_sequence_df, args.output, file_format=fmt, group_by='Gene', verbose=args.verbose)
                    # ---

                    plotting.plot_enc_vs_gc3(combined_per_sequence_df, args.output, file_format=fmt, group_by='Gene', verbose=args.verbose)
                    plotting.plot_neutrality(combined_per_sequence_df, args.output, file_format=fmt, group_by='Gene', verbose=args.verbose)

                    # --- Call Relative Dinucleotide Abundance plot ---
                    if not rel_abund_df.empty:
                        plotting.plot_relative_dinucleotide_abundance(rel_abund_df, args.output, file_format=fmt, verbose=args.verbose)


                    # --- Check if CA was successful before plotting CA results ---
                    if ca_results_combined is not None and combined_ca_input_df is not None:
                        # CA Biplot (already implemented)
                        plotting.plot_ca(ca_results_combined, combined_ca_input_df, args.output,
                                        file_format=fmt, comp_x=args.ca_dims[0], comp_y=args.ca_dims[1],
                                        groups=gene_groups_for_plotting,
                                        filename_suffix="_combined_by_gene", verbose=args.verbose)

                        # --- CA Variance Explained Plot ---
                        plotting.plot_ca_variance(ca_results_combined, n_dims=n_ca_dims_variance,
                                                 output_dir=args.output, file_format=fmt, verbose=args.verbose)

                        # --- CA Contribution Plots (Dim 1 and Dim 2) ---
                        plotting.plot_ca_contribution(ca_results_combined, dimension=0, n_top=n_ca_contrib_top,
                                                    output_dir=args.output, file_format=fmt, verbose=args.verbose)
                        plotting.plot_ca_contribution(ca_results_combined, dimension=1, n_top=n_ca_contrib_top,
                                                    output_dir=args.output, file_format=fmt, verbose=args.verbose)
                    elif not args.skip_ca:
                        print("Skipping CA-related plots as CA calculation failed or produced no results.")
                    # --- End CA Plotting ---

                    # --- Call Correlation Heatmap ---
                    # Select features for correlation (can be same as stats or different)
                    features_for_corr = [
                        'GC', 'GC1', 'GC2', 'GC3', 'GC12', 'ENC',
                        'CAI', 'RCDI', 'Aromaticity', 'GRAVY', 'Length', 'TotalCodons' # Add length?
                        ]
                    # Filter for columns that actually exist in the final combined df
                    features_for_corr = [f for f in features_for_corr if f in combined_per_sequence_df.columns]
                    if len(features_for_corr) > 1:
                        plotting.plot_correlation_heatmap(combined_per_sequence_df, features_for_corr, args.output, fmt, verbose=args.verbose)
                    # ---

                    # ... (Optional: Aggregate plots like RSCU, Freq, Dinuc - need aggregate data) ...

                except Exception as plot_err:
                    print(f"Error occurred during plot generation for format '{fmt}': {plot_err}", file=sys.stderr)
                    traceback.print_exc()
        else:
            print("Skipping plot generation as requested.")

        print("\nAnalysis complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e: # Catch ValueErrors raised during processing
        print(f"Data Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except ImportError as e:
         print(f"Import Error: {e}. Please ensure all dependencies are installed.", file=sys.stderr)
         sys.exit(1)
    except NotImplementedError as e:
         print(f"Feature Error: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in the main workflow: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()