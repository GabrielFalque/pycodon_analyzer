# src/pycodon_analyzer/utils.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Utility functions and constants for the pycodon_analyzer package.
"""
import os
import sys
import logging # <-- Import logging
import math
import csv
from typing import List, Dict, Optional, Tuple, Set, Union, Any

# Third-party imports with checks
try:
    import pandas as pd
except ImportError:
    print("CRITICAL ERROR: pandas is required but not installed.", file=sys.stderr)
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("CRITICAL ERROR: numpy is required but not installed.", file=sys.stderr)
    sys.exit(1)
try:
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
except ImportError:
    print("CRITICAL ERROR: Biopython is required but not installed.", file=sys.stderr)
    sys.exit(1)

# --- Configure logging for this module ---
logger = logging.getLogger(__name__)


# --- Constants ---
# Standard DNA Genetic Code (NCBI table 1)
STANDARD_GENETIC_CODE: Dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Valid DNA characters (including ambiguity codes and gaps)
VALID_DNA_CHARS: Set[str] = set('ATCGN-')
VALID_CODON_CHARS: Set[str] = set('ATCG') # Characters allowed within a valid codon for counting

# --- Hydropathy Scale (Kyte & Doolittle, 1982) ---
KYTE_DOOLITTLE_HYDROPATHY: Dict[str, float] = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# --- Default Human Codon Weights (Placeholder - Only used if loading fails) ---
# These are example values and NOT biologically validated for CAI.
DEFAULT_HUMAN_CAI_WEIGHTS: Dict[str, float] = {
    'TTT': 0.45, 'TTC': 0.55, 'TTA': 0.07, 'TTG': 0.13,
    # ... only a few examples ...
}

# --- Ambiguity Handling ---
AMBIGUOUS_DNA_LETTERS: str = 'RYSWKMBDHVN' # Include N here for replacement logic
AMBIGUOUS_TO_N_MAP: Dict[int, int] = str.maketrans(AMBIGUOUS_DNA_LETTERS, 'N' * len(AMBIGUOUS_DNA_LETTERS))

# --- Define standard start/stop codons ---
STANDARD_START_CODONS: Set[str] = {'ATG'}
STANDARD_STOP_CODONS: Set[str] = {'TAA', 'TAG', 'TGA'}

# --- Functions ---

def get_genetic_code(code_id: int = 1) -> Dict[str, str]:
    """
    Returns a dictionary representing a genetic code.
    Currently only supports the standard code (ID=1).

    Args:
        code_id (int): The NCBI genetic code ID (default: 1).

    Returns:
        Dict[str, str]: A dictionary mapping codons to amino acids or '*'.

    Raises:
        NotImplementedError: If a code other than 1 is requested.
    """
    if code_id == 1:
        return STANDARD_GENETIC_CODE.copy() # Return a copy to prevent modification
    else:
        # Log before raising might be useful if more codes were planned
        logger.error(f"Genetic code ID {code_id} is not implemented yet.")
        raise NotImplementedError(f"Genetic code ID {code_id} is not implemented yet.")

def get_synonymous_codons(genetic_code: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Groups codons by the amino acid they encode.

    Args:
        genetic_code (Dict[str, str]): A dictionary mapping codons to amino acids.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are amino acids (or '*') and
                              values are lists of codons encoding that amino acid.
    """
    syn_codons: Dict[str, List[str]] = {}
    if not genetic_code:
        logger.warning("get_synonymous_codons called with an empty genetic code dictionary.")
        return syn_codons

    try:
        for codon, aa in genetic_code.items():
            # Basic validation of codon/aa? Assume input dict is correct for now.
            if aa not in syn_codons:
                syn_codons[aa] = []
            syn_codons[aa].append(codon)
    except AttributeError:
        logger.error("Invalid genetic_code dictionary passed to get_synonymous_codons (expected dict).")
        return {} # Return empty on bad input type
    return syn_codons


def load_reference_usage(
    filepath: str,
    genetic_code: Dict[str, str],
    genetic_code_id: int = 1,
    delimiter: Optional[str] = None # New parameter
) -> Optional[pd.DataFrame]:
    """
    Loads a reference codon usage table (e.g., for CAI weights or comparison).
    Expects a CSV or TSV file with columns for Codon and one of:
    Frequency, Count, RSCU, or 'Frequency (per thousand)'.
    Infers delimiter if not specified. Calculates RSCU and CAI weights (w).

    Args:
        filepath (str): Path to the reference file.
        genetic_code (Dict[str, str]): Genetic code mapping dictionary.
        genetic_code_id (int): The ID of the genetic code being used (default: 1).
        delimiter (Optional[str]): The delimiter used in the file. If None,
                                   it will be auto-detected. Default None.

    Returns:
        Optional[pd.DataFrame]: DataFrame with Codon as index and columns
                                ['AminoAcid', 'Frequency', 'RSCU', 'Weight'].
                                Returns None if file cannot be loaded or parsed correctly.
    """
    if not os.path.isfile(filepath):
        logger.error(f"Reference file not found: {filepath}")
        return None

    df: Optional[pd.DataFrame] = None
    read_error: Optional[Exception] = None
    used_delimiter: Optional[str] = delimiter # Keep track of the delimiter used

    # --- Attempt to read the file ---
    try:
        if used_delimiter:
            logger.debug(f"Attempting to read reference file '{os.path.basename(filepath)}' using specified delimiter: '{used_delimiter}'")
            df = pd.read_csv(filepath, sep=used_delimiter, engine='python', comment='#')
        else:
            # Try to sniff the delimiter first
            try:
                with open(filepath, 'r', newline='') as csvfile:
                    # Read a sample of the file for sniffing
                    sample = csvfile.read(2048) # Read more bytes for better sniffing
                    csvfile.seek(0) # Reset file pointer
                    dialect = csv.Sniffer().sniff(sample, delimiters=',\t; ') # Common delimiters
                    used_delimiter = dialect.delimiter
                    logger.info(f"Delimiter sniffed for '{os.path.basename(filepath)}': '{used_delimiter}'")
                    df = pd.read_csv(filepath, sep=used_delimiter, engine='python', comment='#')
            except csv.Error as sniff_err:
                logger.warning(f"Could not reliably sniff delimiter for '{os.path.basename(filepath)}': {sniff_err}. Falling back to trying common delimiters.")
                # Fallback to trying a list of delimiters
                delimiters_to_try: List[Optional[str]] = ['\t', ',', None] # Try tab, then comma, then pandas auto-detect
                for i, delim_try in enumerate(delimiters_to_try):
                    attempt_desc = f"delimiter '{delim_try}'" if delim_try else "pandas auto-detection"
                    logger.debug(f"Attempting to read reference file '{os.path.basename(filepath)}' using {attempt_desc}...")
                    try:
                        df = pd.read_csv(filepath, sep=delim_try, engine='python', comment='#')
                        if df is not None and (len(df.columns) > 1 or delim_try is None):
                            logger.info(f"Successfully read reference file using {attempt_desc}.")
                            used_delimiter = delim_try if delim_try else "auto" # Store what worked
                            read_error = None
                            break
                        else:
                            logger.debug(f"Reading with {attempt_desc} resulted in <= 1 column or empty DataFrame. Trying next.")
                            df = None # Reset df if it wasn't a good read
                            if i == len(delimiters_to_try) - 1 and df is None: # Last attempt failed
                                logger.error(f"Could not parse reference file '{os.path.basename(filepath)}' with any fallback delimiter.")
                                return None
                    except pd.errors.ParserError as pe:
                        logger.debug(f"ParserError reading with {attempt_desc}: {pe}")
                        read_error = pe
                        if i == len(delimiters_to_try) - 1: # If it's the last fallback attempt
                             logger.error(f"Could not parse reference file '{os.path.basename(filepath)}' with any fallback delimiter after sniffing failed.")
                             return None
                    except Exception as e_fallback: # Catch other read errors during fallback
                        logger.exception(f"Unexpected error reading reference file '{filepath}' with {attempt_desc}: {e_fallback}")
                        read_error = e_fallback
                        if i == len(delimiters_to_try) - 1: return None
            except FileNotFoundError: # Should have been caught earlier, but safety check
                 logger.error(f"Reference file not found during read attempt: {filepath}")
                 return None
            except Exception as e_sniff_or_read: # Catch other errors during initial sniff/read
                logger.exception(f"Unexpected error reading reference file '{filepath}' (delimiter: {used_delimiter}): {e_sniff_or_read}")
                read_error = e_sniff_or_read
                return None

    except pd.errors.EmptyDataError:
        logger.error(f"Reference file '{os.path.basename(filepath)}' is empty or contains no data.")
        return None
    except pd.errors.ParserError as pe_user_delim:
        logger.error(f"ParserError reading reference file '{os.path.basename(filepath)}' with specified delimiter '{used_delimiter}': {pe_user_delim}")
        return None
    except Exception as e_main_read: # Catch other general read errors
        logger.exception(f"A critical error occurred while attempting to read reference file '{filepath}' (delimiter: {used_delimiter}): {e_main_read}")
        return None


    # If df is still None or empty after all attempts
    if df is None or df.empty:
        log_message = f"Failed to read or DataFrame is empty for reference file '{filepath}'."
        if read_error:
            log_message += f" Last error: {read_error}"
        logger.error(log_message)
        return None

    # --- Process the DataFrame (the rest of the function remains largely the same) ---
    try:
        # Normalize column names
        df.columns = [str(col).strip().lower() for col in df.columns]
        original_columns: List[str] = df.columns.tolist()

        # --- Identify Codon and Value columns ---
        codon_col: Optional[str] = None
        value_col_name: Optional[str] = None
        value_col_type: Optional[str] = None

        # Find Codon column
        for col_name_iter in original_columns: # Renamed col to col_name_iter to avoid conflict
            if 'codon' in col_name_iter:
                codon_col = col_name_iter
                break
        if not codon_col:
            logger.error("Could not find a 'Codon' column in reference file.")
            return None

        # Find Value column (prioritize RSCU)
        priority_order: Dict[str, List[str]] = {
             'rscu': ['rscu'],
             'per_thousand': ['frequency (per thousand)', 'frequency per thousand'],
             'freq': ['frequency', 'fraction', 'freq'],
             'count': ['count', 'number', 'total num', 'total']
        }
        for v_type, possible_names in priority_order.items():
             if value_col_name:
                break # Stop if already found
             for col_name_iter in original_columns: # Renamed col to col_name_iter
                col_norm = col_name_iter.replace('_', ' ').replace('-', ' ')
                for name in possible_names:
                    if name in col_norm:
                        # Avoid matching 'frequency' if a 'per_thousand' column exists and v_type is 'freq'
                        if v_type == 'freq' and name == 'frequency' and \
                        any(pt_name in c.replace('_', ' ').replace('-', ' ') for c in original_columns for pt_name in priority_order['per_thousand']):
                            continue
                        value_col_name = col_name_iter # Use col_name_iter
                        value_col_type = v_type
                        break
                if value_col_name:
                    break
        
        if not value_col_name or not value_col_type:
            logger.error(f"Could not find a suitable value column (e.g., RSCU, Frequency, Count) in columns: {original_columns}")
            return None

        logger.info(f"Identified Codon column: '{codon_col}', Value column: '{value_col_name}' (type: {value_col_type})")

        # Select, rename, and ensure Value is numeric
        ref_df: pd.DataFrame = df[[codon_col, value_col_name]].copy()
        ref_df.rename(columns={codon_col: 'Codon', value_col_name: 'Value'}, inplace=True)
        
        # Convert 'Value' to numeric, coercing errors and logging how many rows were affected
        initial_value_rows = len(ref_df)
        ref_df['Value'] = pd.to_numeric(ref_df['Value'], errors='coerce')
        rows_with_nan_value = ref_df['Value'].isnull().sum()
        
        if rows_with_nan_value > 0:
            logger.warning(
                f"Found {rows_with_nan_value} non-numeric entries in value column '{value_col_name}' "
                f"from reference file '{os.path.basename(filepath)}'. These rows will be dropped."
            )
            ref_df.dropna(subset=['Value'], inplace=True) # Drop rows where 'Value' became NaN

        if ref_df.empty:
            logger.error(
                f"No valid numeric data remaining in value column '{value_col_name}' "
                f"from reference file '{os.path.basename(filepath)}' after coercing to numeric."
            )
            return None


        # Normalize codons (string, uppercase, T instead of U)
        ref_df['Codon'] = ref_df['Codon'].astype(str).str.upper().str.replace('U', 'T')
        # Filter invalid codon formats (ensure 3 letters ATCG)
        # Also log how many were filtered
        initial_codon_rows = len(ref_df)
        ref_df = ref_df[ref_df['Codon'].str.match(r'^[ATCG]{3}$')]
        filtered_codon_rows = initial_codon_rows - len(ref_df)
        if filtered_codon_rows > 0:
            logger.warning(f"Filtered out {filtered_codon_rows} rows with invalid codon format from reference file.")


        # Map Amino Acids using provided genetic code
        ref_df['AminoAcid'] = ref_df['Codon'].map(genetic_code.get)
        
        # Log rows dropped due to unrecognized codons or stop codons
        initial_aa_rows = len(ref_df)
        ref_df = ref_df.dropna(subset=['AminoAcid']) # Keep only codons recognized by the code
        dropped_unrecognized_aa = initial_aa_rows - len(ref_df)
        if dropped_unrecognized_aa > 0:
            logger.warning(f"Dropped {dropped_unrecognized_aa} rows for codons not in the provided genetic code (ID: {genetic_code_id}).")

        initial_stop_rows = len(ref_df)
        ref_df = ref_df[ref_df['AminoAcid'] != '*'] # Exclude stop codons
        dropped_stop_codons = initial_stop_rows - len(ref_df)
        if dropped_stop_codons > 0:
            logger.info(f"Excluded {dropped_stop_codons} stop codon entries from reference data.")


        if ref_df.empty:
             logger.error("No valid coding codons found in reference file after filtering and mapping.")
             return None

        # --- Calculate Frequency column ---
        if value_col_type == 'count':
            total_count: float = ref_df['Value'].sum()
            ref_df['Frequency'] = ref_df['Value'] / total_count if total_count > 0 else 0.0
        elif value_col_type == 'freq':
             ref_df['Frequency'] = ref_df['Value']
        elif value_col_type == 'per_thousand':
             ref_df['Frequency'] = ref_df['Value'] / 1000.0
        else: # RSCU case or others
             ref_df['Frequency'] = np.nan
        # Ensure column exists
        if 'Frequency' not in ref_df.columns: ref_df['Frequency'] = np.nan

        # --- Get or Calculate RSCU column ---
        if value_col_type == 'rscu':
            logger.info("Using RSCU values directly from reference file.")
            ref_df['RSCU'] = ref_df['Value']
        elif value_col_type == 'count' or not ref_df['Frequency'].isnull().all():
             logger.info(f"Calculating RSCU values from reference file {value_col_type}...")
             # Use the analysis module's calculate_rscu (ensure it's importable)
             try:
                 from .analysis import calculate_rscu as calculate_rscu_analysis
                 # Prepare input for calculate_rscu: DataFrame with Codon index, 'Count' column
                 if value_col_type == 'count':
                     rscu_input_df = ref_df.set_index('Codon')[['Value']].rename(columns={'Value':'Count'})
                 else: # Calculate pseudo-counts from frequency for RSCU function if needed
                      # Avoid very small numbers causing issues; scale arbitrarily
                      pseudo_total = 1e6
                      ref_df['PseudoCount'] = (ref_df['Frequency'] * pseudo_total).round().astype(int)
                      rscu_input_df = ref_df.set_index('Codon')[['PseudoCount']].rename(columns={'PseudoCount':'Count'})

                 if not rscu_input_df.empty:
                      # Pass counts DataFrame and genetic code ID
                      temp_rscu_df = calculate_rscu_analysis(rscu_input_df, genetic_code_id=genetic_code_id)
                      # Map calculated RSCU values back
                      rscu_map: Dict[str, float] = temp_rscu_df.set_index('Codon')['RSCU'].to_dict()
                      ref_df['RSCU'] = ref_df['Codon'].map(rscu_map)
                 else:
                      ref_df['RSCU'] = np.nan
             except ImportError:
                  logger.error("Cannot import calculate_rscu from .analysis. Unable to calculate RSCU from reference counts/frequencies.")
                  ref_df['RSCU'] = np.nan
             except Exception as rscu_calc_err:
                  logger.exception(f"Error calculating RSCU from reference {value_col_type}: {rscu_calc_err}")
                  ref_df['RSCU'] = np.nan
        else: # Fallback if neither RSCU, Count, nor Frequency usable
             ref_df['RSCU'] = np.nan

        # Ensure RSCU column exists and is numeric
        if 'RSCU' not in ref_df.columns: ref_df['RSCU'] = np.nan
        ref_df['RSCU'] = pd.to_numeric(ref_df['RSCU'], errors='coerce')

        # --- Calculate CAI Weights (w_i = RSCU_i / max(RSCU_synonymous)) ---
        logger.info("Calculating CAI reference weights (w)...")
        ref_df['Weight'] = np.nan
        # Use drop_duplicates before setting index to handle potential redundant entries safely
        calc_df = ref_df.drop_duplicates(subset=['Codon']).set_index('Codon')
        valid_rscu_df = calc_df.dropna(subset=['AminoAcid', 'RSCU'])
        aa_groups = valid_rscu_df.groupby('AminoAcid')

        calculated_weights: Dict[str, float] = {}
        for aa, group in aa_groups:
            max_rscu: float = group['RSCU'].max()
            if pd.notna(max_rscu) and max_rscu > 1e-9:
                weights: pd.Series = group['RSCU'] / max_rscu
                calculated_weights.update(weights.to_dict())
            else:
                 # Handle single codon AAs (Met, Trp) or cases where max RSCU is invalid
                 # Assign weight 1.0 to all codons in such groups
                 for codon_index_iter in group.index: # Renamed codon to codon_index_iter
                     if codon_index_iter not in calculated_weights: # Avoid overwriting if duplicates existed
                          calculated_weights[codon_index_iter] = 1.0

        # Apply calculated weights and fill remaining NaNs (e.g., codons not in valid_rscu_df) with 1.0
        # Note: Should weights for codons missing from the reference be NaN or 1.0? Using 1.0 assumes neutral.
        ref_df['Weight'] = ref_df['Codon'].map(calculated_weights).fillna(1.0)
        logger.info("CAI weights calculated.")

        # Set Codon as index before returning
        ref_df_final = ref_df.set_index('Codon')

        # Select and return only necessary columns
        final_cols: List[str] = ['AminoAcid', 'Frequency', 'RSCU', 'Weight']
        missing_final_cols = [c for c in final_cols if c not in ref_df_final.columns]
        if missing_final_cols:
             # This indicates an internal logic error
             logger.error(f"Internal Error: Final columns missing after processing reference: {missing_final_cols}")
             return None # Return None if expected columns are missing

        logger.info(f"Successfully loaded and processed reference usage from: {os.path.basename(filepath)}")
        return ref_df_final[final_cols]

    except FileNotFoundError: # Should be caught earlier, but defensive check
        logger.error(f"Reference file disappeared during processing: '{filepath}'")
        return None
    except (ValueError, KeyError, AttributeError) as proc_err: # Catch specific processing errors
         logger.error(f"Error processing reference file '{os.path.basename(filepath)}': {proc_err}")
         return None
    except Exception as e: # Catch any other unexpected errors during processing
        logger.exception(f"Unexpected error processing reference file '{os.path.basename(filepath)}': {e}")
        return None


def clean_and_filter_sequences(
    sequences: List[SeqRecord],
    max_ambiguity_pct: float = 15.0
) -> List[SeqRecord]:
    """
    Cleans and filters a list of SeqRecord objects.

    Steps:
    1. Removes gaps ('-').
    2. Checks if gapless length is multiple of 3. Removes if not.
    3. Conditionally removes standard START ('ATG') and STOP ('TAA', 'TAG', 'TGA') codons.
    4. Replaces IUPAC ambiguous characters with 'N'.
    5. Filters based on maximum ambiguity percentage.

    Args:
        sequences (List[SeqRecord]): The input list of sequence records.
        max_ambiguity_pct (float): Maximum allowed percentage of 'N' characters (0-100). Default 15.0.

    Returns:
        List[SeqRecord]: A new list containing the cleaned and filtered SeqRecord objects.
    """
    cleaned_sequences: List[SeqRecord] = []
    initial_count: int = len(sequences)
    removed_count: int = 0
    logger.debug(f"Starting cleaning/filtering for {initial_count} sequences (max ambiguity: {max_ambiguity_pct}%)...")

    for record in sequences:
        try:
            # Ensure record has necessary attributes
            if not isinstance(record, SeqRecord) or not hasattr(record, 'seq') or not hasattr(record, 'id'):
                 logger.warning(f"Skipping invalid/incomplete record object: {record}")
                 removed_count += 1
                 continue

            seq_id: str = record.id
            original_seq_str: str = str(record.seq)

            # 1. Remove gaps
            gapless_seq: str = original_seq_str.replace('-', '')
            if not gapless_seq:
                logger.debug(f"Seq {seq_id} removed (empty after gap removal).")
                removed_count += 1
                continue

            # 2. Check length multiple of 3
            if len(gapless_seq) % 3 != 0:
                logger.debug(f"Seq {seq_id} removed (length {len(gapless_seq)} not multiple of 3 after gap removal).")
                removed_count += 1
                continue

            seq_to_process: str = gapless_seq
            len_before_trim: int = len(seq_to_process)

            # 3. Check and remove START codon (only if present at the very beginning)
            if len_before_trim >= 3 and seq_to_process.startswith(tuple(STANDARD_START_CODONS)):
                seq_to_process = seq_to_process[3:]
                logger.debug(f"Removed standard START codon from Seq {seq_id}")

            # 4. Check and remove STOP codon (only if present at the very end)
            if len(seq_to_process) >= 3 and seq_to_process.endswith(tuple(STANDARD_STOP_CODONS)):
                seq_to_process = seq_to_process[:-3]
                logger.debug(f"Removed standard STOP codon from Seq {seq_id}")

            # Check length again after potential trimming (must be >= 1 codon and multiple of 3)
            if not seq_to_process or len(seq_to_process) % 3 != 0:
                logger.debug(f"Seq {seq_id} removed (length {len(seq_to_process)} invalid after start/stop trim).")
                removed_count += 1
                continue

            # 5. Replace ambiguities with 'N'
            cleaned_cds_seq_str: str = seq_to_process.translate(AMBIGUOUS_TO_N_MAP)

            # 6. Calculate ambiguity percentage
            n_count: int = cleaned_cds_seq_str.count('N')
            seq_len: int = len(cleaned_cds_seq_str) # Should not be 0 here due to earlier checks
            ambiguity_pct: float = (n_count / seq_len) * 100 if seq_len > 0 else 0

            # 7. Filter based on ambiguity
            if ambiguity_pct > max_ambiguity_pct:
                logger.debug(f"Seq {seq_id} removed (ambiguity {ambiguity_pct:.1f}% > {max_ambiguity_pct}%).")
                removed_count += 1
                continue

            # If sequence passed all filters, create a new cleaned record
            # Keep original description but append indicator
            new_description = record.description + " [cleaned]" if record.description else "[cleaned]"
            cleaned_record = SeqRecord(
                Seq(cleaned_cds_seq_str),
                id=record.id,
                description=new_description,
                name=record.name # Preserve name attribute if present
            )
            cleaned_sequences.append(cleaned_record)

        except Exception as e:
             # Catch unexpected errors during processing of a single record
             record_id_str = getattr(record, 'id', 'UNKNOWN_ID') # Safely get ID
             logger.exception(f"Error cleaning/filtering record '{record_id_str}': {e}. Skipping record.")
             removed_count += 1
             continue # Skip to the next record

    # Log summary of removed sequences
    if removed_count > 0:
        logger.info(f"Removed {removed_count} out of {initial_count} sequences during cleaning/filtering.")
    else:
         logger.debug(f"All {initial_count} sequences passed cleaning/filtering.")

    return cleaned_sequences