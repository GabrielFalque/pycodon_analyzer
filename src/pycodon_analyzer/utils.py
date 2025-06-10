# src/pycodon_analyzer/utils.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Utility functions and constants for the pycodon_analyzer package.

This module provides essential functionality for the pycodon_analyzer package, including:
- Genetic code definitions and amino acid properties
- Functions for handling sequence data and file operations
- Utilities for loading and processing codon usage reference data
- Sequence cleaning and filtering capabilities

Most functions include robust error handling and logging to facilitate debugging.
"""
import os
import sys
import logging # <-- Import logging
import math
import csv
from typing import List, Dict, Optional, Tuple, Set, Union, Any
import re

# Third-party imports with dependency checks
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

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# CONSTANTS AND REFERENCE DATA
# -----------------------------------------------------------------------------
# Standard DNA Genetic Code (NCBI table 1)
# Maps each codon (triplet of nucleotides) to its corresponding amino acid (single letter code)
# '*' represents stop codons (TAA, TAG, TGA)
STANDARD_GENETIC_CODE: Dict[str, str] = {
    # Phenylalanine (F)
    'TTT': 'F', 'TTC': 'F',
    # Leucine (L)
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    # Serine (S)
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    # Tyrosine (Y)
    'TAT': 'Y', 'TAC': 'Y',
    # Stop codons (*)
    'TAA': '*', 'TAG': '*', 'TGA': '*',
    # Cysteine (C)
    'TGT': 'C', 'TGC': 'C',
    # Tryptophan (W)
    'TGG': 'W',
    # Proline (P)
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Histidine (H)
    'CAT': 'H', 'CAC': 'H',
    # Glutamine (Q)
    'CAA': 'Q', 'CAG': 'Q',
    # Arginine (R)
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Isoleucine (I)
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    # Methionine (M) - also the standard start codon
    'ATG': 'M',
    # Threonine (T)
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Asparagine (N)
    'AAT': 'N', 'AAC': 'N',
    # Lysine (K)
    'AAA': 'K', 'AAG': 'K',
    # Valine (V)
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    # Alanine (A)
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Aspartic Acid (D)
    'GAT': 'D', 'GAC': 'D',
    # Glutamic Acid (E)
    'GAA': 'E', 'GAG': 'E',
    # Glycine (G)
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# DNA sequence validation constants
VALID_DNA_CHARS: Set[str] = set('ATCGN-')  # Valid DNA characters including ambiguity code N and gaps
VALID_CODON_CHARS: Set[str] = set('ATCG')  # Characters allowed within a valid codon for counting

# Hydropathy Scale (Kyte & Doolittle, 1982)
# Reference: Kyte J, Doolittle RF. J Mol Biol. 1982;157(1):105-32
# Positive values indicate hydrophobic amino acids, negative values indicate hydrophilic amino acids
KYTE_DOOLITTLE_HYDROPATHY: Dict[str, float] = {
    # Hydrophobic amino acids (positive values)
    'I': 4.5,  # Isoleucine (most hydrophobic)
    'V': 4.2,  # Valine
    'L': 3.8,  # Leucine
    'F': 2.8,  # Phenylalanine
    'C': 2.5,  # Cysteine
    'M': 1.9,  # Methionine
    'A': 1.8,  # Alanine
    
    # Neutral to slightly hydrophobic/hydrophilic
    'G': -0.4,  # Glycine
    'T': -0.7,  # Threonine
    'S': -0.8,  # Serine
    'W': -0.9,  # Tryptophan
    'Y': -1.3,  # Tyrosine
    'P': -1.6,  # Proline
    
    # Hydrophilic amino acids (negative values)
    'H': -3.2,  # Histidine
    'E': -3.5,  # Glutamic Acid
    'Q': -3.5,  # Glutamine
    'D': -3.5,  # Aspartic Acid
    'N': -3.5,  # Asparagine
    'K': -3.9,  # Lysine
    'R': -4.5,  # Arginine (most hydrophilic)
}

# Default Human Codon Weights (Placeholder values)
# These are example values used as fallback if loading reference data fails.
# WARNING: These are NOT biologically validated for CAI calculations and should be replaced
# with actual values from reference datasets in production use.
DEFAULT_HUMAN_CAI_WEIGHTS: Dict[str, float] = {
    # Phenylalanine (F)
    'TTT': 0.45, 'TTC': 0.55,
    # Leucine (L) - partial example
    'TTA': 0.07, 'TTG': 0.13,
    # ... additional codons would be defined here in a complete implementation ...
}

# Ambiguity Handling
# IUPAC ambiguity codes for nucleotides:
# R = A or G (puRine)
# Y = C or T (pYrimidine)
# S = G or C (Strong bond)
# W = A or T (Weak bond)
# K = G or T (Keto)
# M = A or C (aMino)
# B = C, G, or T (not A)
# D = A, G, or T (not C)
# H = A, C, or T (not G)
# V = A, C, or G (not T)
# N = any base (A, C, G, or T)
AMBIGUOUS_DNA_LETTERS: str = 'RYSWKMBDHVN'
# Translation map to convert all ambiguous nucleotides to N for consistent handling
AMBIGUOUS_TO_N_MAP: Dict[int, int] = str.maketrans(AMBIGUOUS_DNA_LETTERS, 'N' * len(AMBIGUOUS_DNA_LETTERS))

# Standard genetic code start and stop codons
STANDARD_START_CODONS: Set[str] = {'ATG'}  # Methionine (M) - standard translation initiation codon
STANDARD_STOP_CODONS: Set[str] = {'TAA', 'TAG', 'TGA'}  # Termination codons (*, ochre, amber, opal)

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def sanitize_filename(name: Any) -> str:
    """
    Sanitizes a string to make it safe for use as a filename.
    
    This function performs the following operations:
    1. Converts non-string inputs to strings
    2. Removes brackets and parentheses
    3. Replaces whitespace, slashes, and colons with underscores
    4. Removes any other characters that might cause issues in filenames
    5. Strips leading/trailing periods, underscores, and hyphens
    6. Provides a fallback for empty results
    
    Args:
        name (Any): The input to sanitize (will be converted to string if not already)
        
    Returns:
        str: A sanitized string safe to use as a filename
        
    Examples:
        >>> sanitize_filename("My File (1).txt")
        'My_File_1.txt'
        >>> sanitize_filename("data/results:output")
        'data_results_output'
    """
    # Store original name for logging in case of errors
    original_name_for_log = name
    
    # Convert to string if not already
    if not isinstance(name, str):
        name = str(name)

    # Remove brackets and parentheses first, as they can interfere with other patterns
    name = re.sub(r'[\[\]()]', '', name)
    
    # Replace whitespace sequences, slashes, and colons with a single underscore
    name = re.sub(r'[\s/:]+', '_', name)
    
    # Remove any remaining problematic characters while preserving Unicode letters
    name = re.sub(r'[^\w.\-]+', '', name, flags=re.UNICODE)

    # Remove leading/trailing problematic characters
    name = name.strip('._-')
    
    # Return sanitized name or fallback if empty
    if name:
        return name
    else:
        logger.warning(f"Sanitization resulted in an empty or invalid name from input: '{original_name_for_log}'. Using fallback.")
        return "_invalid_name_"

def get_genetic_code(code_id: int = 1) -> Dict[str, str]:
    """
    Returns a dictionary representing a genetic code mapping codons to amino acids.
    
    This function provides access to genetic code tables, which map nucleotide
    triplets (codons) to their corresponding amino acids. Currently, only the
    standard genetic code (NCBI table ID 1) is implemented, but the function
    is designed to be extended to support additional genetic codes in the future.
    
    The standard genetic code is used by most organisms, but some organisms
    (particularly in mitochondria) use variant genetic codes.
    
    Args:
        code_id (int): The NCBI genetic code ID number. Currently only 1 (standard code)
                      is supported. See https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi
                      for a list of all NCBI genetic code IDs.

    Returns:
        Dict[str, str]: A dictionary mapping codons (3-letter strings) to amino acids
                       (1-letter codes) or '*' for stop codons. For example:
                       {'ATG': 'M', 'TAA': '*', 'GGC': 'G', ...}

    Raises:
        NotImplementedError: If a genetic code ID other than 1 is requested.
        
    Example:
        >>> code = get_genetic_code()
        >>> code['ATG']  # Methionine
        'M'
        >>> code['TAA']  # Stop codon
        '*'
    """
    if code_id == 1:
        return STANDARD_GENETIC_CODE.copy() # Return a copy to prevent modification
    else:
        # Log before raising might be useful if more codes were planned
        logger.error(f"Genetic code ID {code_id} is not implemented yet.")
        raise NotImplementedError(f"Genetic code ID {code_id} is not implemented yet.")

def get_synonymous_codons(genetic_code: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Groups codons by the amino acid they encode to identify synonymous codons.
    
    Synonymous codons are different nucleotide triplets that encode the same amino acid.
    This function reverses the genetic code mapping to group codons by their amino acid.
    
    Args:
        genetic_code (Dict[str, str]): A dictionary mapping codons to amino acids.
                                      Example: {'ATG': 'M', 'TGG': 'W', 'TTT': 'F', 'TTC': 'F'}

    Returns:
        Dict[str, List[str]]: A dictionary where keys are amino acids (or '*' for stop codons) and
                              values are lists of codons encoding that amino acid.
                              Example: {'M': ['ATG'], 'W': ['TGG'], 'F': ['TTT', 'TTC']}
    
    Examples:
        >>> code = {'ATG': 'M', 'TGG': 'W', 'TTT': 'F', 'TTC': 'F'}
        >>> get_synonymous_codons(code)
        {'M': ['ATG'], 'W': ['TGG'], 'F': ['TTT', 'TTC']}
    """
    # Initialize empty dictionary for results
    syn_codons: Dict[str, List[str]] = {}
    
    # Handle empty input
    if not genetic_code:
        logger.warning("get_synonymous_codons called with an empty genetic code dictionary.")
        return syn_codons

    try:
        # Group codons by amino acid
        for codon, aa in genetic_code.items():
            if aa not in syn_codons:
                syn_codons[aa] = []
            syn_codons[aa].append(codon)
    except AttributeError:
        # Handle case where input is not a dictionary
        logger.error("Invalid genetic_code dictionary passed to get_synonymous_codons (expected dict).")
        return {}  # Return empty on bad input type
        
    return syn_codons


# -----------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

def load_reference_usage(
    filepath: str,
    genetic_code: Dict[str, str],
    genetic_code_id: int = 1,
    delimiter: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Loads and processes a reference codon usage table from a file.
    
    This function reads codon usage data from a CSV/TSV file and processes it to create
    a standardized DataFrame with codon usage statistics. It performs extensive validation,
    normalization, and calculation of derived metrics.
    
    The function is designed to be robust against various file formats and will:
    - Auto-detect delimiters if not specified
    - Identify relevant columns regardless of exact naming
    - Normalize codon formats (uppercase, T instead of U)
    - Filter invalid codons
    - Map codons to amino acids using the provided genetic code
    - Calculate frequencies, RSCU values, and CAI weights
    
    Input file requirements:
    - Must contain a column with codon sequences (will be identified by name containing 'codon')
    - Must contain at least one of these value columns:
      * RSCU (Relative Synonymous Codon Usage)
      * Frequency (proportion between 0-1)
      * Frequency per thousand (values typically between 0-1000)
      * Count (absolute codon counts)
    
    Args:
        filepath (str): Path to the reference codon usage file
        genetic_code (Dict[str, str]): Dictionary mapping codons to amino acids
        genetic_code_id (int): NCBI genetic code ID number (default: 1 for standard code)
        delimiter (Optional[str]): File delimiter character. If None (default),
                                  the function will attempt to auto-detect it
    
    Returns:
        Optional[pd.DataFrame]: DataFrame with standardized codon usage data:
            - Index: Codon (str) - 3-letter nucleotide sequences
            - Columns:
                * AminoAcid (str): Single-letter amino acid code
                * Frequency (float): Normalized frequency (0-1)
                * RSCU (float): Relative Synonymous Codon Usage values
                * Weight (float): CAI weights (w) calculated as RSCU/max(RSCU) for each amino acid
            
            Returns None if the file cannot be loaded or processed correctly.
    
    Example:
        >>> genetic_code = get_genetic_code()
        >>> usage_df = load_reference_usage("human_codon_usage.csv", genetic_code)
        >>> usage_df.loc["ATG"]
        AminoAcid     M
        Frequency     0.23
        RSCU          1.0
        Weight        1.0
        Name: ATG, dtype: object
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
    Cleans and filters a list of DNA sequence records for codon analysis.
    
    This function performs several cleaning and validation steps to prepare DNA sequences
    for codon usage analysis. Each sequence is processed independently, and sequences that
    don't meet the criteria are filtered out.
    
    Processing steps:
    1. Validates each record is a proper SeqRecord object with a seq attribute
    2. Removes all whitespace and gap characters ('-')
    3. Verifies the gapless sequence length is a multiple of 3 (complete codons)
    4. Optionally removes standard START codon ('ATG') from the beginning
    5. Optionally removes standard STOP codons ('TAA', 'TAG', 'TGA') from the end
    6. Replaces all IUPAC ambiguous nucleotide characters with 'N'
    7. Filters out sequences with ambiguity percentage exceeding the threshold
    8. Creates new SeqRecord objects with cleaned sequences
    
    The function includes robust error handling and detailed logging at each step.
    
    Args:
        sequences (List[SeqRecord]): The input list of Biopython SeqRecord objects to process.
        max_ambiguity_pct (float): Maximum allowed percentage of ambiguous 'N' characters
                                  in a sequence (0-100). Default: 15.0%.
    
    Returns:
        List[SeqRecord]: A new list containing only the cleaned and filtered SeqRecord objects.
                        Each returned record has "[cleaned]" appended to its description.
    
    Example:
        >>> from Bio.SeqRecord import SeqRecord
        >>> from Bio.Seq import Seq
        >>> records = [SeqRecord(Seq("ATG-ACGTACGTA-TAG"), id="seq1")]
        >>> cleaned = clean_and_filter_sequences(records)
        >>> print(cleaned[0].seq)
        ACGTACGTA
    """
    cleaned_sequences: List[SeqRecord] = []
    initial_count: int = len(sequences)
    removed_count: int = 0
    logger.debug(f"Starting cleaning/filtering for {initial_count} sequences (max ambiguity: {max_ambiguity_pct}%)...")

    for record_idx, record in enumerate(sequences): # Added enumerate for better logging in case of ID issues
        # Robust ID handling for logging from the start of the loop
        seq_id_for_log: str
        original_id_attr = getattr(record, 'id', f"RECORD_AT_INDEX_{record_idx}")
        if original_id_attr is None:
            seq_id_for_log = f"UNKNOWN_ID_AT_INDEX_{record_idx}"
        else:
            seq_id_for_log = str(original_id_attr)

        current_processing_stage = "start"
        current_seq_state = ""
        original_seq_str_for_log = ""


        try:
            if not isinstance(record, SeqRecord) or not hasattr(record, 'seq'):
                logger.warning(f"Skipping invalid/incomplete record object: {record} (ID for log: {seq_id_for_log})")
                removed_count += 1
                continue

            original_seq_str: str = str(record.seq)
            original_seq_str_for_log = original_seq_str # For logging in case of error

            current_processing_stage = "after_str_conversion"
            current_seq_state = original_seq_str

            # STEP 1: Remove all whitespace and gap characters
            # This ensures we have a continuous sequence of nucleotides
            gapless_seq: str = re.sub(r'\s|-', '', original_seq_str)
            current_processing_stage = "after_gap_removal"
            current_seq_state = gapless_seq

            # Check if sequence is empty after gap removal
            if not gapless_seq:
                logger.debug(f"Seq {seq_id_for_log} removed (empty after gap removal). Original: '{original_seq_str_for_log}'")
                removed_count += 1
                continue
            
            # STEP 2: Verify sequence length is a multiple of 3 (complete codons)
            # This is essential for proper codon analysis
            if len(gapless_seq) % 3 != 0:
                logger.debug(f"Seq {seq_id_for_log} removed (length {len(gapless_seq)} not multiple of 3 after gap removal). Original: '{original_seq_str_for_log}'")
                removed_count += 1
                continue

            # Prepare for start/stop codon trimming
            seq_to_process: str = gapless_seq
            len_before_trim: int = len(seq_to_process)
            
            current_processing_stage = "before_trim"
            current_seq_state = seq_to_process

            # STEP 3a: Remove START codon if present at beginning
            if len_before_trim >= 3 and seq_to_process.startswith(tuple(STANDARD_START_CODONS)):
                seq_to_process = seq_to_process[3:]  # Remove first 3 nucleotides (ATG)
                logger.debug(f"Removed standard START codon from Seq {seq_id_for_log}. Sequence now: '{seq_to_process}'")

            # STEP 3b: Remove STOP codon if present at end
            if len(seq_to_process) >= 3 and seq_to_process.endswith(tuple(STANDARD_STOP_CODONS)):
                seq_to_process = seq_to_process[:-3]  # Remove last 3 nucleotides (stop codon)
                logger.debug(f"Removed standard STOP codon from Seq {seq_id_for_log}. Sequence now: '{seq_to_process}'")
            
            current_processing_stage = "after_trim"
            current_seq_state = seq_to_process

            # Verify sequence is still valid after trimming (not empty and multiple of 3)
            if not seq_to_process or len(seq_to_process) % 3 != 0:
                logger.debug(f"Seq {seq_id_for_log} removed (length {len(seq_to_process)} invalid after start/stop trim). Original: '{original_seq_str_for_log}', Gapless: '{gapless_seq}', After trim: '{seq_to_process}'")
                removed_count += 1
                continue

            # STEP 4: Replace ambiguous nucleotides with 'N'
            # This uses the translation map defined in AMBIGUOUS_TO_N_MAP
            cleaned_cds_seq_str: str = seq_to_process.translate(AMBIGUOUS_TO_N_MAP)
            current_processing_stage = "after_ambiguity_replacement"
            current_seq_state = cleaned_cds_seq_str

            # STEP 5: Calculate ambiguity percentage and filter if too high
            n_count: int = cleaned_cds_seq_str.count('N')
            seq_len: int = len(cleaned_cds_seq_str)
            ambiguity_pct: float = (n_count / seq_len) * 100 if seq_len > 0 else 0

            # Filter out sequences with too many ambiguous nucleotides
            if ambiguity_pct > max_ambiguity_pct:
                logger.debug(f"Seq {seq_id_for_log} removed (ambiguity {ambiguity_pct:.1f}% > {max_ambiguity_pct}%). Final CDS candidate: '{cleaned_cds_seq_str}'. Original: '{original_seq_str_for_log}'")
                removed_count += 1
                continue

            # STEP 6: Create new SeqRecord with cleaned sequence
            # Append "[cleaned]" to description to indicate processing
            new_description = record.description + " [cleaned]" if record.description else "[cleaned]"
            
            # Use original ID if available, otherwise generate a new one
            final_id_for_record = original_id_attr if original_id_attr is not None and original_id_attr != f"RECORD_AT_INDEX_{record_idx}" else f"PROCESSED_AUTO_ID_{record_idx}"

            # Create new SeqRecord with cleaned sequence and add to results list
            cleaned_record = SeqRecord(
                Seq(cleaned_cds_seq_str),
                id=final_id_for_record,
                description=new_description,
                name=record.name
            )
            cleaned_sequences.append(cleaned_record)

        except Exception as e:
             current_state_info = f"Last good state: stage='{current_processing_stage}', seq='{current_seq_state[:30]}...'"
             logger.exception(f"Error cleaning/filtering record (log ID: {seq_id_for_log}, original attribute: {getattr(record, 'id', 'N/A')}): {e}. {current_state_info} Original: '{original_seq_str_for_log}'")
             removed_count += 1
             continue

    if removed_count > 0:
        logger.info(f"Removed {removed_count} out of {initial_count} sequences during cleaning/filtering.")
    else:
         logger.debug(f"All {initial_count} sequences passed cleaning/filtering.")

    return cleaned_sequences