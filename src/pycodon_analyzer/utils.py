# src/pycodonanalyzer/utils.py

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Utility functions and constants for the pycodon_analyzer package.
"""
import pandas as pd # type: ignore
import os
import sys
import numpy as np # type: ignore
import math # Keep math if potentially needed elsewhere, though not directly used here now
from Bio.SeqRecord import SeqRecord # type: ignore
from Bio.Seq import Seq # type: ignore

# --- Constants ---
# Standard DNA Genetic Code (NCBI table 1)
# Maps codons (keys) to amino acids (values) or '*' for stop codons.
STANDARD_GENETIC_CODE = {
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
VALID_DNA_CHARS = set('ATCGN-')
VALID_CODON_CHARS = set('ATCG') # Characters allowed within a valid codon for counting

# --- Hydropathy Scale (Kyte & Doolittle, 1982) ---
# Dictionary mapping single-letter amino acid codes to hydropathy values
KYTE_DOOLITTLE_HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    # U (Selenocysteine) and O (Pyrrolysine) are often ignored.
}

# --- Default Human Codon Weights (Placeholder - Only used if loading fails) ---
# These are example values and NOT biologically validated for CAI.
DEFAULT_HUMAN_CAI_WEIGHTS = {
    'TTT': 0.45, 'TTC': 0.55, 'TTA': 0.07, 'TTG': 0.13,
    # ... only a few examples ...
}

# --- Ambiguity Handling ---
AMBIGUOUS_DNA_LETTERS = 'RYSWKMBDHVN' # Include N here for replacement logic
AMBIGUOUS_TO_N_MAP = str.maketrans(AMBIGUOUS_DNA_LETTERS, 'N' * len(AMBIGUOUS_DNA_LETTERS))


# --- Define standard start/stop codons ---
STANDARD_START_CODONS = {'ATG'}
STANDARD_STOP_CODONS = {'TAA', 'TAG', 'TGA'}

# --- Functions ---

def get_genetic_code(code_id=1):
    """
    Returns a dictionary representing a genetic code.
    Currently only supports the standard code (ID=1).

    Args:
        code_id (int): The NCBI genetic code ID (default: 1).

    Returns:
        dict: A dictionary mapping codons to amino acids or '*'.

    Raises:
        NotImplementedError: If a code other than 1 is requested.
    """
    if code_id == 1:
        return STANDARD_GENETIC_CODE
    else:
        raise NotImplementedError(f"Genetic code ID {code_id} is not implemented yet.")

def get_synonymous_codons(genetic_code):
    """
    Groups codons by the amino acid they encode.

    Args:
        genetic_code (dict): A dictionary mapping codons to amino acids.

    Returns:
        dict: A dictionary where keys are amino acids (or '*') and
              values are lists of codons encoding that amino acid.
    """
    syn_codons = {}
    for codon, aa in genetic_code.items():
        if aa not in syn_codons:
            syn_codons[aa] = []
        syn_codons[aa].append(codon)
    return syn_codons


def load_reference_usage(filepath, genetic_code, genetic_code_id=1):
    """
    Loads a reference codon usage table (e.g., for CAI weights or comparison).
    Expects a CSV or TSV file with columns for Codon and one of:
    Frequency, Count, RSCU, or 'Frequency (per thousand)'.
    Infers delimiter. Calculates RSCU and CAI weights (w).

    Args:
        filepath (str): Path to the reference file.
        genetic_code (dict): Genetic code mapping dictionary.
        genetic_code_id (int): The ID of the genetic code being used (default: 1).

    Returns:
        pd.DataFrame or None: DataFrame with Codon as index and columns
                              ['AminoAcid', 'Frequency', 'RSCU', 'Weight'].
                              Returns None if file cannot be loaded or parsed.
    """
    if not os.path.isfile(filepath):
        print(f"Warning: Reference file not found: {filepath}", file=sys.stderr)
        return None

    df = None # Initialize df
    try:
        # Try reading with common delimiters
        try:
             # Try tab first, as it's common in bioinformatics and the default human file
             df = pd.read_csv(filepath, sep='\t', engine='python', comment='#')
             # Simple check if tab worked reasonably well (more than 1 column)
             if len(df.columns) <= 1:
                  print("Warning: Tab delimiter read failed or resulted in <=1 column, trying comma...", file=sys.stderr)
                  df = pd.read_csv(filepath, sep=',', engine='python', comment='#')
        except pd.errors.ParserError as e_tab:
             print(f"Warning: Failed reading with tab delimiter ({e_tab}), trying comma...", file=sys.stderr)
             try:
                  df = pd.read_csv(filepath, sep=',', engine='python', comment='#')
             except pd.errors.ParserError as e_comma:
                  print(f"Warning: Failed reading with comma delimiter ({e_comma}), trying sep=None...", file=sys.stderr)
                  # Last resort: sep=None
                  try:
                       df = pd.read_csv(filepath, sep=None, engine='python', comment='#')
                  except pd.errors.ParserError as e_none:
                        raise ValueError(f"Could not parse reference file with tab, comma, or auto-detection. Please check format. Error: {e_none}")
        except Exception as read_e: # Catch other potential reading errors
             raise ValueError(f"Error reading reference file '{filepath}': {read_e}")

        if df is None or df.empty:
            raise ValueError(f"Failed to read or DataFrame is empty for reference file '{filepath}'.")


        # --- Process the DataFrame ---
        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        original_columns = df.columns.tolist()

        # --- Identify Codon and Value columns (Improved Logic) ---
        codon_col = None
        value_col_name = None # Store the name of the identified value column
        value_col_type = None # Store type: 'freq', 'count', 'rscu', 'per_thousand'

        # Find Codon column
        for col in original_columns:
            if 'codon' in col: # Simple check, assumes column name contains 'codon'
                codon_col = col
                break
        if not codon_col: raise ValueError("Could not find a 'Codon' column in reference file.")

        # Find Value column, PRIORITIZING RSCU
        priority_order = {
             # Try RSCU first!
             'rscu': ['rscu'],
             # Then other types if RSCU column wasn't found
             'per_thousand': ['frequency (per thousand)', 'frequency per thousand'],
             'freq': ['frequency', 'fraction'],
             'count': ['count', 'number', 'total num', 'total'],
             }


        for v_type, possible_names in priority_order.items():
             for col in original_columns:
                 col_norm = col.replace('_', ' ').replace('-', ' ')
                 for name in possible_names:
                      if name in col_norm:
                           # Avoid matching 'frequency' when 'per_thousand' is desired later if RSCU isn't found
                           if v_type == 'freq' and name == 'frequency' and \
                              any(pt_name in col_norm for pt_name in priority_order['per_thousand']):
                                continue
                           value_col_name = col
                           value_col_type = v_type
                           break # Found match for this type name
                 if value_col_name: break # Found match for this type priority
             if value_col_name: break # Found value column, stop searching priority


        if not value_col_name:
            raise ValueError(f"Could not find a suitable value column "
                             f"(e.g., 'RSCU','Frequency', 'Count', 'Frequency (per thousand)', 'Fraction', 'Number') "
                             f"in columns: {original_columns}")

        print(f"  Identified Codon column: '{codon_col}', Value column: '{value_col_name}' (type: {value_col_type})")

        # Select and rename essential columns
        ref_df = df[[codon_col, value_col_name]].copy()
        ref_df.rename(columns={codon_col: 'Codon', value_col_name: 'Value'}, inplace=True)

        # Ensure 'Value' column is numeric, coerce errors to NaN
        ref_df['Value'] = pd.to_numeric(ref_df['Value'], errors='coerce')
        # Drop rows where Value could not be converted
        initial_rows = len(ref_df)
        ref_df.dropna(subset=['Value'], inplace=True)
        if len(ref_df) < initial_rows:
             print(f"Warning: Dropped {initial_rows - len(ref_df)} rows from reference file due to non-numeric values in column '{value_col_name}'.", file=sys.stderr)

        # Normalize codons
        ref_df['Codon'] = ref_df['Codon'].astype(str).str.upper().str.replace('U', 'T')
        # Drop rows with invalid codon format if any slipped through (e.g., non-triplet)
        ref_df = ref_df[ref_df['Codon'].str.match(r'^[ATCG]{3}$')]


        # Map Amino Acids
        ref_df['AminoAcid'] = ref_df['Codon'].map(genetic_code.get)
        ref_df = ref_df.dropna(subset=['AminoAcid']) # Keep only valid codons recognised by the genetic code
        # Exclude stop codons explicitly using the AA symbol '*'
        ref_df = ref_df[ref_df['AminoAcid'] != '*']

        if ref_df.empty:
             raise ValueError("No valid coding codons found in reference file after filtering.")

        # --- Calculate Frequency column ---
        if value_col_type == 'count':
            total_count = ref_df['Value'].sum()
            ref_df['Frequency'] = ref_df['Value'] / total_count if total_count > 0 else 0.0
        elif value_col_type == 'freq':
             ref_df['Frequency'] = ref_df['Value']
        elif value_col_type == 'per_thousand':
             ref_df['Frequency'] = ref_df['Value'] / 1000.0
        else: # RSCU case or others where frequency isn't directly available
             ref_df['Frequency'] = np.nan

        # Ensure Frequency column exists even if it's NaN
        if 'Frequency' not in ref_df.columns:
             ref_df['Frequency'] = np.nan

        # --- Get RSCU column (Use directly if possible) ---
        if value_col_type == 'rscu':
            print("  Using RSCU values directly from reference file.")
            ref_df['RSCU'] = ref_df['Value']
        elif value_col_type == 'count':
             # If input was 'count', we still need to calculate RSCU
             print("  Calculating RSCU values from reference file counts...")
             rscu_input_df = ref_df.set_index('Codon')[['Value']].rename(columns={'Value':'Count'})
             if not rscu_input_df.empty:
                  from .analysis import calculate_rscu # Assuming this function is correct
                  temp_rscu_df = calculate_rscu(rscu_input_df, genetic_code_id=genetic_code_id)
                  # Use map to avoid merge/index issues
                  rscu_map = temp_rscu_df.set_index('Codon')['RSCU'].to_dict()
                  ref_df['RSCU'] = ref_df['Codon'].map(rscu_map)
             else:
                  ref_df['RSCU'] = np.nan
        elif not ref_df['Frequency'].isnull().all(): # value_col_type is 'freq' or 'per_thousand'
             # Calculate RSCU directly from frequencies
             print("  Calculating RSCU values directly from reference file frequencies...")
             rscu_values = {}
             try: # Added try/except for get_synonymous_codons
                 syn_codons = get_synonymous_codons(genetic_code)
             except Exception as e:
                 print(f"Error getting synonymous codons: {e}", file=sys.stderr)
                 syn_codons = {} # Fallback

             # Group by AminoAcid using the already mapped 'AminoAcid' column
             aa_freq_groups = ref_df.groupby('AminoAcid')

             for aa, group_df in aa_freq_groups:
                 # Use the list of synonymous codons from the full genetic code
                 syn_list = syn_codons.get(aa, [])
                 num_syn_codons = len(syn_list)

                 if aa == '*' or num_syn_codons <= 1: # Skip stops and single-codon families
                     for codon in group_df['Codon']: rscu_values[codon] = np.nan
                     continue

                 # Sum frequencies for this AA *present in the file*
                 total_aa_freq_in_file = group_df['Frequency'].sum()

                 if total_aa_freq_in_file < 1e-9: # If this AA has effectively zero frequency in the file
                     for codon in group_df['Codon']: rscu_values[codon] = 0.0
                     continue

                 # Calculate RSCU for each codon found in the file for this AA
                 # Formula: RSCU = (ObservedCodonFrequency / SumSynonymousFrequencies) * NumberOfSynonymousCodons
                 for idx, row in group_df.iterrows():
                     codon = row['Codon']
                     observed_freq = row['Frequency']
                     # Robust calculation even if sum of frequencies != 1
                     rscu = (observed_freq * num_syn_codons) / total_aa_freq_in_file if total_aa_freq_in_file > 1e-9 else 0.0
                     rscu_values[codon] = rscu

             # Map calculated RSCU values back to the main DataFrame
             ref_df['RSCU'] = ref_df['Codon'].map(rscu_values)
        else:
             # Fallback if we have neither valid RSCU, Count, nor Frequency
             ref_df['RSCU'] = np.nan

        # --- Ensure RSCU column exists ---
        if 'RSCU' not in ref_df.columns:
            ref_df['RSCU'] = np.nan
        # Ensure RSCU is numeric before weight calculation
        ref_df['RSCU'] = pd.to_numeric(ref_df['RSCU'], errors='coerce')

        # --- Calculate CAI Weights (w_i = RSCU_i / max(RSCU_synonymous)) ---
        # This part should now receive correct RSCU values
        ref_df['Weight'] = np.nan
        # Work on a copy with 'Codon' as index for weight calculation
        # Need to handle potential duplicate Codon entries before setting index if input is messy,
        # but assuming input file has unique codons after initial filtering.
        calc_df = ref_df.drop_duplicates(subset=['Codon']).set_index('Codon') # Ensure unique index
        valid_rscu_df = calc_df.dropna(subset=['AminoAcid', 'RSCU'])
        aa_groups = valid_rscu_df.groupby('AminoAcid')

        calculated_weights = {} # Store calculated weights
        for aa, group in aa_groups:
            max_rscu = group['RSCU'].max()
            if pd.notna(max_rscu) and max_rscu > 1e-9: # Check max RSCU is valid and > 0
                weights = group['RSCU'] / max_rscu
                calculated_weights.update(weights.to_dict())
            else:
                 # Handle cases where max RSCU is 0 or NaN (includes single codon AA)
                 num_syn_codons = len(group)
                 # Weight=1.0 for single codons (Met/Trp) or if maxRSCU invalid
                 weight_val = 1.0
                 for codon in group.index:
                     # Only assign weight if not already calculated (safety for potential duplicates)
                     if codon not in calculated_weights:
                          calculated_weights[codon] = weight_val

        # Apply calculated weights and fill remaining NaNs with 1.0 (safety for single codons missed)
        ref_df['Weight'] = ref_df['Codon'].map(calculated_weights).fillna(1.0)

        print(f"Successfully loaded and processed reference usage from: {filepath}")

        # Set Codon as index before returning
        if 'Codon' in ref_df.columns:
            ref_df.set_index('Codon', inplace=True)
        elif not isinstance(ref_df.index, pd.Index) or ref_df.index.name != 'Codon':
             # Fallback if index wasn't set or column got lost - this shouldn't happen
             print("Warning: Could not set Codon index before returning reference data.", file=sys.stderr)
             # Attempt to find and set index again if possible, otherwise return might fail selection
             if 'Codon' in ref_df.columns: ref_df.set_index('Codon', inplace=True)

        # Select and return only necessary columns
        final_cols = ['AminoAcid', 'Frequency', 'RSCU', 'Weight']
        missing_final_cols = [c for c in final_cols if c not in ref_df.columns]
        if missing_final_cols:
             raise ValueError(f"Internal Error: Final columns missing after processing reference: {missing_final_cols}")

        return ref_df[final_cols]

    except FileNotFoundError: # Catch specific error if file not found
        print(f"Error: Reference file not found at '{filepath}'", file=sys.stderr)
        return None
    except ValueError as ve: # Catch value errors raised during processing
         print(f"Error processing reference file '{filepath}': {ve}", file=sys.stderr)
         return None
    except Exception as e: # Catch any other unexpected errors
        print(f"Unexpected error loading or processing reference file '{filepath}': {e}", file=sys.stderr)
        # import traceback # Uncomment for detailed debugging
        # traceback.print_exc()
        return None
    
def clean_and_filter_sequences(sequences: list[SeqRecord],
                               max_ambiguity_pct: float = 15.0) -> list[SeqRecord]:
    """
    Cleans and filters a list of SeqRecord objects based on user requirements.

    Steps:
    1. Removes gaps ('-').
    2. Checks if gapless length is a multiple of 3. Removes sequence if not.
    3. Removes the first codon ONLY IF it's a standard START codon ('ATG').
    4. Removes the last codon ONLY IF it's a standard STOP codon ('TAA', 'TAG', 'TGA').
       Removes sequence if it becomes too short (< 1 codon remaining) after trimming.
    5. Replaces all IUPAC ambiguous DNA characters (R,Y,S,W,K,M,B,D,H,V) with 'N'.
    6. Calculates the percentage of 'N's (ambiguity).
    7. Removes sequence if ambiguity percentage exceeds max_ambiguity_pct.

    Args:
        sequences (list[SeqRecord]): The input list of sequence records.
        max_ambiguity_pct (float): Maximum allowed percentage of 'N' characters
                                   (0 to 100). Default is 15.0.

    Returns:
        list[SeqRecord]: A new list containing the cleaned and filtered SeqRecord objects
                         with modified sequences.
    """
    cleaned_sequences = []
    initial_count = len(sequences)
    removed_count = 0

    for record in sequences:
        original_seq_str = str(record.seq)
        seq_id = record.id

        # 1. Remove gaps
        gapless_seq = original_seq_str.replace('-', '')
        if not gapless_seq:
            removed_count += 1
            continue

        # 2. Check length multiple of 3
        if len(gapless_seq) % 3 != 0:
            # print(f"    Debug: Seq {seq_id} removed (length {len(gapless_seq)} not multiple of 3 after gap removal).")
            removed_count += 1
            continue

        seq_to_process = gapless_seq
        original_len_for_check = len(seq_to_process)

        # 3. Check and remove START codon
        if original_len_for_check >= 3 and seq_to_process.startswith(tuple(STANDARD_START_CODONS)):
            seq_to_process = seq_to_process[3:]
            # print(f"    Debug: Removed START from {seq_id}")

        # 4. Check and remove STOP codon
        if len(seq_to_process) >= 3 and seq_to_process.endswith(tuple(STANDARD_STOP_CODONS)):
            seq_to_process = seq_to_process[:-3]
            # print(f"    Debug: Removed STOP from {seq_id}")

        # Check if sequence became too short or invalid length after trimming
        if not seq_to_process or len(seq_to_process) % 3 != 0:
            # print(f"    Debug: Seq {seq_id} removed (length {len(seq_to_process)} issue after start/stop check/removal).")
            removed_count += 1
            continue

        # 5. Replace ambiguities with 'N'
        cleaned_cds_seq_str = seq_to_process.translate(AMBIGUOUS_TO_N_MAP)

        # 6. Calculate ambiguity percentage
        n_count = cleaned_cds_seq_str.count('N')
        seq_len = len(cleaned_cds_seq_str)
        ambiguity_pct = (n_count / seq_len) * 100 if seq_len > 0 else 0

        # 7. Filter based on ambiguity
        if ambiguity_pct > max_ambiguity_pct:
            # print(f"    Debug: Seq {seq_id} removed (ambiguity {ambiguity_pct:.1f}% > {max_ambiguity_pct}%).")
            removed_count += 1
            continue

        # If sequence passed all filters, create a new record
        cleaned_record = SeqRecord(
            Seq(cleaned_cds_seq_str),
            id=record.id,
            description=record.description + " [cleaned]"
        )
        cleaned_sequences.append(cleaned_record)

    if removed_count > 0:
        print(f"    Note: Removed {removed_count} out of {initial_count} sequences during cleaning/filtering.")

    return cleaned_sequences