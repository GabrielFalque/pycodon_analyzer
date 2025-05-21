# src/pycodon_analyzer/analysis.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Core analysis functions for codon usage and sequence properties.
"""
import itertools
import os
import sys
import logging # <-- Import logging
import math # For log, exp in CAI, etc.
from collections import Counter
from typing import List, Dict, Optional, Set, Tuple, Any, Union, TYPE_CHECKING

import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.SeqUtils import GC123, molecular_weight # type: ignore # Use BioPython's GC utils
from Bio.SeqUtils.ProtParam import ProteinAnalysis # type: ignore # For GRAVY, Aromaticity

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

# Import scipy.stats - handle optional dependency for typing
if TYPE_CHECKING:
    from scipy import stats as scipy_stats_module # Import only for type checking
    ScipyStatsModule = Any # Could be more specific, e.g. type(scipy.stats)
                           # but 'Any' is often sufficient for module-level objects
                           # or define a Protocol if specific functions are consistently used.
else:
    ScipyStatsModule = Any # Fallback type for runtime
    try:
        from scipy import stats as scipy_stats_module
    except ImportError:
        scipy_stats_module = None # Set to None if not found

from functools import partial

# --- Local modules ---
try:
    from .utils import (get_genetic_code, get_synonymous_codons,
                        VALID_CODON_CHARS, KYTE_DOOLITTLE_HYDROPATHY)
except ImportError as e:
    # Use logger here if possible, but it might not be configured yet
    print(f"ERROR: Failed importing from .utils: {e}. Check package structure/installation.", file=sys.stderr)
    # Define minimal fallbacks if needed for the script to load, though functionality will be broken
    STANDARD_GENETIC_CODE: Dict[str, str] = {}
    VALID_CODON_CHARS: set[str] = set('ATCG')
    KYTE_DOOLITTLE_HYDROPATHY: Dict[str, float] = {}
    def get_genetic_code(code_id: int = 1) -> Dict[str, str]: return STANDARD_GENETIC_CODE
    def get_synonymous_codons(gc: Dict[str, str]) -> Dict[str, List[str]]: return {}
    # Exit if core utilities cannot be loaded? Or let subsequent code fail?
    # sys.exit("Core utilities could not be imported.")


# --- Configure logging for this module ---
logger = logging.getLogger(__name__)


# === Nucleotide and Dinucleotide Frequency Calculations ===

def calculate_relative_dinucleotide_abundance(
    nucl_freqs: Dict[str, float],
    dinucl_freqs: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculates the relative dinucleotide abundance (Observed/Expected ratio).

    Expected frequency Exp(XY) = Freq(X) * Freq(Y).

    Args:
        nucl_freqs (Dict[str, float]): Dictionary mapping nucleotides ('A', 'C', 'G', 'T')
                                       to their frequencies.
        dinucl_freqs (Dict[str, float]): Dictionary mapping the 16 dinucleotides (e.g., 'AA')
                                         to their observed frequencies.

    Returns:
        Dict[str, float]: Dictionary mapping dinucleotides to their O/E ratio.
                          Returns empty dict if input is invalid or calculation fails.
                          Ratio is np.nan if expected frequency is zero but observed is non-zero.
                          Ratio is 1.0 if both observed and expected are zero (or close to zero).
    """
    if not nucl_freqs or not dinucl_freqs or len(nucl_freqs) != 4:
        logger.warning("Invalid input for relative dinucleotide abundance calculation. Returning empty dict.")
        return {}

    relative_abundance: Dict[str, float] = {}
    bases: str = 'ACGT'
    min_freq_threshold: float = 1e-9 # Threshold for effectively zero frequency

    for d1, d2 in itertools.product(bases, repeat=2):
        dinucleotide: str = d1 + d2
        observed_freq: float = dinucl_freqs.get(dinucleotide, 0.0)
        freq_n1: float = nucl_freqs.get(d1, 0.0)
        freq_n2: float = nucl_freqs.get(d2, 0.0)
        expected_freq: float = freq_n1 * freq_n2

        ratio: float
        if expected_freq > min_freq_threshold:
            ratio = observed_freq / expected_freq
        elif observed_freq < min_freq_threshold: # Both effectively zero
            ratio = 1.0
        else: # Observed > 0 but Expected is effectively zero
            ratio = np.nan

        relative_abundance[dinucleotide] = ratio

    return relative_abundance

def calculate_nucleotide_frequencies(sequences: List[SeqRecord]) -> Tuple[Dict[str, float], int]:
    """
    Calculates A, T, C, G frequencies across all provided sequences.

    Args:
        sequences (List[SeqRecord]): List of Biopython SeqRecord objects.

    Returns:
        Tuple[Dict[str, float], int]: A tuple containing:
            - Dictionary of nucleotide frequencies {'A': freq_A, ...}.
            - Total number of valid bases (A, T, C, G) counted.
    """
    counts: Counter[str] = Counter()
    total_bases: int = 0
    valid_bases: Set[str] = set('ATCG') # Consider only standard bases for frequency

    for record in sequences:
        try:
            seq_str: str = str(record.seq) # Assumes already uppercase
            for base in seq_str:
                 if base in valid_bases:
                     counts[base] += 1
                     total_bases += 1
        except AttributeError:
            logger.warning(f"Skipping record with invalid sequence object: ID {record.id}")
            continue

    freqs: Dict[str, float] = {base: counts.get(base, 0) / total_bases if total_bases > 0 else 0.0
                               for base in valid_bases}
    # Ensure all bases are present even if count is 0
    for base in 'ATCG':
        if base not in freqs:
            freqs[base] = 0.0

    return freqs, total_bases

def calculate_dinucleotide_frequencies(sequences: List[SeqRecord]) -> Tuple[Dict[str, float], int]:
    """
    Calculates frequencies of all 16 possible ATCG dinucleotides.

    Args:
        sequences (List[SeqRecord]): List of Biopython SeqRecord objects.

    Returns:
        Tuple[Dict[str, float], int]: A tuple containing:
            - Dictionary of dinucleotide frequencies {'AA': freq_AA, ...}.
            - Total number of valid dinucleotides counted.
    """
    counts: Counter[str] = Counter()
    total_dinucl: int = 0
    bases: str = 'ACGT'

    for record in sequences:
        try:
            seq_str: str = str(record.seq)
            for i in range(len(seq_str) - 1):
                dinucl: str = seq_str[i:i+2]
                # Only count valid ATCG dinucleotides
                if dinucl[0] in bases and dinucl[1] in bases:
                    counts[dinucl] += 1
                    total_dinucl += 1
        except AttributeError:
            logger.warning(f"Skipping record with invalid sequence object: ID {record.id}")
            continue

    freqs: Dict[str, float] = {}
    for d1 in bases:
        for d2 in bases:
             dinucl_key = d1 + d2
             freqs[dinucl_key] = counts.get(dinucl_key, 0) / total_dinucl if total_dinucl > 0 else 0.0

    return freqs, total_dinucl


# === Codon and Sequence Property Calculations ===

# Type alias for the return tuple of calculate_gc_content
GcContentType = Tuple[float, float, float, float, float]

def calculate_gc_content(sequence_str: str) -> GcContentType:
    """
    Calculates GC, GC1, GC2, GC3, GC12 content for a single DNA sequence string.
    Uses Bio.SeqUtils.GC123. Assumes input is cleaned (ATCG, no gaps, multiple of 3).

    Args:
        sequence_str (str): The DNA sequence string.

    Returns:
        GcContentType: Tuple (GC, GC1, GC2, GC3, GC12) as percentages (float).
                       Returns NaNs if calculation is not possible.
    """
    # Basic validation of input
    if not sequence_str or len(sequence_str) < 3 or len(sequence_str) % 3 != 0:
         logger.debug(f"Cannot calculate GC content for sequence of length {len(sequence_str)}. Needs multiple of 3.")
         return (np.nan,) * 5 # Return tuple of NaNs

    try:
        # GC123 expects string, returns percentages
        gc1, gc2, gc3, gc_overall = GC123(sequence_str)

        # Ensure results are numeric before calculating GC12
        gc1_f = float(gc1) if pd.notna(gc1) else np.nan
        gc2_f = float(gc2) if pd.notna(gc2) else np.nan
        gc3_f = float(gc3) if pd.notna(gc3) else np.nan
        gc_overall_f = float(gc_overall) if pd.notna(gc_overall) else np.nan

        gc12: float = np.nan
        if not np.isnan(gc1_f) and not np.isnan(gc2_f):
             gc12 = (gc1_f + gc2_f) / 2.0

        return (gc_overall_f, gc1_f, gc2_f, gc3_f, gc12)

    except (ZeroDivisionError, ValueError, TypeError) as e:
         # Catch specific errors during GC123 calculation
         logger.warning(f"Could not calculate GC123 content for sequence (len {len(sequence_str)}): {e}. Returning NaNs.")
         return (np.nan,) * 5
    except Exception as e: # Catch any other unexpected error
        logger.exception(f"Unexpected error in calculate_gc_content for sequence (len {len(sequence_str)}): {e}")
        return (np.nan,) * 5


def translate_sequence(sequence_str: str, genetic_code: Dict[str, str]) -> Optional[str]:
    """
    Translates a cleaned DNA sequence string (assumed multiple of 3) into protein sequence.

    Args:
        sequence_str (str): The DNA sequence string.
        genetic_code (Dict[str, str]): Genetic code dictionary mapping codons to amino acids.

    Returns:
        Optional[str]: Protein sequence string (may include '*' or 'X'), or None if input is empty.
    """
    if not sequence_str:
        return None

    protein: List[str] = []
    seq_len: int = len(sequence_str)
    # Calculate the end position for the loop to only include full codons
    last_full_codon_start: int = seq_len - (seq_len % 3)

    try:
        for i in range(0, last_full_codon_start, 3):
            codon: str = sequence_str[i:i+3]
            # Use 'X' for codons not found in the dictionary (e.g., containing 'N')
            # or if codon somehow isn't 3 chars (shouldn't happen with check before)
            aa: str = genetic_code.get(codon, 'X')
            protein.append(aa)
        return "".join(protein)
    except Exception as e:
        logger.exception(f"Error during translation of sequence (len {seq_len}): {e}")
        return None # Return None if translation fails


# Type alias for protein property tuple
ProteinPropType = Tuple[float, float]

def calculate_protein_properties(protein_sequence: Optional[str]) -> ProteinPropType:
    """
    Calculates GRAVY and Aromaticity for a protein sequence string.
    Handles None input, stop codons ('*'), and unknown AAs ('X', '?').

    Args:
        protein_sequence (Optional[str]): The protein sequence string.

    Returns:
        ProteinPropType: Tuple (GRAVY, Aromaticity) as floats, or (NaN, NaN).
    """
    if protein_sequence is None or not isinstance(protein_sequence, str) or not protein_sequence:
        return (np.nan, np.nan)

    # Remove potential stop codons ('*') and unknown AAs ('X', '?') before analysis
    # Use regex for potentially more robust cleaning if needed
    protein_sequence_cleaned: str = protein_sequence.replace('*', '').replace('X','').replace('?','')
    if not protein_sequence_cleaned:
        logger.debug("Protein sequence empty after removing non-standard AAs/stops.")
        return (np.nan, np.nan)

    try:
        # Create a new ProteinAnalysis object each time
        analysed_protein = ProteinAnalysis(protein_sequence_cleaned)

        # Calculate GRAVY
        try:
             gravy: float = analysed_protein.gravy()
             # Ensure result is float or NaN
             gravy = float(gravy) if pd.notna(gravy) else np.nan
        except (ValueError, TypeError, Exception) as gravy_err:
             logger.debug(f"Could not calculate GRAVY for sequence '{protein_sequence_cleaned[:20]}...': {gravy_err}")
             gravy = np.nan

        # Calculate Aromaticity
        try:
             aromaticity: float = analysed_protein.aromaticity()
             aromaticity = float(aromaticity) if pd.notna(aromaticity) else np.nan
        except (ValueError, TypeError, Exception) as arom_err:
             logger.debug(f"Could not calculate aromaticity for sequence '{protein_sequence_cleaned[:20]}...': {arom_err}")
             aromaticity = np.nan

        return (gravy, aromaticity)

    except (ValueError, KeyError) as e: # Catch specific errors from ProteinAnalysis init
        logger.warning(f"Could not instantiate ProteinAnalysis for sequence '{protein_sequence_cleaned[:20]}...': {e}")
        return (np.nan, np.nan)
    except Exception as e: # Catch other unexpected errors
        logger.exception(f"Unexpected error calculating protein properties for '{protein_sequence_cleaned[:20]}...': {e}")
        return (np.nan, np.nan)


# === Codon Usage Indices ===

def calculate_rscu(codon_counts_df: pd.DataFrame, genetic_code_id: int = 1) -> pd.DataFrame:
    """
    Calculates Relative Synonymous Codon Usage (RSCU) from codon counts.

    Args:
        codon_counts_df (pd.DataFrame): DataFrame with 'Codon' as index and a 'Count' column.
        genetic_code_id (int): NCBI genetic code ID. Default is 1.

    Returns:
        pd.DataFrame: DataFrame with columns ['Codon', 'AminoAcid', 'Count', 'Frequency', 'RSCU'].
                      Returns an empty DataFrame if input is invalid or calculation fails.
    """
    output_cols: List[str] = ['Codon', 'AminoAcid', 'Count', 'Frequency', 'RSCU']
    empty_df = pd.DataFrame(columns=output_cols) # DataFrame to return on error

    # Input validation
    if not isinstance(codon_counts_df, pd.DataFrame) or 'Count' not in codon_counts_df.columns:
        logger.warning("Invalid input for calculate_rscu: Input must be a DataFrame with a 'Count' column.")
        return empty_df
    if codon_counts_df.index.name != 'Codon':
        # Try setting index if 'Codon' column exists, otherwise fail
        if 'Codon' in codon_counts_df.columns:
            try:
                codon_counts_df = codon_counts_df.set_index('Codon')
            except Exception as idx_err:
                logger.error(f"Failed to set 'Codon' as index for RSCU calculation: {idx_err}")
                return empty_df
        else:
             logger.error("Invalid input for calculate_rscu: DataFrame must have 'Codon' as index or a 'Codon' column.")
             return empty_df

    # Ensure counts are numeric, handle NaNs, filter zero counts which don't contribute to RSCU sums
    try:
        rscu_df = codon_counts_df.copy()
        rscu_df['Count'] = pd.to_numeric(rscu_df['Count'], errors='coerce').fillna(0).astype(int)
        # Keep zero counts for Frequency calculation later, but they don't affect RSCU value directly
        # rscu_df = rscu_df[rscu_df['Count'] > 0] # Filter non-observed codons? Maybe not here.
    except Exception as conv_err:
        logger.error(f"Error converting counts to numeric for RSCU: {conv_err}")
        return empty_df

    # Check if any counts remain
    total_raw_codons = rscu_df['Count'].sum()
    if total_raw_codons <= 0:
        logger.debug("No non-zero codon counts found for RSCU calculation.")
        # Return df with original codons but NaN/0 values? Or empty? Let's return with values.
        rscu_df['AminoAcid'] = rscu_df.index.map(get_genetic_code(genetic_code_id).get) # Add AA column
        rscu_df['Frequency'] = 0.0
        rscu_df['RSCU'] = np.nan
        # Check if standard columns exist
        for col in output_cols:
            if col not in rscu_df: rscu_df[col] = 0 if col == 'Count' else (np.nan if col=='RSCU' else 0.0)
        return rscu_df.reset_index()[output_cols]


    try:
        genetic_code: Dict[str, str] = get_genetic_code(genetic_code_id)
        syn_codons: Dict[str, List[str]] = get_synonymous_codons(genetic_code)
    except (NotImplementedError, Exception) as e:
        logger.error(f"Error getting genetic code info (ID: {genetic_code_id}) for RSCU: {e}")
        return empty_df

    # Map Amino Acids
    rscu_df['AminoAcid'] = rscu_df.index.map(genetic_code.get)

    # Calculate totals using only valid coding codons present in the input
    valid_coding_df = rscu_df.dropna(subset=['AminoAcid'])
    valid_coding_df = valid_coding_df[valid_coding_df['AminoAcid'] != '*'] # Exclude stops
    total_coding_codons: int = int(valid_coding_df['Count'].sum()) # Ensure integer
    aa_counts: pd.Series = valid_coding_df.groupby('AminoAcid')['Count'].sum()

    rscu_values: Dict[str, float] = {} # Use dict for easier assignment

    for aa, syn_list in syn_codons.items():
        num_syn_codons: int = len(syn_list)
        # Skip stops and single-codon families (Met, Trp) - their RSCU is undefined or 1
        if aa == '*' or num_syn_codons <= 1:
            for codon in syn_list:
                if codon in rscu_df.index:
                    rscu_values[codon] = np.nan # Or 1.0 for single codons? NaN is safer.
            continue

        # Total count for this specific amino acid in the input data
        total_aa_count: int = int(aa_counts.get(aa, 0))

        # If this AA was not observed at all
        if total_aa_count <= 0:
            for codon in syn_list:
                if codon in rscu_df.index:
                    rscu_values[codon] = 0.0 # RSCU is 0 if AA count is 0
            continue

        # Calculate expected count under equal usage
        expected_count: float = total_aa_count / num_syn_codons

        # Avoid division by zero if expected count is somehow zero (shouldn't happen if total_aa_count > 0)
        if expected_count < 1e-9:
             for codon in syn_list:
                 if codon in rscu_df.index: rscu_values[codon] = 0.0
             continue

        # Calculate RSCU for each synonymous codon for this AA
        for codon in syn_list:
            if codon in rscu_df.index:
                 observed_count: int = int(rscu_df.loc[codon, 'Count'])
                 rscu: float = observed_count / expected_count
                 rscu_values[codon] = rscu
            # else: Codon from genetic code wasn't in the input counts df, do nothing

    # Add calculated RSCU values back to the main df
    rscu_df['RSCU'] = rscu_df.index.map(rscu_values)

    # Calculate overall frequency based on *coding* codons
    if total_coding_codons > 0:
         # Calculate frequency only for coding codons relative to total coding codons
         coding_indices = rscu_df['AminoAcid'] != '*' # Boolean mask
         rscu_df['Frequency'] = np.where(
             coding_indices,
             rscu_df['Count'] / total_coding_codons,
             0.0 # Frequency is 0 for stop codons
         )
    else:
         rscu_df['Frequency'] = 0.0

    # Final formatting
    rscu_df_final = rscu_df.reset_index()
    # Ensure all expected columns exist, handling potential missing AAs etc.
    for col in output_cols:
        if col not in rscu_df_final.columns:
            # Provide sensible defaults if a column is somehow missing
            if col == 'Count': rscu_df_final[col] = 0
            elif col == 'Frequency': rscu_df_final[col] = 0.0
            elif col == 'RSCU': rscu_df_final[col] = np.nan
            elif col == 'AminoAcid': rscu_df_final[col] = rscu_df_final['Codon'].map(genetic_code.get) # Remap just in case
            else: rscu_df_final[col] = np.nan # Default for any other unexpected missing column

    return rscu_df_final[output_cols]


def calculate_enc(codon_counts: Union[Dict[str, int], Counter[str]], genetic_code_id: int = 1) -> float:
    """
    Calculates the Effective Number of Codons (ENC) using Wright's formula.

    Args:
        codon_counts (Union[Dict[str, int], Counter[str]]): Codon counts for the sequence.
        genetic_code_id (int): NCBI genetic code ID. Default is 1.

    Returns:
        float: ENC value, or np.nan if insufficient data or calculation fails.
    """
    if not codon_counts: return np.nan

    try:
        genetic_code: Dict[str, str] = get_genetic_code(genetic_code_id)
        syn_codons: Dict[str, List[str]] = get_synonymous_codons(genetic_code)
    except (NotImplementedError, Exception) as e:
        logger.error(f"Error getting genetic code info (ID: {genetic_code_id}) for ENC: {e}")
        return np.nan

    # Group counts by amino acid, considering only synonymous families > 1
    aa_codon_counts: Dict[str, Dict[str, int]] = {}
    total_codons_in_families: int = 0 # Sum of codons in multi-codon families (2, 3, 4, 6)
    for codon, count in codon_counts.items():
        count = int(count) # Ensure integer
        if count <= 0: continue
        aa: Optional[str] = genetic_code.get(codon)
        # Process if it's a valid coding AA belonging to a multi-codon family
        if aa and aa != '*' and len(syn_codons.get(aa,[])) > 1:
            if aa not in aa_codon_counts: aa_codon_counts[aa] = {}
            aa_codon_counts[aa][codon] = count
            total_codons_in_families += count

    # Check threshold based on total codons in relevant families
    # Wright suggested this requires sufficient data within families.
    # Using a simple threshold on the sum across families might be less robust
    # than checking counts *within* families, but simpler to implement.
    min_codons_threshold = 30 # Minimum codons across all multi-codon families
    if total_codons_in_families < min_codons_threshold:
         logger.warning(f"Insufficient codons ({total_codons_in_families} < {min_codons_threshold}) in multi-codon families for reliable ENC calculation. ENC will be NaN.") #
         return np.nan

    # Calculate F_i values (homozygosity) for each synonymous family degree (2, 3, 4, 6)
    F_values: Dict[int, List[float]] = {2: [], 3: [], 4: [], 6: []}
    num_families_processed: Dict[int, int] = {2: 0, 3: 0, 4: 0, 6: 0}

    for aa, counts in aa_codon_counts.items():
        num_syn: int = len(syn_codons.get(aa, []))
        if num_syn in F_values:
            n_aa: int = sum(counts.values()) # Total codons for this AA
            # Require at least 2 codons observed for this AA to calculate F_i meaningfully
            if n_aa >= 2:
                try:
                    sum_p_sq: float = sum((c / n_aa) ** 2 for c in counts.values())
                    # Wright's F_i formula
                    F_i: float = (n_aa * sum_p_sq - 1) / (n_aa - 1)
                    # Check if F_i is valid (can be NaN if n_aa=1, handled above)
                    if pd.notna(F_i) and F_i >= 0: # Homozygosity should be non-negative
                        F_values[num_syn].append(F_i)
                        num_families_processed[num_syn] += 1
                    else:
                        logger.warning(f"Invalid F_{num_syn} calculated for AA {aa} (n_aa={n_aa}, sum_p_sq={sum_p_sq:.3f}). Skipping family for ENC.") #
                except ZeroDivisionError:
                     # Should not happen due to n_aa >= 2 check, but handle defensively
                     logger.warning(f"ZeroDivisionError calculating F_{num_syn} for AA {aa}. Skipping family for ENC.")
            # else: logger.debug(f"Skipping F_{num_syn} for AA {aa} (n_aa={n_aa} < 2).")

    # Calculate average F value for each degree with valid data
    avg_F: Dict[int, float] = {}
    for deg, vals in F_values.items():
        if vals: # If list is not empty
            avg_F[deg] = np.mean(vals)
        else:
            avg_F[deg] = 0.0 # Or should be NaN if no families of this degree were processed? Wright implies contribution is 0 then.

    # Calculate ENC using Wright's formula (or Fuglsang's modification if preferred)
    # ENC = 2 + (9 / F_avg_2) + (1 / F_avg_3) + (5 / F_avg_4) + (3 / F_avg_6)
    # Handle cases where avg_F is zero (meaning infinite contribution - cap ENC at 61)
    enc: float = 2.0 # Start with Met (1) + Trp (1) contribution

    # Add contributions only if the family type was observed (num_families_processed > 0)
    # and average homozygosity is non-zero (to avoid division by zero)
    if num_families_processed[2] > 0 and avg_F[2] > 1e-9: enc += 9.0 / avg_F[2]
    if num_families_processed[3] > 0 and avg_F[3] > 1e-9: enc += 1.0 / avg_F[3]
    if num_families_processed[4] > 0 and avg_F[4] > 1e-9: enc += 5.0 / avg_F[4]
    if num_families_processed[6] > 0 and avg_F[6] > 1e-9: enc += 3.0 / avg_F[6]

    # Check validity and cap ENC between 20 (min possible theoretically) and 61 (max)
    if not np.isfinite(enc) or enc < 2.0: # Check < 2.0 as contributions are added to 2.0
        logger.debug(f"ENC calculation resulted in invalid value ({enc:.3f}). Returning NaN.")
        return np.nan
    enc = min(enc, 61.0)
    # Optional lower bound check, though Wright's formula starts at 2.
    # enc = max(enc, 20.0) # Theoretical min if all families have max bias?

    return enc


def calculate_cai(codon_counts: Union[Dict[str, int], Counter[str]], reference_weights: Dict[str, float]) -> float:
    """
    Calculates the Codon Adaptation Index (CAI).

    Args:
        codon_counts (Union[Dict[str, int], Counter[str]]): Codon counts for the sequence.
        reference_weights (Dict[str, float]): Dictionary mapping codons to their relative
                                              adaptiveness weights (w). Should be pre-calculated.

    Returns:
        float: CAI value (geometric mean of weights), or np.nan if calculation fails.
    """
    # Check inputs
    if not reference_weights or not codon_counts:
        logger.debug("Cannot calculate CAI: Missing codon counts or reference weights.") # Redundant if called from wrapper
        return np.nan

    log_weights_sum: float = 0.0
    total_codons_in_calc: int = 0 # Only codons included in the reference weights

    for codon, count in codon_counts.items():
        count = int(count)
        if count <= 0: continue

        weight: Optional[float] = reference_weights.get(codon)

        # Only include codons present in the reference weight set
        if weight is not None:
            # Handle zero weight: CAI should be 0 if any codon used has weight 0
            if weight <= 1e-9: # Using threshold for floating point
                logger.debug(f"Codon {codon} has zero or near-zero weight ({weight}). CAI is 0.")
                return 0.0
            # Add log(weight) * count to sum
            try:
                log_weights_sum += math.log(weight) * count
                total_codons_in_calc += count
            except ValueError: # Log of negative weight (should not happen if weights are validated)
                logger.warning(f"Cannot calculate log for non-positive weight ({weight}) of codon {codon}. Skipping codon in CAI.")
            except Exception as e: # Catch other math errors
                logger.warning(f"Math error processing weight for codon {codon} (weight={weight}): {e}. Skipping codon.")

    # Check if any valid codons were found
    if total_codons_in_calc == 0:
        logger.debug("Cannot calculate CAI: No valid codons found with corresponding reference weights.")
        return np.nan

    # Calculate geometric mean: exp( sum(log(w_i) * count_i) / total_codons )
    try:
        cai: float = math.exp(log_weights_sum / total_codons_in_calc)
    except OverflowError:
        logger.warning("OverflowError calculating CAI (extremely high weights?). Returning infinity.")
        return np.inf # Or np.nan? Inf might indicate issue with weights.
    except Exception as e:
        logger.exception(f"Unexpected error calculating final CAI value: {e}")
        return np.nan

    # Ensure CAI is within reasonable bounds [0, 1] if weights are relative adaptiveness
    # This might depend on how weights were calculated. Standard CAI should be <= 1.
    if np.isfinite(cai):
        return max(0.0, min(cai, 1.0)) # Cap at 0 and 1
    else:
        logger.warning(f"CAI calculation resulted in non-finite value ({cai}). Returning NaN.")
        return np.nan

def calculate_fop(codon_counts: Union[Dict[str, int], Counter[str]], reference_weights: Dict[str, float]) -> float:
    """
    Calculates the Frequency of Optimal Codons (Fop).
    Optimal codons are those with reference_weight == 1.0.

    Args:
        codon_counts (Union[Dict[str, int], Counter[str]]): Codon counts for the sequence.
        reference_weights (Dict[str, float]): Dictionary mapping codons to weights.

    Returns:
        float: Fop value, or np.nan if calculation fails.
    """
    if not reference_weights or not codon_counts: 
        logger.debug("Cannot calculate Fop: Missing codon counts or reference weights.")       
        return np.nan

    # Identify optimal codons (weight exactly 1.0, allowing for float comparison)
    optimal_codons: Set[str] = {c for c, w in reference_weights.items() if np.isclose(w, 1.0)}
    if not optimal_codons:
        logger.warning("Cannot calculate Fop: No optimal codons (weight=1.0) found in reference set.")
        return np.nan

    optimal_count: int = 0
    total_count: int = 0 # Count only codons present in the reference set

    for codon, count in codon_counts.items():
        count = int(count)
        if count <= 0: continue
        # Check if codon exists in the reference weights
        if codon in reference_weights:
             total_count += count
             if codon in optimal_codons:
                 optimal_count += count

    # Calculate Fop
    if total_count == 0:
        logger.debug("Cannot calculate Fop: No codons found matching the reference set.")
        return np.nan
    else:
        return optimal_count / total_count


def calculate_rcdi(codon_counts: Union[Dict[str, int], Counter[str]], reference_weights: Dict[str, float]) -> float:
    """
    Calculates the Relative Codon Deoptimization Index (RCDI).
    This measures similarity to a "deoptimized" reference (low usage frequency weights).
    Formula: exp( sum( -log(w_i) * count_i ) / total_codons )
    Assumes weights (w_i) represent relative adaptiveness (higher = better).

    Args:
        codon_counts (Union[Dict[str, int], Counter[str]]): Codon counts for the sequence.
        reference_weights (Dict[str, float]): Dictionary mapping codons to weights.

    Returns:
        float: RCDI value, or np.nan if calculation fails. High value indicates deoptimization.
    """
    if not reference_weights or not codon_counts:
        logger.debug("Cannot calculate RCDI: Missing codon counts or reference weights.") # <-- Log ajouté/décommenté
        return np.nan

    log_inv_weights_sum: float = 0.0
    total_codons_in_calc: int = 0 # Only codons included in the reference weights

    for codon, count in codon_counts.items():
        count = int(count)
        if count <= 0: continue

        weight: Optional[float] = reference_weights.get(codon)
        # Only include codons present in the reference weight set
        if weight is not None:
            # Handle zero or negative weights: RCDI is undefined
            if weight <= 1e-9:
                logger.debug(f"Codon {codon} has zero or non-positive weight ({weight}). RCDI is undefined (NaN).")
                return np.nan
            # Add -log(weight) * count to sum
            try:
                log_inv_weights_sum += (-math.log(weight)) * count
                total_codons_in_calc += count
            except ValueError: # Should be caught by weight <= 1e-9, but safety check
                logger.warning(f"Cannot calculate log for non-positive weight ({weight}) of codon {codon}. Skipping codon in RCDI.")
            except Exception as e:
                 logger.warning(f"Math error processing weight for codon {codon} (weight={weight}) in RCDI: {e}. Skipping codon.")

    # Check if any valid codons were found
    if total_codons_in_calc == 0:
        logger.debug("Cannot calculate RCDI: No valid codons found with corresponding reference weights.")
        return np.nan

    # Calculate RCDI: exp( sum(-log(w_i) * count_i) / total_codons )
    try:
        rcdi: float = math.exp(log_inv_weights_sum / total_codons_in_calc)
    except OverflowError:
        logger.warning("OverflowError calculating RCDI (extremely low weights?). Returning infinity.")
        rcdi = np.inf
    except Exception as e:
        logger.exception(f"Unexpected error calculating final RCDI value: {e}")
        return np.nan

    # RCDI > 0. High value means deoptimized. No strict upper bound like CAI.
    return rcdi if np.isfinite(rcdi) else np.nan


# === Main Analysis Orchestration ===

# Type alias for the return tuple of analyze_single_sequence
SingleSeqResultType = Tuple[Optional[Dict[str, Any]], Optional[Counter[str]]]

def analyze_single_sequence(
    record: SeqRecord,
    genetic_code_id: int,
    reference_weights: Dict[str, float] # Assume passed as non-Optional dict (or empty)
) -> SingleSeqResultType:
    """
    Performs analysis for a single cleaned sequence record. Calculates metrics
    like GC, ENC, CAI, Fop, RCDI, protein properties.

    Args:
        record (SeqRecord): The sequence record (assumed cleaned).
        genetic_code_id (int): NCBI genetic code ID.
        reference_weights (Dict[str, float]): Pre-calculated reference weights. Pass empty dict if none.

    Returns:
        SingleSeqResultType: Tuple containing:
            - Dictionary of calculated metrics (or None).
            - Codon counts (Counter) for the sequence (or None).
    """
    try:
        genetic_code: Dict[str, str] = get_genetic_code(genetic_code_id)
    except (NotImplementedError, Exception) as e:
        logger.error(f"Cannot get genetic code {genetic_code_id} for seq {record.id}: {e}")
        return None, None

    seq_id: str = record.id
    seq_str: str = str(record.seq)
    seq_len: int = len(seq_str)
    # Initialize results dictionary
    results: Dict[str, Any] = {'ID': seq_id, 'Length': seq_len}
    # Initialize codon counts
    codon_counts_seq: Counter[str] = Counter()
    total_codons_seq: int = 0

    # Calculate Codon Counts
    try:
        for i in range(0, seq_len, 3):
            codon: str = seq_str[i:i+3]
            if all(base in VALID_CODON_CHARS for base in codon):
                codon_counts_seq[codon] += 1
                total_codons_seq += 1
        results['TotalCodons'] = total_codons_seq
    except Exception as count_err:
        logger.error(f"Error counting codons for sequence {seq_id}: {count_err}")
        # Return None if counting fails fundamentally? Or return partial results?
        # Let's return partial results with NaNs for calculated metrics.
        results['TotalCodons'] = 0
        total_codons_seq = 0 # Ensure this is 0

    # Define default NaN metrics
    nan_metrics: Dict[str, Any] = { # Use Any for mixed types
        'GC': np.nan, 'GC1': np.nan, 'GC2': np.nan, 'GC3': np.nan, 'GC12': np.nan,
        'ENC': np.nan, 'CAI': np.nan, 'Fop': np.nan, 'RCDI': np.nan,
        'GRAVY': np.nan, 'Aromaticity': np.nan, 'ProteinLength': 0
    }

    # If no codons counted, update with NaNs and return early
    if total_codons_seq == 0:
        logger.debug(f"No valid codons found or counted for sequence {seq_id}.")
        results.update(nan_metrics)
        return results, None # Return metrics dict (with NaNs) and None counts

    # Calculate Metrics (if codons > 0)
    try:
        # GC Content
        gc_tuple = calculate_gc_content(seq_str)
        results['GC'], results['GC1'], results['GC2'], results['GC3'], results['GC12'] = gc_tuple

        # Translation and Protein Properties
        protein_seq: Optional[str] = translate_sequence(seq_str, genetic_code)
        results['ProteinLength'] = len(protein_seq.replace('*','').replace('X','').replace('?','')) if protein_seq else 0
        gravy, aromaticity = calculate_protein_properties(protein_seq)
        results['GRAVY'], results['Aromaticity'] = gravy, aromaticity

        # Codon Usage Indices
        results['ENC'] = calculate_enc(codon_counts_seq, genetic_code_id)
        # Calculate reference-based indices only if reference_weights were provided
        if reference_weights:
            results['CAI'] = calculate_cai(codon_counts_seq, reference_weights)
            results['Fop'] = calculate_fop(codon_counts_seq, reference_weights)
            results['RCDI'] = calculate_rcdi(codon_counts_seq, reference_weights)
        else:
            # Assign NaN if no reference weights
            results['CAI'], results['Fop'], results['RCDI'] = np.nan, np.nan, np.nan

    except Exception as calc_err:
        # Log error during metric calculation
        logger.error(f"Error calculating metrics for sequence {seq_id} (len {seq_len}): {calc_err}")
        # Update results with NaNs for potentially missing metrics
        results.update(nan_metrics) # Ensure all metric keys exist

    # Return the metrics dictionary and the codon counts
    return results, codon_counts_seq


# --- Main Analysis Function (SEQUENTIAL per-sequence analysis) ---
# Define return type alias
AnalysisResultType = Tuple[
    pd.DataFrame,               # agg_usage_df
    pd.DataFrame,               # per_sequence_df
    Dict[str, float],           # overall_nucl_freqs
    Dict[str, float],           # overall_dinucl_freqs
    None,                       # Placeholder
    None,                       # ca_results always None now
    Optional[pd.DataFrame]      # ca_input_df (still calculated)
]

def run_full_analysis(
    sequences: List[SeqRecord],
    genetic_code_id: int = 1,
    reference_weights: Optional[Dict[str, float]] = None
) -> AnalysisResultType:
    """
    Performs all analyses SEQUENTIALLY for the provided list of sequences.
    Relies on pre-loaded reference_weights for CAI/Fop/RCDI.
    Optionally prepares data for and performs Correspondence Analysis (CA).

    Args:
        sequences (List[SeqRecord]): Input sequences for a single gene/dataset.
        genetic_code_id (int): NCBI genetic code ID.
        reference_weights (Optional[Dict[str, float]]): Pre-loaded reference weights (w).
        fit_ca_model (bool): If True, performs Correspondence Analysis fitting.
                           If False, CA input data might still be generated but fitting is skipped.

    Returns:
        AnalysisResultType: Tuple containing aggregated results, per-sequence results,
                            frequencies, and optional CA results.
    """
    # --- Setup ---
    try:
        genetic_code: Dict[str, str] = get_genetic_code(genetic_code_id)
    except (NotImplementedError, Exception) as e:
        logger.critical(f"Failed to get genetic code {genetic_code_id}. Cannot run analysis. Error: {e}")
        # Return empty/None structure
        empty_df = pd.DataFrame()
        return empty_df, empty_df, {}, {}, None, None, None

    logger.debug(f"Starting full analysis for {len(sequences)} sequences.")
    # Calculate overall frequencies based on the input sequences
    overall_nucl_freqs: Dict[str, float]
    total_bases: int
    overall_nucl_freqs, total_bases = calculate_nucleotide_frequencies(sequences)
    logger.debug(f"Calculated overall nucleotide frequencies (total bases: {total_bases}).")

    overall_dinucl_freqs: Dict[str, float]
    total_dinucl: int
    overall_dinucl_freqs, total_dinucl = calculate_dinucleotide_frequencies(sequences)
    logger.debug(f"Calculated overall dinucleotide frequencies (total dinucleotides: {total_dinucl}).")

    # --- Per-Sequence Analysis (SEQUENTIAL) ---
    # `partial` fixes constant arguments for `analyze_single_sequence`
    # Pass an empty dict for weights if None, to simplify analyze_single_sequence logic
    _reference_weights_internal: Dict[str, float] = reference_weights if reference_weights is not None else {}
    analysis_func = partial(analyze_single_sequence,
                            genetic_code_id=genetic_code_id,
                            reference_weights=_reference_weights_internal)

    # Sequential loop over the provided sequences
    logger.debug("Starting sequential analysis of individual sequences...")
    all_results: List[SingleSeqResultType] = [analysis_func(seq) for seq in sequences]
    logger.debug("Finished sequential analysis of individual sequences.")

    # --- Process results ---
    per_sequence_metrics_list: List[Dict[str, Any]] = []
    aggregate_codon_counts: Counter[str] = Counter()
    sequences_for_ca: Dict[str, pd.Series] = {} # {ID: Series(RSCU)}

    for result_tuple in all_results:
        if result_tuple and result_tuple[0] is not None: # Check if metrics dict exists
             metrics_dict, counts_dict = result_tuple
             per_sequence_metrics_list.append(metrics_dict)
             # Update aggregate counts and prepare CA data if counts exist
             if counts_dict:
                 aggregate_codon_counts.update(counts_dict)
                 seq_id = metrics_dict.get('ID')
                 # Prepare RSCU data for CA if ID exists
                 if seq_id:
                     try:
                         # Create temporary DataFrame for this sequence only
                         temp_counts_df = pd.DataFrame.from_dict(counts_dict, orient='index', columns=['Count'])
                         if not temp_counts_df.empty: # Check df is not empty
                            temp_counts_df.index.name = 'Codon'
                            # Calculate RSCU for this single sequence
                            temp_rscu_df = calculate_rscu(temp_counts_df, genetic_code_id=genetic_code_id)
                            # Store the RSCU vector (index=Codon, values=RSCU), fill NaN with 0.0
                            if not temp_rscu_df.empty:
                                 rscu_vector: pd.Series = temp_rscu_df.set_index('Codon')['RSCU'].fillna(0.0)
                                 sequences_for_ca[seq_id] = rscu_vector
                     except Exception as rscu_err:
                          logger.warning(f"Could not calculate RSCU for CA prep for seq {seq_id}: {rscu_err}")

    # --- Check for results ---
    empty_df = pd.DataFrame() # Define once
    if not per_sequence_metrics_list:
         logger.warning("No per-sequence metrics generated for this set.")
         # Still return overall frequencies, but empty DFs and None for CA
         return empty_df, empty_df, overall_nucl_freqs, overall_dinucl_freqs, None, None, None

    # Create the final DataFrame for per-sequence metrics
    try:
        per_sequence_df = pd.DataFrame(per_sequence_metrics_list)
    except Exception as df_err:
         logger.exception(f"Error creating per-sequence DataFrame: {df_err}")
         # Return overall frequencies, but empty DFs and None for CA
         return empty_df, empty_df, overall_nucl_freqs, overall_dinucl_freqs, None, None, None

    # --- Aggregate Codon Usage ---
    agg_usage_df: pd.DataFrame = empty_df # Initialize empty
    if aggregate_codon_counts:
        logger.debug("Calculating aggregate codon usage (RSCU, Freq)...")
        try:
            agg_counts_df = pd.DataFrame.from_dict(aggregate_codon_counts, orient='index', columns=['Count'])
            agg_counts_df.index.name = 'Codon'
            agg_counts_df.sort_index(inplace=True)
            agg_usage_df = calculate_rscu(agg_counts_df, genetic_code_id=genetic_code_id)
            logger.debug("Finished calculating aggregate codon usage.")
        except Exception as agg_err:
             logger.exception(f"Error calculating aggregate codon usage: {agg_err}")
             agg_usage_df = empty_df # Ensure it's empty on error

    # --- CA Data Preparation ---
    ca_input_df: Optional[pd.DataFrame] = None
    if sequences_for_ca:
        logger.debug("Preparing CA input DataFrame (per-sequence RSCU)...")
        try:
            ca_input_df = pd.DataFrame.from_dict(sequences_for_ca, orient='index')
            # Ensure all coding codons are present as columns, in order
            all_codons: List[str] = sorted([c for c, aa in genetic_code.items() if aa != '*'])
            ca_input_df = ca_input_df.reindex(columns=all_codons, fill_value=0.0)
            # Clean data (NaNs already filled, handle Inf)
            ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True)
            # Filter rows (sequences) and columns (codons) with zero/near-zero variance
            rows_before = ca_input_df.shape[0]
            cols_before = ca_input_df.shape[1]
            ca_input_df = ca_input_df.loc[ca_input_df.sum(axis=1).abs() > 1e-9]
            ca_input_df = ca_input_df.loc[:, ca_input_df.sum(axis=0).abs() > 1e-9]
            logger.debug(f"CA input filtering: {rows_before}x{cols_before} -> {ca_input_df.shape[0]}x{ca_input_df.shape[1]}")
            # If DataFrame becomes empty after filtering, set to None
            if ca_input_df.empty:
                 logger.warning("CA input DataFrame became empty after filtering zero-variance rows/columns.")
                 ca_input_df = None
            else:
                 logger.debug("CA input DataFrame prepared successfully.")
        except Exception as e:
             ca_input_df = None
             logger.exception(f"Error preparing CA input data: {e}")

    # --- Return all results ---
    logger.debug("run_full_analysis finished.")
    return (agg_usage_df, per_sequence_df, overall_nucl_freqs, overall_dinucl_freqs,
            None, None, ca_input_df)


# === Statistical Comparison Function ===
def compare_features_between_genes(
    combined_per_sequence_df: pd.DataFrame,
    features: List[str],
    method: str = 'kruskal'
) -> Optional[pd.DataFrame]:
    """
    Performs statistical tests to compare features between gene groups.

    Args:
        combined_per_sequence_df (pd.DataFrame): DataFrame containing per-sequence
                                                 metrics including a 'Gene' column.
        features (List[str]): List of column names (features) to compare.
        method (str): Statistical test method ('kruskal' for Kruskal-Wallis H-test
                      or 'anova' for one-way ANOVA). Default is 'kruskal'.

    Returns:
        Optional[pd.DataFrame]: DataFrame summarizing the test results (Feature,
                                Test Statistic, P-value) or None if stats cannot be run.
    """
    if scipy_stats_module is None: # Check the runtime import        logger.warning("scipy.stats module not found. Cannot perform statistical comparisons.")
        return None
    if combined_per_sequence_df is None or combined_per_sequence_df.empty:
        logger.warning("Input data empty for compare_features_between_genes.")
        return None
    if 'Gene' not in combined_per_sequence_df.columns:
        logger.error("'Gene' column missing. Cannot perform statistical comparisons.")
        return None

    results: List[Dict[str, Any]] = []
    valid_genes: List[str] = combined_per_sequence_df['Gene'].unique().tolist()
    if len(valid_genes) < 2:
        logger.warning("Need at least two gene groups for comparison.")
        return None

    logger.info(f"Comparing features between groups using {method} test...")
    for feature in features:
        if feature not in combined_per_sequence_df.columns:
            logger.warning(f"Feature '{feature}' not found in data. Skipping comparison.")
            continue

        # Prepare data for the test: list of arrays/Series, one per gene group
        try:
            # Ensure data is numeric and drop NaNs for the specific feature
            feature_data: pd.DataFrame = combined_per_sequence_df[['Gene', feature]].copy()
            feature_data[feature] = pd.to_numeric(feature_data[feature], errors='coerce')
            feature_data = feature_data.dropna(subset=[feature]) # Drop only if feature value is NaN

            # Group data by gene
            groups_data: List[pd.Series] = [
                group[feature] for name, group in feature_data.groupby('Gene') if not group.empty
            ]

            # Check if we have enough data in enough groups AFTER dropping NaNs for this feature
            valid_groups_data: List[pd.Series] = [g for g in groups_data if len(g) > 0]
            if len(valid_groups_data) < 2:
                logger.debug(f"Not enough groups with valid data for feature '{feature}' after NaN removal. Skipping.")
                continue

        except KeyError:
             logger.error(f"KeyError preparing data for feature '{feature}'. Skipping comparison.")
             continue
        except Exception as prep_err:
             logger.exception(f"Error preparing data for feature '{feature}': {prep_err}. Skipping comparison.")
             continue


        statistic: float = np.nan
        p_value: float = np.nan
        test_name: str = "N/A"

        try:
            if method.lower() == 'kruskal':
                test_name = "Kruskal-Wallis H"
                statistic, p_value = scipy_stats_module.kruskal(*valid_groups_data)
            elif method.lower() == 'anova':
                test_name = "One-way ANOVA F"
                statistic, p_value = scipy_stats_module.f_oneway(*valid_groups_data)
            else:
                logger.warning(f"Unknown statistical method '{method}'. Skipping comparison for '{feature}'.")
                continue

        except ValueError as ve:
            # Catches errors like non-numeric data if coercion failed subtly, or unequal lengths for some tests
            logger.warning(f"Statistical test ({method}) failed for feature '{feature}'. Error: {ve}. Skipping.")
        except Exception as e: # Catch other stats errors
            logger.exception(f"Error during statistical test ({method}) for feature '{feature}': {e}")

        # Append results even if test failed (will have NaN values)
        results.append({
            'Feature': feature,
            'Test': test_name,
            'Statistic': statistic,
            'P_value': p_value
        })
        logger.debug(f"  Comparison for '{feature}': Stat={statistic:.4g}, p={p_value:.4g}")

    if not results:
        logger.warning("No features were successfully compared.")
        return None

    return pd.DataFrame(results)


# === CA Fitting Function ===
def perform_ca(ca_input_df: pd.DataFrame, n_components: int = 10) -> Optional[PrinceCA]: # type: ignore
    """
    Performs Correspondence Analysis on the input DataFrame using the 'prince' library.

    Args:
        ca_input_df (pd.DataFrame): DataFrame with sequences as index, codons as columns,
                                   and RSCU (or counts) as values. Should be cleaned
                                   (no NaNs/Infs, zero-variance rows/cols potentially removed).
        n_components (int): Maximum number of components for CA. Default 10.

    Returns:
        Optional[prince.CA]: Fitted CA object from prince, or None if CA fails or library missing.
    """
    if prince is None:
        logger.error("'prince' library not installed. Cannot perform CA.")
        return None
    if ca_input_df is None or ca_input_df.empty:
         logger.error("No valid input data provided for CA.")
         return None

    logger.info(f"Performing CA on DataFrame of shape {ca_input_df.shape}...")
    try:
        # Ensure data is numeric (should be already, but double check)
        ca_input_df = ca_input_df.astype(float)

        # Check shape again after potential filtering outside this function
        if ca_input_df.shape[0] < 2 or ca_input_df.shape[1] < 2:
            logger.error(f"Cannot perform CA: Input data shape ({ca_input_df.shape}) is too small (requires >= 2 rows and >= 2 columns).")
            return None

        # Determine number of components dynamically, limited by data dimensions
        actual_n_components: int = min(n_components,
                                       ca_input_df.shape[0] - 1,
                                       ca_input_df.shape[1] - 1)

        if actual_n_components < 1:
            logger.error(f"Calculated number of components for CA ({actual_n_components}) is less than 1.")
            return None

        logger.debug(f"Fitting CA with n_components = {actual_n_components}")
        # Initialize and fit CA
        ca = prince.CA(n_components=actual_n_components, n_iter=10, random_state=42, copy=True)
        ca_fitted: PrinceCA = ca.fit(ca_input_df) # type: ignore # Use the potentially filtered DataFrame
        logger.info("CA fitting completed.")
        return ca_fitted

    except ValueError as ve: # Catch potential errors during fit (e.g., data issues)
         logger.exception(f"ValueError during CA fitting: {ve}. Check input data matrix.")
         return None
    except Exception as e: # Catch other unexpected errors from prince
        logger.exception(f"Unexpected error during CA calculation: {e}")
        return None