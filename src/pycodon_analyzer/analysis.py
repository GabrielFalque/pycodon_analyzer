# src/pycodon_analyzer/analysis.py

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Core analysis functions for codon usage and sequence properties.
"""
import itertools
import os
from collections import Counter
import sys
import pandas as pd # type: ignore
import numpy as np # type: ignore
import math # For log, exp in CAI, etc.
from Bio.SeqUtils import GC123, molecular_weight # type: ignore # Use BioPython's GC utils
from Bio.SeqUtils.ProtParam import ProteinAnalysis # type: ignore # For GRAVY, Aromaticity

# Import prince for CA
try:
    import prince # type: ignore
except ImportError:
    prince = None # Handle optional dependency
try:
    from scipy import stats
except ImportError:
    stats = None # Handle optional dependency if making scipy optional
from functools import partial # Added for passing args with map

# --- Local modules ---
try:
    from .utils import (get_genetic_code, get_synonymous_codons,
                        VALID_CODON_CHARS, KYTE_DOOLITTLE_HYDROPATHY)
except ImportError: # Fallback for potential import issues during testing?
    print("Error importing from .utils, check package structure/installation.", file=sys.stderr)
    # Define fallbacks if absolutely needed for basic functionality without utils
    STANDARD_GENETIC_CODE = {} # Placeholder
    VALID_CODON_CHARS = set('ATCG')
    KYTE_DOOLITTLE_HYDROPATHY = {}
    def get_genetic_code(code_id=1): return STANDARD_GENETIC_CODE
    def get_synonymous_codons(gc): return {}

# --- Multiprocessing ---
try:
    import multiprocessing as mp
    from functools import partial
except ImportError:
    mp = None
    partial = None
    print("Warning: multiprocessing module not found, analysis will run sequentially.", file=sys.stderr)


# === Nucleotide and Dinucleotide Frequency Calculations ===

def calculate_relative_dinucleotide_abundance(nucl_freqs, dinucl_freqs):
    """
    Calculates the relative dinucleotide abundance (Observed/Expected ratio).

    Expected frequency Exp(XY) = Freq(X) * Freq(Y).

    Args:
        nucl_freqs (dict): Dictionary mapping nucleotides ('A', 'C', 'G', 'T')
                           to their frequencies.
        dinucl_freqs (dict): Dictionary mapping the 16 dinucleotides (e.g., 'AA')
                             to their observed frequencies.

    Returns:
        dict: Dictionary mapping dinucleotides to their O/E ratio.
              Returns empty dict if input is invalid or calculation fails.
              Ratio is np.nan if expected frequency is zero but observed is non-zero.
              Ratio is 1.0 if both observed and expected are zero (or close to zero).
    """
    if not nucl_freqs or not dinucl_freqs or len(nucl_freqs) != 4:
        # print("Warning: Invalid input for relative dinucleotide abundance calculation.", file=sys.stderr)
        return {}

    relative_abundance = {}
    bases = 'ACGT'
    min_freq_threshold = 1e-9 # Threshold for effectively zero frequency

    for d1, d2 in itertools.product(bases, repeat=2):
        dinucleotide = d1 + d2
        observed_freq = dinucl_freqs.get(dinucleotide, 0.0)
        freq_n1 = nucl_freqs.get(d1, 0.0)
        freq_n2 = nucl_freqs.get(d2, 0.0)
        expected_freq = freq_n1 * freq_n2

        if expected_freq > min_freq_threshold:
            ratio = observed_freq / expected_freq
        elif observed_freq < min_freq_threshold: # Both effectively zero
            ratio = 1.0
        else: # Observed > 0 but Expected is effectively zero
            ratio = np.nan

        relative_abundance[dinucleotide] = ratio

    return relative_abundance

def calculate_nucleotide_frequencies(sequences: list['SeqRecord']) -> tuple[dict, int]:
    """Calculates A, T, C, G frequencies across all sequences."""
    counts = Counter()
    total_bases = 0
    for record in sequences:
        seq_str = str(record.seq) # Assumes already uppercase
        for base in seq_str:
             if base in 'ATCG':
                 counts[base] += 1
                 total_bases += 1

    freqs = {base: counts.get(base, 0) / total_bases if total_bases > 0 else 0.0
             for base in 'ATCG'}
    return freqs, total_bases

def calculate_dinucleotide_frequencies(sequences: list['SeqRecord']) -> tuple[dict, int]:
    """Calculates frequencies of all 16 dinucleotides."""
    counts = Counter()
    total_dinucl = 0
    bases = 'ATCG'

    for record in sequences:
        seq_str = str(record.seq)
        for i in range(len(seq_str) - 1):
            dinucl = seq_str[i:i+2]
            # Only count valid ATCG dinucleotides
            if dinucl[0] in bases and dinucl[1] in bases:
                counts[dinucl] += 1
                total_dinucl += 1

    freqs = {d1+d2: counts.get(d1+d2, 0) / total_dinucl if total_dinucl > 0 else 0.0
             for d1 in bases for d2 in bases}
    return freqs, total_dinucl


# === Codon and Sequence Property Calculations ===

def calculate_gc_content(sequence_str: str) -> tuple:
    """
    Calculates GC, GC1, GC2, GC3, GC12 content for a single DNA sequence string.
    Uses Bio.SeqUtils.GC123. Assumes input is cleaned (ATCG, no gaps, mult of 3).

    Returns: tuple: (GC, GC1, GC2, GC3, GC12) as percentages (float), or NaNs.
    """
    if not sequence_str or len(sequence_str) < 3 or len(sequence_str) % 3 != 0:
         # Return NaNs if basic conditions not met
         return (np.nan,) * 5

    try:
        # GC123 expects string, returns percentages
        gc1, gc2, gc3, gc_overall = GC123(sequence_str)
        # Ensure results are numeric before calculating GC12
        if pd.isna(gc1) or pd.isna(gc2):
             gc12 = np.nan
        else:
             gc12 = (float(gc1) + float(gc2)) / 2.0
        # Return as floats, handling potential non-numeric returns from GC123
        return (
            float(gc_overall) if pd.notna(gc_overall) else np.nan,
            float(gc1) if pd.notna(gc1) else np.nan,
            float(gc2) if pd.notna(gc2) else np.nan,
            float(gc3) if pd.notna(gc3) else np.nan,
            gc12 # Already float or nan
            )
    except (ZeroDivisionError, ValueError, TypeError, Exception) as e:
         # Catch errors during GC123 calculation
         print(f"Warning: Could not calculate GC123 content: {e}. Seq Len: {len(sequence_str)}", file=sys.stderr)
         # Try calculating overall GC manually as fallback
         try:
             gc_overall_manual = (sequence_str.count('G') + sequence_str.count('C')) / len(sequence_str) * 100
             return (gc_overall_manual, np.nan, np.nan, np.nan, np.nan)
         except ZeroDivisionError: # Handles empty string case after filtering?
              return (np.nan,) * 5


def translate_sequence(sequence_str: str, genetic_code: dict) -> str | None:
    """
    Translates a cleaned DNA sequence string into protein sequence.

    Returns: str or None: Protein sequence string (may include '*'), or None if empty.
    """
    if not sequence_str: return None

    protein = []
    seq_len = len(sequence_str)

    for i in range(0, seq_len, 3): # Assumes length is multiple of 3
        codon = sequence_str[i:i+3]
        # Use 'X' for codons not found in the dictionary (e.g., containing 'N' if not filtered)
        aa = genetic_code.get(codon, 'X')
        protein.append(aa)

    return "".join(protein)


def calculate_protein_properties(protein_sequence: str | None) -> tuple:
    """
    Calculates GRAVY and Aromaticity for a protein sequence string.
    Handles None input and potential errors from BioPython.

    Returns: tuple: (GRAVY, Aromaticity) as floats, or (NaN, NaN).
    """
    if protein_sequence is None or not isinstance(protein_sequence, str):
        return (np.nan, np.nan)

    # Remove potential stop codons ('*') and unknown AAs ('X', '?') before analysis
    protein_sequence_cleaned = protein_sequence.replace('*', '').replace('X','').replace('?','')
    if not protein_sequence_cleaned:
        return (np.nan, np.nan)

    try:
        # Use a new ProteinAnalysis object each time
        analysed_protein = ProteinAnalysis(protein_sequence_cleaned)

        try:
             gravy = analysed_protein.gravy()
             # Ensure float conversion happens correctly
             gravy = float(gravy) if pd.notna(gravy) else np.nan
        except Exception: gravy = np.nan # Catch error during calculation

        try:
             aromaticity = analysed_protein.aromaticity()
             aromaticity = float(aromaticity) if pd.notna(aromaticity) else np.nan
        except Exception: aromaticity = np.nan

        return (gravy, aromaticity)
    except (ValueError, KeyError, Exception) as e:
        # Catch errors during ProteinAnalysis object creation
        print(f"Warning: Could not instantiate ProteinAnalysis: {e}", file=sys.stderr)
        return (np.nan, np.nan)
    

# === Codon Usage Indices ===

# --- Keep RSCU function (maybe rename calculate_aggregate_rscu?) ---
def calculate_rscu(codon_counts_df: pd.DataFrame, genetic_code_id: int = 1) -> pd.DataFrame:
    """
    Calculates Relative Synonymous Codon Usage (RSCU).

    Returns: pd.DataFrame with columns ['Codon', 'AminoAcid', 'Count', 'Frequency', 'RSCU'].
    """
    output_cols = ['Codon', 'AminoAcid', 'Count', 'Frequency', 'RSCU']
    # Input validation
    if not isinstance(codon_counts_df, pd.DataFrame) or 'Count' not in codon_counts_df.columns:
        # print("Warning: Invalid input for calculate_rscu.", file=sys.stderr)
        return pd.DataFrame(columns=output_cols)
    if codon_counts_df.index.name != 'Codon':
        if 'Codon' in codon_counts_df.columns: codon_counts_df = codon_counts_df.set_index('Codon')
        elif isinstance(codon_counts_df.index, pd.Index): pass # Assume index is codons
        else: return pd.DataFrame(columns=output_cols) # Invalid structure

    # Ensure counts are numeric
    codon_counts_df['Count'] = pd.to_numeric(codon_counts_df['Count'], errors='coerce').fillna(0).astype(int)
    codon_counts_df = codon_counts_df[codon_counts_df['Count'] > 0] # Filter zero counts early? No, keep for frequency calculation

    if codon_counts_df.empty or codon_counts_df['Count'].sum() <= 0:
        return pd.DataFrame(columns=output_cols)

    try:
        genetic_code = get_genetic_code(genetic_code_id)
        syn_codons = get_synonymous_codons(genetic_code)
    except Exception as e:
        print(f"Error getting genetic code info for RSCU: {e}", file=sys.stderr)
        return pd.DataFrame(columns=output_cols)

    rscu_df = codon_counts_df.copy()
    rscu_df['AminoAcid'] = rscu_df.index.map(genetic_code.get)

    # Calculate totals using only valid coding codons present in the input
    valid_coding_df = rscu_df.dropna(subset=['AminoAcid'])
    valid_coding_df = valid_coding_df[valid_coding_df['AminoAcid'] != '*']
    total_coding_codons = valid_coding_df['Count'].sum()
    aa_counts = valid_coding_df.groupby('AminoAcid')['Count'].sum()

    rscu_values = {} # Use dict for easier assignment

    for aa, syn_list in syn_codons.items():
        if aa == '*' or len(syn_list) <= 1: # Skip stops and single-codon families
            for codon in syn_list:
                if codon in rscu_df.index: rscu_values[codon] = np.nan
            continue

        total_aa_count = aa_counts.get(aa, 0)
        num_syn_codons = len(syn_list)

        if total_aa_count <= 0: # If this AA was not observed at all
            for codon in syn_list:
                if codon in rscu_df.index: rscu_values[codon] = 0.0 # RSCU is 0 if AA count is 0
            continue

        expected_count = total_aa_count / num_syn_codons
        if expected_count < 1e-9: # Avoid division by zero if somehow expected is zero
             for codon in syn_list:
                 if codon in rscu_df.index: rscu_values[codon] = 0.0 # Or NaN? Let's use 0.
             continue

        # Calculate RSCU for each codon in the family
        for codon in syn_list:
            if codon in rscu_df.index:
                 observed_count = rscu_df.loc[codon, 'Count']
                 rscu = observed_count / expected_count
                 rscu_values[codon] = rscu
            # else: Codon wasn't in input, do nothing

    # Add calculated RSCU values back to the main df
    rscu_df['RSCU'] = rscu_df.index.map(rscu_values)

    # Calculate overall frequency
    if total_coding_codons > 0:
         rscu_df['Frequency'] = rscu_df['Count'] / total_coding_codons
    else:
         rscu_df['Frequency'] = 0.0

    # Final formatting
    rscu_df = rscu_df.reset_index()
    for col in output_cols: # Ensure all expected columns exist
        if col not in rscu_df.columns: rscu_df[col] = np.nan

    return rscu_df[output_cols]

def calculate_enc(codon_counts: dict | Counter, genetic_code_id: int = 1) -> float:
    """ Calculates ENC using Wright's formula. Returns NaN if insufficient data. """
    if not codon_counts: return np.nan

    try:
        genetic_code = get_genetic_code(genetic_code_id)
        syn_codons = get_synonymous_codons(genetic_code)
    except Exception as e:
        print(f"Error getting genetic code info for ENC: {e}", file=sys.stderr)
        return np.nan

    aa_codon_counts = {}
    total_codons_in_families = 0
    for codon, count in codon_counts.items():
        if count <= 0: continue
        aa = genetic_code.get(codon)
        if aa and aa != '*' and len(syn_codons.get(aa,[])) > 1:
            if aa not in aa_codon_counts: aa_codon_counts[aa] = {}
            aa_codon_counts[aa][codon] = count
            total_codons_in_families += count

    # Threshold Check
    # Check based on total codons in multi-codon families
    if total_codons_in_families < 50: # Keep threshold for now
         # print("Warning: Not enough codons in multi-codon families for reliable ENC calc.", file=sys.stderr)
         return np.nan
    
    F_values = {2: [], 3: [], 4: [], 6: []}
    for aa, counts in aa_codon_counts.items():
        num_syn = len(syn_codons.get(aa, []))
        if num_syn in F_values:
            n_aa = sum(counts.values())
            if n_aa > 1:
                sum_p_sq = sum((c / n_aa) ** 2 for c in counts.values() if n_aa > 0) # Check n_aa > 0
                F_i = (n_aa * sum_p_sq - 1) / (n_aa - 1)
                if pd.notna(F_i) and F_i >= 0: F_values[num_syn].append(F_i)

    avg_F = {deg: np.mean(vals) if vals else 0 for deg, vals in F_values.items()}
    n_deg_families = {deg: len(vals) for deg, vals in F_values.items()}

    enc = 2.0 # Start with Met, Trp contribution

    # Add contributions, checking if family type had data (n_deg_families > 0)
    if avg_F[2] > 1e-9 and n_deg_families[2] > 0: enc += 9.0 / avg_F[2]
    if avg_F[3] > 1e-9 and n_deg_families[3] > 0: enc += 1.0 / avg_F[3]
    if avg_F[4] > 1e-9 and n_deg_families[4] > 0: enc += 5.0 / avg_F[4]
    if avg_F[6] > 1e-9 and n_deg_families[6] > 0: enc += 3.0 / avg_F[6]

    if not np.isfinite(enc) or enc < 2.0: return np.nan # Check validity
    enc = min(enc, 61.0) # Cap at max

    return enc


def calculate_cai(codon_counts: dict | Counter, reference_weights: dict) -> float:
    """
    Calculates the Codon Adaptation Index (CAI).

    Args:
        codon_counts (dict or Counter): Codon counts for the sequence.
        reference_weights (dict): Dictionary mapping codons to their relative
                                   adaptiveness weights (w).

    Returns:
        float: CAI value, or np.nan if calculation fails.
    """
    if not reference_weights or not codon_counts: 
        return np.nan
    log_weights_sum = 0.0
    total_codons_in_calc = 0
    for codon, count in codon_counts.items():
        if count <= 0: continue
        weight = reference_weights.get(codon)
        if weight is not None:
            if weight > 1e-9: log_weights_sum += math.log(weight) * count; total_codons_in_calc += count
            elif weight <= 1e-9: return 0.0 # Weight 0 means CAI=0
    if total_codons_in_calc == 0: return np.nan
    try: cai = math.exp(log_weights_sum / total_codons_in_calc)
    except OverflowError: return np.inf # Should not happen if weights <= 1
    return max(0.0, min(cai, 1.0)) if np.isfinite(cai) else np.nan

def calculate_fop(codon_counts: dict | Counter, reference_weights: dict) -> float:
    """ Calculates Frequency of Optimal Codons (Fop). """
    if not reference_weights or not codon_counts: return np.nan
    optimal_codons = {c for c, w in reference_weights.items() if np.isclose(w, 1.0)}
    if not optimal_codons: return np.nan
    optimal_count = 0
    total_count = 0
    for codon, count in codon_counts.items():
        if count <= 0: continue
        if codon in reference_weights: # Only count codons defined in reference
             total_count += count
             if codon in optimal_codons: optimal_count += count
    return optimal_count / total_count if total_count > 0 else np.nan

def calculate_rcdi(codon_counts: dict | Counter, reference_weights: dict) -> float:
    """ Calculates Relative Codon Deoptimization Index (RCDI). """
    if not reference_weights or not codon_counts: 
        return np.nan
    
    log_inv_weights_sum = 0.0
    total_codons_in_calc = 0
    for codon, count in codon_counts.items():
        if count <= 0: 
            continue
        weight = reference_weights.get(codon)
        if weight is not None:
            if weight > 1e-9: 
                log_inv_weights_sum += (-math.log(weight)) * count; total_codons_in_calc += count
            elif weight <= 1e-9: 
                return np.nan # Weight=0 -> undefined RCDI
    if total_codons_in_calc == 0: 
        return np.nan
    try: 
        rcdi = math.exp(log_inv_weights_sum / total_codons_in_calc)
    except OverflowError: 
        rcdi = np.inf
    return rcdi if np.isfinite(rcdi) else np.nan


# === Main Analysis Orchestration ===

def analyze_single_sequence(record: 'SeqRecord', genetic_code_id: int, reference_weights: dict) -> tuple[dict | None, dict | None]:
    """ Performs analysis for a single cleaned sequence record. """
    try: 
        genetic_code = get_genetic_code(genetic_code_id)
    except Exception as e: 
        return None, None
    seq_id = record.id
    seq_str = str(record.seq)
    seq_len = len(seq_str)
    results = {'ID': seq_id, 'Length': seq_len}

    # Codon Counts
    codon_counts_seq = Counter()
    total_codons_seq = 0
    for i in range(0, seq_len, 3):
        codon = seq_str[i:i+3]
        if all(base in VALID_CODON_CHARS for base in codon):
            codon_counts_seq[codon] += 1
            total_codons_seq += 1
    results['TotalCodons'] = total_codons_seq

    # Fill NaNs if no codons
    nan_metrics = {'GC': np.nan, 'GC1': np.nan, 'GC2': np.nan, 'GC3': np.nan, 'GC12': np.nan,
                   'ENC': np.nan, 'CAI': np.nan, 'Fop': np.nan, 'RCDI': np.nan,
                   'GRAVY': np.nan, 'Aromaticity': np.nan, 'ProteinLength': 0}
    if total_codons_seq == 0: 
        results.update(nan_metrics)
        return results, None
    
    # Calculate Metrics
    results['GC'], results['GC1'], results['GC2'], results['GC3'], results['GC12'] = calculate_gc_content(seq_str)
    protein_seq = translate_sequence(seq_str, genetic_code)
    results['ProteinLength'] = len(protein_seq.replace('*','').replace('X','').replace('?','')) if protein_seq else 0
    results['GRAVY'], results['Aromaticity'] = calculate_protein_properties(protein_seq)
    results['ENC'] = calculate_enc(codon_counts_seq, genetic_code_id)
    results['CAI'] = calculate_cai(codon_counts_seq, reference_weights) if reference_weights else np.nan
    results['Fop'] = calculate_fop(codon_counts_seq, reference_weights) if reference_weights else np.nan
    results['RCDI'] = calculate_rcdi(codon_counts_seq, reference_weights) if reference_weights else np.nan
    
    return results, codon_counts_seq


# --- Main Analysis Function (Modified for Multiprocessing) ---

def run_full_analysis(sequences: list['SeqRecord'], genetic_code_id: int = 1,
                      reference_weights: dict | None = None,
                      num_threads: int = 1, perform_ca: bool = True) -> tuple:
    """
    Performs all analyses using multiprocessing for per-sequence calculations.
    Relies on pre-loaded reference_weights for CAI/Fop/RCDI.
    Optionally skips the final CA fitting step.

    Args:
        sequences (list[SeqRecord]): Input sequences for a single gene/dataset.
        genetic_code_id (int): NCBI genetic code ID.
        reference_weights (dict | None): Pre-loaded dictionary mapping codons
                                         to reference weights (w). Default None.
        num_threads (int): Number of parallel processes.
        perform_ca (bool): If True, perform Correspondence Analysis fitting.

    Returns:
        tuple: Contains:
            - pd.DataFrame: Aggregate codon usage table (Counts, Freq, RSCU).
            - pd.DataFrame: Per-sequence metrics table.
            - dict: Overall nucleotide frequencies for this input set.
            - dict: Overall dinucleotide frequencies for this input set.
            - pd.DataFrame or None: *No longer returns reference_data*. Kept None for consistent tuple size.
            - prince.CA or None: Fitted CA object (if perform_ca is True).
            - pd.DataFrame or None: CA input data (per-sequence RSCU).
    """
    # --- Setup ---
    genetic_code = get_genetic_code(genetic_code_id)
    overall_nucl_freqs, _ = calculate_nucleotide_frequencies(sequences)
    overall_dinucl_freqs, _ = calculate_dinucleotide_frequencies(sequences)

    # --- Per-Sequence Analysis (Parallelized) ---
    analysis_func = partial(analyze_single_sequence, genetic_code_id=genetic_code_id, reference_weights=reference_weights)
    all_results = []
    effective_threads = max(1, num_threads) if isinstance(num_threads, int) else 1
    if effective_threads > 1 and len(sequences) > 1 and mp: # Check if mp imported
        try: # Add try block for pool
            with mp.Pool(processes=effective_threads) as pool:
                all_results = pool.map(analysis_func, sequences)
        except Exception as e:
             print(f"Multiprocessing error: {e}. Falling back to sequential.", file=sys.stderr)
             all_results = [analysis_func(seq) for seq in sequences]
    else:
         all_results = [analysis_func(seq) for seq in sequences]

    # Process results
    per_sequence_metrics_list = []; aggregate_codon_counts = Counter(); sequences_for_ca = {}
    for result_tuple in all_results:
        if result_tuple: # Check not None
             metrics_dict, counts_dict = result_tuple
             if metrics_dict: per_sequence_metrics_list.append(metrics_dict)
             if counts_dict:
                  aggregate_codon_counts.update(counts_dict)
                  seq_id = metrics_dict.get('ID')
                  if seq_id: # Calculate and store RSCU vector for CA
                      temp_counts_df = pd.DataFrame.from_dict(counts_dict, orient='index', columns=['Count']); temp_counts_df.index.name = 'Codon'
                      temp_rscu_df = calculate_rscu(temp_counts_df, genetic_code_id=genetic_code_id)
                      if not temp_rscu_df.empty and not temp_rscu_df['RSCU'].isnull().all():
                           sequences_for_ca[seq_id] = temp_rscu_df.set_index('Codon')['RSCU'].fillna(0)

    # Check for results
    if not per_sequence_metrics_list:
         empty_df = pd.DataFrame(); return empty_df, empty_df, overall_nucl_freqs, overall_dinucl_freqs, None, None, None

    per_sequence_df = pd.DataFrame(per_sequence_metrics_list)

    # --- Aggregate Codon Usage ---
    agg_usage_df = pd.DataFrame()
    if aggregate_codon_counts:
        agg_counts_df = pd.DataFrame.from_dict(aggregate_codon_counts, orient='index', columns=['Count']); agg_counts_df.index.name = 'Codon'; agg_counts_df.sort_index(inplace=True)
        agg_usage_df = calculate_rscu(agg_counts_df, genetic_code_id=genetic_code_id)

    # --- CA Data Prep ---
    ca_input_df = None
    if sequences_for_ca:
        try:
            ca_input_df = pd.DataFrame.from_dict(sequences_for_ca, orient='index')
            all_codons = sorted([c for c, aa in genetic_code.items() if aa != '*'])
            ca_input_df = ca_input_df.reindex(columns=all_codons, fill_value=0.0)
            ca_input_df.fillna(0.0, inplace=True); ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True)
            ca_input_df = ca_input_df.loc[ca_input_df.sum(axis=1).abs() > 1e-9]; ca_input_df = ca_input_df.loc[:, ca_input_df.sum(axis=0).abs() > 1e-9]
            if ca_input_df.empty: ca_input_df = None
        except Exception as e: ca_input_df = None; print(f"Error preparing CA data: {e}", file=sys.stderr)

    # --- Perform CA ---
    ca_results = None
    if perform_ca and ca_input_df is not None:
        ca_results = perform_ca(ca_input_df)

    # --- Return ---
    return (agg_usage_df, per_sequence_df, overall_nucl_freqs, overall_dinucl_freqs, None, ca_results, ca_input_df)



# === Statistical Comparison Function ===
def compare_features_between_genes(combined_per_sequence_df: pd.DataFrame,
                                   features: list[str],
                                   method: str = 'kruskal') -> pd.DataFrame | None:
    """
    Performs statistical tests to compare features between gene groups.

    Args:
        combined_per_sequence_df (pd.DataFrame): DataFrame containing per-sequence
                                                 metrics including a 'Gene' column.
        features (list[str]): List of column names (features) to compare.
        method (str): Statistical test method ('kruskal' for Kruskal-Wallis H-test
                      or 'anova' for one-way ANOVA). Default is 'kruskal'.

    Returns:
        pd.DataFrame or None: DataFrame summarizing the test results (Feature,
                              Test Statistic, P-value) or None if stats cannot be run.
    """
    if stats is None:
        print("Warning: scipy.stats module not found. Cannot perform statistical comparisons.", file=sys.stderr)
        return None
    if combined_per_sequence_df is None or combined_per_sequence_df.empty:
        print("Warning: Input data empty. Cannot perform statistical comparisons.", file=sys.stderr)
        return None
    if 'Gene' not in combined_per_sequence_df.columns:
        print("Warning: 'Gene' column missing. Cannot perform statistical comparisons.", file=sys.stderr)
        return None

    results = []
    valid_genes = combined_per_sequence_df['Gene'].unique()
    if len(valid_genes) < 2:
        print("Warning: Need at least two gene groups for comparison.", file=sys.stderr)
        return None

    for feature in features:
        if feature not in combined_per_sequence_df.columns:
            print(f"Warning: Feature '{feature}' not found in data. Skipping comparison.", file=sys.stderr)
            continue

        # Prepare data for the test: list of arrays/Series, one per gene group
        # Ensure data is numeric and drop NaNs for the specific feature
        feature_data = combined_per_sequence_df[['Gene', feature]].copy()
        feature_data[feature] = pd.to_numeric(feature_data[feature], errors='coerce')
        feature_data = feature_data.dropna()

        groups_data = [group[feature] for name, group in feature_data.groupby('Gene')]

        # Check if we have enough data in enough groups
        valid_groups_data = [g for g in groups_data if len(g) > 0] # Groups with at least one data point
        if len(valid_groups_data) < 2:
            print(f"Warning: Not enough gene groups with valid data for feature '{feature}'. Skipping comparison.", file=sys.stderr)
            continue

        statistic = np.nan
        p_value = np.nan
        test_name = "N/A"

        try:
            if method == 'kruskal':
                test_name = "Kruskal-Wallis H"
                statistic, p_value = stats.kruskal(*valid_groups_data)
            elif method == 'anova':
                test_name = "One-way ANOVA F"
                statistic, p_value = stats.f_oneway(*valid_groups_data)
            else:
                print(f"Warning: Unknown statistical method '{method}'. Skipping comparison for '{feature}'.", file=sys.stderr)
                continue

        except ValueError as ve: # Handle errors like non-numeric data if coercion failed subtly
            print(f"Warning: Statistical test failed for feature '{feature}'. Error: {ve}. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"Error during statistical test for feature '{feature}': {e}", file=sys.stderr)

        results.append({
            'Feature': feature,
            'Test': test_name,
            'Statistic': statistic,
            'P_value': p_value
        })

    if not results:
        return None
    return pd.DataFrame(results)

# === CA Fitting Function ===
def perform_ca(ca_input_df, n_components=10):
    """
    Performs Correspondence Analysis on the input DataFrame.

    Args:
        ca_input_df (pd.DataFrame): DataFrame with sequences as index, codons as columns,
                                   and RSCU (or counts) as values.
        n_components (int): Number of components for CA.

    Returns:
        prince.CA or None: Fitted CA object, or None if CA fails or library missing.
    """
    if prince is None:
        print("Warning: 'prince' library not installed. Cannot perform CA.", file=sys.stderr)
        return None
    if ca_input_df is None or ca_input_df.empty:
         print("Warning: No input data provided for CA.", file=sys.stderr)
         return None

    try:
        # Ensure data is numeric and handle potential NaNs/Infs if not already done
        ca_input_df = ca_input_df.fillna(0.0) # Fill NaNs first
        ca_input_df.replace([np.inf, -np.inf], 0.0, inplace=True) # Replace Inf

        # Filter zero variance rows/columns
        ca_input_df_filtered = ca_input_df.loc[ca_input_df.sum(axis=1).abs() > 1e-9]
        ca_input_df_filtered = ca_input_df_filtered.loc[:, ca_input_df_filtered.sum(axis=0).abs() > 1e-9]


        if ca_input_df_filtered.shape[0] < 2 or ca_input_df_filtered.shape[1] < 2:
            print("Warning: Not enough data (rows/columns >= 2) remaining for CA after filtering.", file=sys.stderr)
            return None

        # Determine number of components dynamically
        actual_n_components = min(n_components,
                                  ca_input_df_filtered.shape[0] - 1,
                                  ca_input_df_filtered.shape[1] - 1)

        if actual_n_components < 1:
            print("Warning: Calculated number of components for CA is less than 1.", file=sys.stderr)
            return None

        ca = prince.CA(n_components=actual_n_components, n_iter=10, random_state=42, copy=True)
        # --- Correction: Use the filtered DataFrame ---
        ca_fitted = ca.fit(ca_input_df_filtered)
        return ca_fitted

    except Exception as e:
        print(f"Error during CA calculation: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc() # Uncomment for more detailed debug info
        return None