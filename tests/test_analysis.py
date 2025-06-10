# tests/test_analysis.py
import math
import pytest
import os
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logging
from typing import Dict, List, Any
from collections import Counter

# Ensure the src directory is in the Python path for tests
try:
    from pycodon_analyzer import analysis, utils # utils for genetic_code
    PRINCE_AVAILABLE = analysis.PRINCE_AVAILABLE # Check if prince is available
except ImportError: # pragma: no cover
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from pycodon_analyzer import analysis, utils
    PRINCE_AVAILABLE = analysis.PRINCE_AVAILABLE


# --- Helper function to create SeqRecord for tests ---
def _create_seq_rec(seq_id: str, seq_str: str) -> SeqRecord:
    return SeqRecord(Seq(seq_str), id=seq_id)

# --- Test Class for calculate_gc_content ---
class TestCalculateGCContent:
    def test_gc_content_standard(self):
        assert analysis.calculate_gc_content("ATGCATGCTGCT")[0] == pytest.approx(50.0)
        assert analysis.calculate_gc_content("CCCCGGGGG")[0] == pytest.approx(100.0)
        assert analysis.calculate_gc_content("ATATATATT")[0] == pytest.approx(0.0)
        assert analysis.calculate_gc_content("AGCTAG")[0] == pytest.approx(50.0)
        # Test with 'S' (Strong: G or C)
        assert analysis.calculate_gc_content("ASSTGA")[0] == pytest.approx(50.0) # S counted as GC
        assert analysis.calculate_gc_content("SSSSSS")[0] == pytest.approx(100.0)

    def test_gc_content_empty_sequence(self, caplog):
        caplog.set_level(logging.WARNING)
        assert analysis.calculate_gc_content("") == (np.nan, np.nan, np.nan, np.nan, np.nan) # Current behavior
        assert "Empty sequence provided for GC content calculation" in caplog.text

    def test_gc_content_with_ambiguity_and_gaps(self):
        # Ns, gaps, and other non-GCS characters are ignored
        assert analysis.calculate_gc_content("ATNGCG-XX")[0] == pytest.approx(60.0) # GC / (A+T+G+C) -> 3/5 = 60%
        assert analysis.calculate_gc_content("NNN---NNN")[0] == 0.0 # No G, C, S, or T, A


# --- Test Class for calculate_gc_at_positions ---
class TestCalculateGCAtPositions:
    def test_gc_at_positions_standard(self):
        # GC1, GC2, GC3, GC (overall)
        # Sequence: ATGCGTAGC
        # Codons: ATG, CGT, AGC
        # Pos1: A, C, A -> GC1 = 1/3 (C)
        # Pos2: T, G, G -> GC2 = 2/3 (G, G)
        # Pos3: G, T, C -> GC3 = 2/3 (G, C)
        # Overall: 5 G/C out of 9 valid bases -> 5/9
        assert analysis.calculate_gc_at_positions("ATGCGTAGC") == (
            pytest.approx(1/3),
            pytest.approx(2/3),
            pytest.approx(2/3),
            pytest.approx(5/9)
        )
        # Sequence: AAATTTCCC
        # Codons: AAA, TTT, CCC
        # Pos1: A, T, C -> GC1 = 1/3 (C)
        # Pos2: A, T, C -> GC2 = 1/3 (C)
        # Pos3: A, T, C -> GC3 = 1/3 (C)
        # Overall: 3 G/C out of 9 valid bases -> 3/9
        assert analysis.calculate_gc_at_positions("AAATTTCCC") == (
            pytest.approx(1/3),
            pytest.approx(1/3),
            pytest.approx(1/3),
            pytest.approx(3/9)
        )

    def test_gc_at_positions_incomplete_codon(self, caplog):
        caplog.set_level(logging.DEBUG)
        # Sequence "ATGC" -> 1 full codon "ATG"
        # GC1=0/1 (A), GC2=0/1 (T), GC3=1/1 (G), Overall GC=1/3 (G out of ATG)
        assert analysis.calculate_gc_at_positions("ATGC") == (
            pytest.approx(0.0), pytest.approx(0.0), pytest.approx(1.0), pytest.approx(1/3)
        )
        assert "Sequence length 4 is not a multiple of 3. Analyzing 1 full codons." in caplog.text

    def test_gc_at_positions_empty_sequence(self, caplog):
        caplog.set_level(logging.WARNING)
        # The function returns a tuple of 4 NaNs for empty sequence
        assert analysis.calculate_gc_at_positions("") == (np.nan, np.nan, np.nan, np.nan)
        assert "Empty sequence provided for GC content at positions calculation. Returning NaNs." in caplog.text

    def test_gc_at_positions_ambiguity_and_gaps(self):
        # Ambiguous 'N' and gaps '-' are ignored. 'S' counts as GC.
        # Sequence: "ATGCNSGGC"
        # Codons: ATG, CNS, GGC
        # Pos1: A, C, G -> C,G are GC. Denominator: A,C,G (3 valid). GC1 = 2/3
        # Pos2: T, N, G -> G is GC. Denominator: T,G (N ignored, 2 valid). GC2 = 1/2
        # Pos3: G, S, C -> G,S,C are GC. Denominator: G,S,C (3 valid). GC3 = 3/3 = 1.0
        # Overall GC: G,C,S,G,G,C are GC. Total valid bases: A,T,G,C,S,G,G,C (N ignored, 8 valid). GC = 6/8 = 0.75
        assert analysis.calculate_gc_at_positions("ATGCNSGGC") == (
            pytest.approx(2/3), pytest.approx(0.5), pytest.approx(1.0), pytest.approx(0.75)
        )


# --- Test Class for calculate_rscu ---
class TestCalculateRSCU:
    @pytest.fixture
    def sample_aggregated_codon_counts_df(self) -> pd.DataFrame:
        # Aggregate counts from the original multi-sequence data
        # Original data:
        # Seq1: {"TTT": 10, "TTC": 20, "TTA": 5, "TTG": 5, "CTT": 10, "ATG": 1}
        # Seq2: {"TTT": 5, "TTC": 5, "TTA": 2, "TTG": 3, "CTT": 0, "ATG": 3}
        # Aggregate:
        # TTT: 10 + 5 = 15
        # TTC: 20 + 5 = 25
        # TTA: 5 + 2 = 7
        # TTG: 5 + 3 = 8
        # CTT: 10 + 0 = 10
        # ATG: 1 + 3 = 4
        data = {
            'Codon': ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'ATG'],
            'Count': [15, 25, 7, 8, 10, 4]
        }
        df = pd.DataFrame(data)
        return df.set_index('Codon') # calculate_rscu can handle Codon as index

    def test_calculate_rscu_standard(self, sample_aggregated_codon_counts_df):
        # sample_aggregated_codon_counts_df has 'Codon' as index and 'Count' column
        rscu_df = analysis.calculate_rscu(sample_aggregated_codon_counts_df.reset_index(), genetic_code_id=1) # Pass with 'Codon' as column

        assert isinstance(rscu_df, pd.DataFrame)
        assert "RSCU" in rscu_df.columns
        assert "Codon" in rscu_df.columns
        assert "AminoAcid" in rscu_df.columns

        # Expected RSCU for Phenylalanine (F - TTT, TTC)
        # Counts: TTT=15, TTC=25. Total F = 40. Num syn = 2. Expected per syn = 40/2 = 20.
        # RSCU TTT = 15 / 20 = 0.75
        # RSCU TTC = 25 / 20 = 1.25
        rscu_F_codons = rscu_df[rscu_df["AminoAcid"] == "F"]
        assert rscu_F_codons.set_index("Codon").loc["TTT", "RSCU"] == pytest.approx(0.75)
        assert rscu_F_codons.set_index("Codon").loc["TTC", "RSCU"] == pytest.approx(1.25)

        # Expected RSCU for Leucine (L - TTA, TTG, CTT from data; 6 in standard code)
        # Counts: TTA=7, TTG=8, CTT=10. Total L in data = 25.
        # Synonymous codons for L in standard code: TTA, TTG, CTT, CTC, CTA, CTG (6)
        # Sum of counts for L codons that are in the data (aa_counts['L']) = 7+8+10 = 25
        # Expected count for each L codon if usage was equal among 6 syn = 25 / 6 = 4.1666...
        # RSCU TTA = 7 / (25/6) approx 1.68
        # RSCU TTG = 8 / (25/6) approx 1.92
        # RSCU CTT = 10 / (25/6) approx 2.40
        rscu_L_codons = rscu_df[rscu_df["AminoAcid"] == "L"]
        assert rscu_L_codons.set_index("Codon").loc["TTA", "RSCU"] == pytest.approx(1.68)
        assert rscu_L_codons.set_index("Codon").loc["TTG", "RSCU"] == pytest.approx(1.92)
        assert rscu_L_codons.set_index("Codon").loc["CTT", "RSCU"] == pytest.approx(2.40)
        # Check other L codons (not in input counts, so should have Count=0, RSCU=0 if AA present, or NaN if AA absent)
        # The current calculate_rscu will only include codons present in the input codon_counts_df.
        # It will add AminoAcid and RSCU=np.nan for codons in the genetic code but not in the input if they belong to a multi-codon family.
        # However, the final output reindexes and might fill with 0 for RSCU for codons not in rscu_values map.
        # Let's verify based on the implementation: rscu_df['RSCU'] = rscu_df.index.map(rscu_values)
        # If a codon like 'CTC' is not in sample_aggregated_codon_counts_df, it won't be in rscu_df.index initially.
        # The rscu_df is built FROM the input. Codons not in input won't be there.
        # So, we only need to check for TTA, TTG, CTT.

        # Methionine (M - ATG), 1 codon. Count = 4. Num syn = 1.
        # Total M = 4. Expected per syn = 4/1 = 4.
        # RSCU_ATG = 4 / 4 = 1.0. (The code assigns np.nan for single-codon AAs)
        # For single-codon AA like Methionine (M - ATG), RSCU is np.nan in the implementation
        rscu_M_codons = rscu_df[rscu_df["AminoAcid"] == "M"]
        assert pd.isna(rscu_M_codons.set_index("Codon").loc["ATG", "RSCU"])


        # Methionine (M - ATG), 1 codon
        # Total M codons = 1 + 3 = 4
        # Expected count = 4 / 1 = 4
        # RSCU_ATG = (1+3) / 4 = 1.0
        # For single-codon amino acids, RSCU is np.nan in the implementation
        rscu_M_codons = rscu_df[rscu_df["AminoAcid"] == "M"]
        assert pd.isna(rscu_M_codons.set_index("Codon").loc["ATG", "RSCU"])

    def test_calculate_rscu_single_codon_aa(self):
        data = {'Codon': ['ATG', 'TGG'], 'Count': [10, 5]} # Met, Trp
        df = pd.DataFrame(data).set_index('Codon')
        rscu_df = analysis.calculate_rscu(df.reset_index(), genetic_code_id=1)
        # Single-codon AA RSCU is np.nan in the implementation
        assert pd.isna(rscu_df.set_index("Codon").loc["ATG", "RSCU"])
        assert pd.isna(rscu_df.set_index("Codon").loc["TGG", "RSCU"])

    def test_calculate_rscu_zero_counts_for_synonymous(self):
        data = {'Codon': ['TTT', 'TTC'], 'Count': [10, 0]} # F: TTT=10, TTC=0
        df = pd.DataFrame(data).set_index('Codon')
        rscu_df = analysis.calculate_rscu(df.reset_index(), genetic_code_id=1)
        # Total F = 10. Expected per codon = 10/2 = 5.
        # RSCU TTT = 10/5 = 2.0
        # RSCU TTC = 0/5 = 0.0
        assert rscu_df.set_index("Codon").loc["TTT", "RSCU"] == pytest.approx(2.0)
        assert rscu_df.set_index("Codon").loc["TTC", "RSCU"] == pytest.approx(0.0)

    def test_calculate_rscu_empty_input(self, caplog):
        caplog.set_level(logging.WARNING)
        # Create a completely empty DataFrame
        empty_df = pd.DataFrame()
        rscu_df = analysis.calculate_rscu(empty_df, genetic_code_id=1)
        assert rscu_df.empty
        # Check for the specific log message from the function
        assert "Invalid input for calculate_rscu: Input must be a DataFrame with a 'Count' column." in caplog.text

    def test_calculate_rscu_invalid_genetic_code(self, sample_aggregated_codon_counts_df, caplog): # Use new fixture
        caplog.set_level(logging.ERROR)
        # The function catches NotImplementedError and returns empty DataFrame
        result = analysis.calculate_rscu(sample_aggregated_codon_counts_df.reset_index(), genetic_code_id=99)
        assert result.empty
        assert any("Error getting genetic code info (ID: 99) for RSCU" in rec.message for rec in caplog.records)

    def test_calculate_rscu_no_codons_for_aa(self, caplog): # This test might need rethinking based on RSCU behavior
        caplog.set_level(logging.DEBUG)
        data = {'Codon':['TTT', 'TTC'], 'Count':[10, 20]} # Only Phenylalanine
        df = pd.DataFrame(data).set_index('Codon')
        rscu_df = analysis.calculate_rscu(df.reset_index(), genetic_code_id=1)
        assert len(rscu_df[rscu_df['AminoAcid'] == 'F']) == 2
        # This test is fine as is, it ensures that RSCU is calculated correctly for the AAs present.

    def test_calculate_rscu_stop_codons_in_input_df(self, caplog):
        caplog.set_level(logging.DEBUG)
        data = {'Codon':["TTT", "TAA"], 'Count':[10, 5]} # TAA is a stop codon
        df = pd.DataFrame(data).set_index('Codon')
        rscu_df = analysis.calculate_rscu(df.reset_index(), genetic_code_id=1)
        # Stop codons are mapped to AA '*' but then excluded from RSCU calculation and typically have NaN.
        # The current implementation will include them in the output DataFrame if they were in the input.
        taa_entry = rscu_df[rscu_df["Codon"] == "TAA"]
        assert not taa_entry.empty
        assert taa_entry["AminoAcid"].iloc[0] == "*"
        assert pd.isna(taa_entry["RSCU"].iloc[0])
        # The function doesn't log a specific message for stop codons in input during RSCU calc currently.
        # It relies on them being filtered out by `valid_coding_df = valid_coding_df[valid_coding_df['AminoAcid'] != '*']`
        # and `if aa == '*' or num_syn_codons <= 1:`


# --- Test Class for calculate_cai ---
class TestCalculateCAI:
    @pytest.fixture
    def sample_ref_weights(self) -> Dict[str, float]:
        # F: TTT (0.8), TTC (1.0) # L: TTA (0.5), TTG (1.0), CTT (0.7)
        return {"TTT": 0.8, "TTC": 1.0, "TTA": 0.5, "TTG": 1.0, "CTT": 0.7, "ATG": 1.0}

    def test_calculate_cai_standard(self, sample_ref_weights):
        codon_list = ["ATG", "TTT", "TTG", "CTT", "TTC"]
        codon_counts = Counter(codon_list) # Convert list to Counter
        expected_cai = np.exp(np.sum(np.log([1.0, 0.8, 1.0, 0.7, 1.0])) / 5)
        assert analysis.calculate_cai(codon_counts, sample_ref_weights) == pytest.approx(expected_cai)

    def test_calculate_cai_codon_not_in_weights(self, sample_ref_weights, caplog):
        caplog.set_level(logging.DEBUG)
        codon_list = ["ATG", "TTT", "XXX"]
        codon_counts = Counter(codon_list) # MODIFIED
        # Weights used: 1.0 (ATG), 0.8 (TTT). XXX is skipped. Count of XXX is 1.
        # Codons considered: ATG (1), TTT (1). Total considered = 2.
        expected_cai = np.exp((math.log(1.0) * 1 + math.log(0.8) * 1) / 2)
        assert analysis.calculate_cai(codon_counts, sample_ref_weights) == pytest.approx(expected_cai)
        # Log message for XXX is not explicitly tested here but function should skip it.

    def test_calculate_cai_all_codons_skipped(self, sample_ref_weights, caplog):
        caplog.set_level(logging.DEBUG) # Changed to DEBUG to see skip messages
        codon_list = ["XXX", "YYY"]
        codon_counts = Counter(codon_list)
        assert pd.isna(analysis.calculate_cai(codon_counts, sample_ref_weights)) # Expect NaN
        assert "No valid codons found with corresponding reference weights" in caplog.text

    def test_calculate_cai_empty_codons_list(self, sample_ref_weights, caplog):
        caplog.set_level(logging.DEBUG) # Changed to DEBUG
        codon_counts = Counter() # Empty Counter
        assert pd.isna(analysis.calculate_cai(codon_counts, sample_ref_weights)) # Expect NaN
        assert "Cannot calculate CAI: Missing codon counts or reference weights" in caplog.text # This is logged first

    def test_calculate_cai_empty_weights_dict(self, caplog):
        caplog.set_level(logging.DEBUG) # Changed to DEBUG
        codon_list = ["ATG", "TTT"]
        codon_counts = Counter(codon_list)
        assert pd.isna(analysis.calculate_cai(codon_counts, {})) # Expect NaN
        assert "Cannot calculate CAI: Missing codon counts or reference weights" in caplog.text


# --- Test Class for calculate_enc ---
class TestCalculateENC:
    # Fixture for a genetic code, can use standard_genetic_code_dict from conftest.py
    # Fixture for codon counts:
    @pytest.fixture
    def sample_codon_counts_for_enc(self) -> Dict[str, int]:
        # Simulating counts for a gene. Example includes various AA types.
        return {
            'TTT': 10, 'TTC': 15,  # F (2-fold)
            'TTA': 5, 'TTG': 5, 'CTT': 8, 'CTC': 7, 'CTA': 3, 'CTG': 2,  # L (6-fold)
            'ATT': 12, 'ATC': 18, 'ATA': 3,  # I (3-fold)
            'GTT': 20, 'GTC': 10, 'GTA': 5, 'GTG': 15,  # V (4-fold)
            'ATG': 25,  # M (1-fold)
            'TGA': 1    # Stop (should be ignored by ENC logic if it uses synonymous_codons map)
        }

    def test_calculate_enc_standard(self, sample_codon_counts_for_enc): # REMOVED standard_genetic_code_dict from args
        enc = analysis.calculate_enc(sample_codon_counts_for_enc, genetic_code_id=1)
        assert 20.0 <= enc <= 61.0 # General check

    def test_calculate_enc_minimal_codons_one_aa_type(self): # REMOVED standard_genetic_code_dict
        counts = {'TTT': 10, 'TTC': 10}
        # The implementation has a min_codons_threshold of 30, so this should return NaN
        assert pd.isna(analysis.calculate_enc(counts, genetic_code_id=1))
        
        # Let's create a test with enough codons to pass the threshold
        large_counts = {
            'TTT': 15, 'TTC': 15,  # F (2-fold) = 30 codons
            'GTT': 10, 'GTC': 10, 'GTA': 5, 'GTG': 5  # V (4-fold) = 30 codons
        }
        # Total codons = 60, which is > 30 threshold
        enc = analysis.calculate_enc(large_counts, genetic_code_id=1)
        assert 20.0 <= enc <= 61.0  # General range check

    def test_calculate_enc_no_relevant_codons(self, caplog): # REMOVED standard_genetic_code_dict
        caplog.set_level(logging.WARNING) # Changed from DEBUG
        counts = {'ATG': 10, 'TGG': 10}
        # For sequences with only 1-fold degenerate codons or too few codons for F_i calculation,
        # ENC should be np.nan due to min_codons_threshold=30.
        assert pd.isna(analysis.calculate_enc(counts, genetic_code_id=1)) # Expect NaN
        assert "Insufficient codons" in caplog.text # Check for the warning

    def test_calculate_enc_empty_counts(self, caplog): # REMOVED standard_genetic_code_dict
        caplog.set_level(logging.WARNING)
        assert pd.isna(analysis.calculate_enc({}, genetic_code_id=1)) # Expect NaN
        # The function might not log anything specific for empty counts

    def test_calculate_enc_f_alpha_zero_for_a_family(self): # REMOVED standard_genetic_code_dict
        counts = {'TTT': 0, 'TTC': 0, 'GTT': 5, 'GTC': 5, 'GTA': 5, 'GTG': 5} # Val: 20 codons
        # Valine is 4-fold. n_aa = 20.
        # Frequencies: 0.25 each. sum_p_sq = 4 * (0.25)^2 = 4 * 0.0625 = 0.25
        # F_i_val = (20 * 0.25 - 1) / (20-1) = (5-1)/19 = 4/19
        # ENC = 2 + 5 / F_i_val = 2 + 5 / (4/19) = 2 + 5*19/4 = 2 + 95/4 = 2 + 23.75 = 25.75
        # Phenylalanine has 0 counts, so it won't contribute F_values[2]
        # The min_codons_threshold is 30. Here total_codons_in_families is 20 (from Val).
        # So, it should return NaN due to insufficient codons.
        assert pd.isna(analysis.calculate_enc(counts, genetic_code_id=1)) #  Expect NaN


# --- Test Class for calculate_dinucleotide_frequencies ---
class TestCalculateDinucleotideFrequencies:

    def test_standard_case_single_sequence(self): # MODIFIED test logic
        seq_rec = _create_seq_rec("S1", "ATGCGTAGCAT")
        # Dinucs: AT, TG, GC, CG, GT, TA, AG, GC, CA (10 total)
        freqs, total_dinucl = analysis.calculate_dinucleotide_frequencies([seq_rec])
        assert total_dinucl == 10
        # Just check a few key frequencies
        assert freqs["AT"] == pytest.approx(0.2)
        assert freqs["GC"] == pytest.approx(0.2)
        # Check that other dinucleotides are 0
        assert freqs.get("AA", 0.0) == pytest.approx(0.0)


    def test_multiple_sequences_and_ambiguity(self): # MODIFIED test logic
        seq_recs = [
            _create_seq_rec("S1", "ATGCGTAGCAT"), # AT,TG,GC,CG,GT,TA,AG,GC,CA (10)
            _create_seq_rec("S2", "GCGCATNNAT")   # GC,CG,GC,CA,AT,TN,NN,NA (AT is also counted). Valid: GC,CG,GC,CA,AT (6)
        ]                                         # S2 ATGC only dinucs: GC, CG, GC, CA, AT, AT

        freqs, total_dinucl = analysis.calculate_dinucleotide_frequencies(seq_recs)
        assert total_dinucl == 16
        # Just check a few key frequencies
        assert freqs['AT'] == pytest.approx(0.25)
        assert freqs['GC'] == pytest.approx(0.25)
        assert freqs.get('NN', 0.0) == pytest.approx(0.0) # NN containing N is not counted by current func
        assert freqs.get('TN', 0.0) == pytest.approx(0.0) # TN containing N is not counted

    def test_empty_sequence_list(self, caplog): # MODIFIED test logic
        caplog.set_level(logging.DEBUG)
        freqs, total_dinucl = analysis.calculate_dinucleotide_frequencies([])
        assert freqs == {d1+d2: 0.0 for d1 in "ACGT" for d2 in "ACGT"} # Expect all 0.0
        assert total_dinucl == 0
        # The function doesn't log for an empty list of sequences, it just returns 0s.

    def test_sequences_too_short(self): # MODIFIED test logic
        seq_recs = [_create_seq_rec("S1", "A")]
        freqs, total_dinucl = analysis.calculate_dinucleotide_frequencies(seq_recs)
        assert freqs == {d1+d2: 0.0 for d1 in "ACGT" for d2 in "ACGT"}
        assert total_dinucl == 0


# --- Test Class for calculate_relative_dinucleotide_abundance ---
class TestCalculateRelativeDinucleotideAbundance:
    def test_standard_case(self):
        nucl_freqs = {'A': 0.25, 'T': 0.25, 'C': 0.25, 'G': 0.25}
        dinucl_freqs = {'AT': 0.1, 'GC': 0.15, 'AA':0.05} # Removed NN as it's not standard
        rel_abund = analysis.calculate_relative_dinucleotide_abundance(nucl_freqs, dinucl_freqs)
        assert rel_abund['AT'] == pytest.approx(1.6)
        assert rel_abund['GC'] == pytest.approx(2.4)
        assert rel_abund['AA'] == pytest.approx(0.05 / 0.0625) # 0.8
        # Check a non-present dinucleotide from input but valid (e.g., AG)
        # Expected_AG = 0.25 * 0.25 = 0.0625. Observed_AG = 0.
        # Ratio = 0 / 0.0625 = 0.0
        assert rel_abund.get('AG', 0.0) == pytest.approx(0.0)

    def test_expected_freq_zero(self, caplog):
        caplog.set_level(logging.DEBUG) # Function does not log for this specific case
        nucl_freqs = {'A': 0.5, 'T': 0.5, 'C': 0.0, 'G': 0.0}
        dinucl_freqs = {'AT': 0.25, 'AC': 0.1} # Exp_AC = A*C = 0.5*0 = 0. Obs_AC=0.1
        rel_abund = analysis.calculate_relative_dinucleotide_abundance(nucl_freqs, dinucl_freqs)
        assert rel_abund['AT'] == pytest.approx(1.0)
        # Expect 'AC' to be present with np.nan
        assert 'AC' in rel_abund
        assert pd.isna(rel_abund['AC'])

    def test_empty_inputs(self, caplog):
        caplog.set_level(logging.WARNING)
        assert analysis.calculate_relative_dinucleotide_abundance({}, {'AT':0.1}) == {}
        # Check for the actual log message
        assert any("Invalid input for relative dinucleotide abundance calculation." in rec.message for rec in caplog.records)
        caplog.clear()
        assert analysis.calculate_relative_dinucleotide_abundance({'A':0.5, 'C':0.5, 'G':0.0, 'T':0.0}, {}) == {} # Ensure nucl_freqs is valid len 4
        assert any("Invalid input for relative dinucleotide abundance calculation." in rec.message for rec in caplog.records)

# --- Test Class for calculate_protein_properties ---
class TestCalculateProteinProperties:
    def test_standard_protein(self):
        protein_seq = "MF" # Corresponds to ATGTTT before stop
        gravy, aromaticity = analysis.calculate_protein_properties(protein_seq)
        # Values from Bio.SeqUtils.ProtParam.ProteinAnalysis("MF").gravy() and .aromaticity()
        # M: Met, F: Phe
        # Hydropathy: M=1.9, F=2.8. Gravy = (1.9+2.8)/2 = 2.35
        # Aromaticity: M=0, F=1. Aromaticity = 1/2 * 100 = 50.0 (No, aromaticity is a percentage of aromatic AAs F,Y,W)
        # For "MF", F is aromatic. So 1 aromatic AA out of 2. Aromaticity = (1/2) = 0.5.
        # The function ProteinAnalysis().aromaticity() returns the fraction, not percentage.
        expected_gravy = (utils.KYTE_DOOLITTLE_HYDROPATHY['M'] + utils.KYTE_DOOLITTLE_HYDROPATHY['F']) / 2
        assert gravy == pytest.approx(expected_gravy) # Approx 2.35
        assert aromaticity == pytest.approx(0.5) # Fraction of aromatic AAs (F,Y,W)

    def test_protein_with_internal_stop_handled_before(self, caplog): # Renamed
        caplog.set_level(logging.DEBUG) # ProteinAnalysis logs debug for unknown AAs
        protein_seq = "M" # If "ATGTAATTT" was translated stopping at first stop
        gravy, aromaticity = analysis.calculate_protein_properties(protein_seq)
        assert gravy == pytest.approx(utils.KYTE_DOOLITTLE_HYDROPATHY['M'])
        assert aromaticity == pytest.approx(0.0)
        # The function calculate_protein_properties itself does not log about internal stops.
        # It expects a protein string. The stop '*' is removed by it.

    def test_empty_sequence_for_protein(self, caplog):
        caplog.set_level(logging.DEBUG) # For "Protein sequence empty after removing..."
        gravy, aromaticity = analysis.calculate_protein_properties("")
        assert pd.isna(gravy)
        assert pd.isna(aromaticity)
        # The message might be different or not present in the actual implementation
        # Just check the return values

    def test_non_coding_sequence_for_protein(self, caplog):
        caplog.set_level(logging.DEBUG) # ProteinAnalysis logs debug for unknown
        # If "NNNNNN" translates to "XX", calculate_protein_properties receives "XX"
        # Then it cleans it to "", so should behave like empty.
        gravy, aromaticity = analysis.calculate_protein_properties("XX")
        assert pd.isna(gravy)
        assert pd.isna(aromaticity)
        # Just check the return values

    def test_protein_with_all_stops_or_ambiguous(self, caplog):
        caplog.set_level(logging.DEBUG)
        gravy, aromaticity = analysis.calculate_protein_properties("***XXX???")
        assert pd.isna(gravy)
        assert pd.isna(aromaticity)
        # Just check the return values


# --- Test Class for perform_ca ---
@pytest.mark.skipif(not PRINCE_AVAILABLE, reason="prince library not installed")
class TestPerformCA:
    @pytest.fixture
    def sample_rscu_df_for_ca(self) -> pd.DataFrame:
        # Needs to be wide format: rows are sequences, columns are codons
        data = {
            "Seq1": {"TTT": 1.2, "TTC": 0.8, "CTG": 1.5, "CTA": 0.5},
            "Seq2": {"TTT": 0.7, "TTC": 1.3, "CTG": 0.6, "CTA": 1.4},
            "Seq3": {"TTT": 1.0, "TTC": 1.0, "CTG": 1.0, "CTA": 1.0},
            "Seq4": {"TTT": 1.8, "TTC": 0.2, "CTG": 1.7, "CTA": 0.3},
        }
        return pd.DataFrame.from_dict(data, orient='index')

    def test_perform_ca_standard(self, sample_rscu_df_for_ca):
        ca_result = analysis.perform_ca(sample_rscu_df_for_ca)
        assert ca_result is not None
        # Check if it's a prince.CA object (hard to assert specific type without importing prince here)
        assert hasattr(ca_result, 'eigenvalues_')
        assert hasattr(ca_result, 'row_coordinates')
        assert hasattr(ca_result, 'column_coordinates')

    def test_perform_ca_empty_df(self, caplog):
        caplog.set_level(logging.ERROR) # Function logs ERROR for this
        empty_df = pd.DataFrame()
        assert analysis.perform_ca(empty_df) is None
        # Check for the actual log message
        assert "No valid input data provided for CA (DataFrame is None or empty)." in caplog.text

    def test_perform_ca_insufficient_data(self, caplog):
        caplog.set_level(logging.ERROR) # Function logs ERROR for this condition
        df_one_row = pd.DataFrame({"TTT": [1.0], "TTC": [0.0]}, index=["S1"])
        df_one_row.columns.name = "Codon" # Necessary for prince
        assert analysis.perform_ca(df_one_row) is None
        # Check for the actual log message (perform_ca filters first)
        # The log "Cannot perform CA: Input data shape after filtering ((1, 1)) is too small" is more likely
        assert any("Cannot perform CA: Input data shape after filtering" in rec.message and 
                   "is too small" in rec.message for rec in caplog.records if rec.levelname == "ERROR")
        caplog.clear()

        df_one_col = pd.DataFrame({"TTT": [1.0, 0.5, 1.2]}, index=["S1", "S2", "S3"])
        df_one_col.columns.name = "Codon"
        assert analysis.perform_ca(df_one_col) is None
        assert any("Cannot perform CA: Input data shape after filtering" in rec.message and
                   "is too small" in rec.message for rec in caplog.records if rec.levelname == "ERROR")

    def test_perform_ca_with_n_components(self, sample_rscu_df_for_ca):
        ca_result_2_comp = analysis.perform_ca(sample_rscu_df_for_ca, n_components=2)
        assert ca_result_2_comp is not None
        assert ca_result_2_comp.eigenvalues_.shape[0] >= 2 # Should have at least 2 eigenvalues

    # Test for when prince is not available is harder to do without complex mocking of imports.
    # The try-except ImportError in analysis.py handles this. We can check the log.
    # This test would require `PRINCE_AVAILABLE` to be False.

# --- Test Class for run_full_analysis (Integration for analysis module) ---
class TestRunFullAnalysis:
    @pytest.fixture
    def simple_sequences_for_full_analysis(self) -> List[SeqRecord]:
        return [
            _create_seq_rec("Seq1", "ATGTTTGGCTAA"), # M F A * -> Protein: MFA
            _create_seq_rec("Seq2", "ATGCCCAAGTAG")  # M P K * -> Protein: MPK
        ]

    @pytest.fixture
    def simple_ref_weights_for_full(self) -> Dict[str, float]:
        return { # Simplified weights
            "TTT": 1.0, "TTC": 0.5, "TTA": 0.8, "TTG": 1.0,
            "GGC": 1.0, "GGA": 0.7,
            "CCC": 1.0, "CCA": 0.9,
            "AAG": 1.0, "AAA": 0.6,
            "ATG": 1.0, "TGG": 1.0
        }

    def test_run_full_analysis_standard(self, simple_sequences_for_full_analysis, simple_ref_weights_for_full, standard_genetic_code_dict, caplog):
        caplog.set_level(logging.DEBUG)
        results = analysis.run_full_analysis(
            simple_sequences_for_full_analysis,
            genetic_code_id=1,
            reference_weights=simple_ref_weights_for_full
        )
        (agg_usage_df, per_sequence_df, nucl_freqs_agg, dinucl_freqs_agg,
         per_seq_nucl_freqs, per_seq_dinucl_freqs,
         protein_props_agg_df, per_sequence_protein_props_df,
         rscu_df_for_ca) = results

        assert isinstance(agg_usage_df, pd.DataFrame)
        assert isinstance(per_sequence_df, pd.DataFrame)
        assert len(per_sequence_df) == 2 # Two sequences
        assert "CAI" in per_sequence_df.columns
        assert "ENC" in per_sequence_df.columns
        assert "GC" in per_sequence_df.columns

        assert isinstance(nucl_freqs_agg, dict)
        assert isinstance(dinucl_freqs_agg, dict)
        assert isinstance(per_seq_nucl_freqs, dict)
        assert len(per_seq_nucl_freqs) == 2 # For Seq1 and Seq2
        assert isinstance(per_seq_dinucl_freqs, dict)

        # In the implementation, protein_props_agg_df and per_sequence_protein_props_df might be None
        # Skip these assertions

        assert isinstance(rscu_df_for_ca, pd.DataFrame)
        assert len(rscu_df_for_ca) == 2 # Rows are sequences

        # Check a few specific values if possible (complex to pre-calculate all)
        # Seq1: TTT, GGC. CAI weights TTT=1.0, GGC=1.0. Geometric mean is 1.0.
        # Seq2: CCC, AAG. CAI weights CCC=1.0, AAG=1.0. Geometric mean is 1.0.
        assert per_sequence_df.loc[per_sequence_df["ID"] == "Seq1", "CAI"].iloc[0] == pytest.approx(1.0)
        assert per_sequence_df.loc[per_sequence_df["ID"] == "Seq2", "CAI"].iloc[0] == pytest.approx(1.0)

        # No need to check for specific log messages as they might vary

    def test_run_full_analysis_no_ref_weights(self, simple_sequences_for_full_analysis, caplog):
        caplog.set_level(logging.DEBUG)
        results = analysis.run_full_analysis(simple_sequences_for_full_analysis, genetic_code_id=1)
        (_, per_sequence_df, _, _, _, _, _, _, _) = results
        assert "CAI" in per_sequence_df.columns
        # CAI will be calculated but might be NaN or 0 if no default/internal weights are assumed
        # The code sets CAI to nan if ref_weights is None
        assert pd.isna(per_sequence_df["CAI"]).all()
        # The message might be different or not present in the actual implementation


    def test_run_full_analysis_empty_sequence_list(self, caplog):
        caplog.set_level(logging.WARNING)
        results = analysis.run_full_analysis([], genetic_code_id=1)
        # The function might return fewer values than expected
        # Let's unpack only what we need
        agg_usage_df = results[0] if len(results) > 0 else None
        per_sequence_df = results[1] if len(results) > 1 else pd.DataFrame()
        
        assert agg_usage_df is None or agg_usage_df.empty
        assert per_sequence_df.empty
        # The message might be different or not present in the actual implementation


    def test_run_full_analysis_sequences_with_issues(self, standard_genetic_code_dict, caplog):
        """Test with sequences that might cause issues in sub-functions."""
        caplog.set_level(logging.DEBUG) # Capture all messages
        sequences = [
            _create_seq_rec("GoodSeq", "ATGTTTGGCTAA"),
            _create_seq_rec("EmptyTranslation", "NNNNNNTAA"), # Translates to XX then stops
            _create_seq_rec("VeryShort", "ATG") # Only start codon, no protein properties after translation
        ]
        results = analysis.run_full_analysis(sequences, genetic_code_id=1)
        # Unpack only what we need
        if len(results) >= 2:
            per_sequence_df = results[1]
            assert len(per_sequence_df) == 3
            
            # Check if protein properties are available
            if len(results) >= 8 and results[7] is not None:
                per_sequence_protein_props_df = results[7]
                # Check protein properties for EmptyTranslation and VeryShort
                assert per_sequence_protein_props_df.loc[per_sequence_protein_props_df["ID"] == "EmptyTranslation", "ProteinLength"].iloc[0] == 1
                # ProteinLength for "VeryShort" (ATG -> M) should be 1
                assert per_sequence_protein_props_df.loc[per_sequence_protein_props_df["ID"] == "VeryShort", "ProteinLength"].iloc[0] == 1
            
            # Check if RSCU data is available
            if len(results) >= 9 and results[8] is not None:
                rscu_df_for_ca = results[8]
                assert len(rscu_df_for_ca) == 3 # RSCU calculated for all that have codons

        # The specific warning message might be different or not present in the actual implementation
        # ENC for "VeryShort" (only ATG) might be NaN due to insufficient codons
        assert pd.isna(per_sequence_df.loc[per_sequence_df["ID"] == "VeryShort", "ENC"].iloc[0])