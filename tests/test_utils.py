# tests/test_utils.py
import pytest # type: ignore
import os
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Adjust import based on how pytest is run (from root or tests dir)
try:
    from src.pycodon_analyzer import utils # type: ignore
except ImportError:
     import sys
     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
     from pycodon_analyzer import utils


# --- Tests for cleaning function ---

def test_clean_and_filter_sequences_standard(simple_seq_records):
    """Test standard cleaning operations."""
    cleaned = utils.clean_and_filter_sequences(simple_seq_records)
    cleaned_ids = {rec.id for rec in cleaned}
    # --- Expected survivors are SeqD('TTT'), SeqE('CGA'), SeqF('ACGACG') -> No, SeqF has no start/stop -> ACGACG ---
    # Let's re-verify SeqF: ACG---ACG -> ACGACG (len 6). Start=ACG(no), Stop=ACG(no). Trimmed: ACGACG (len 6). OK.
    assert cleaned_ids == {"SeqA", "SeqB", "SeqD_short", "SeqE_nostop_ok", "SeqF_no_start"}
    # Check specific sequence transformations
    for rec in cleaned:
        if rec.id == "SeqD_short": assert str(rec.seq) == "TTT"
        if rec.id == "SeqE_nostop_ok": assert str(rec.seq) == "CGA"
        if rec.id == "SeqF_no_start": assert str(rec.seq) == "ACGACG" # Start/Stop not present, not removed

def test_clean_filter_ambiguity(simple_seq_records):
    """Test ambiguity filtering."""
    # SeqC_with_N: ATGAAAGGGNNN---TGA -> AAAGGGNNN (len=9, N=3, 33% > 15%) -> Removed
    seq_c = [rec for rec in simple_seq_records if rec.id == "SeqC_with_N"]
    cleaned_c = utils.clean_and_filter_sequences(seq_c, max_ambiguity_pct=15.0)
    assert len(cleaned_c) == 0

    # Test with higher threshold
    cleaned_c_high_thresh = utils.clean_and_filter_sequences(seq_c, max_ambiguity_pct=40.0)
    assert len(cleaned_c_high_thresh) == 1
    assert str(cleaned_c_high_thresh[0].seq) == "AAAGGGNNN" # Ns are kept if below threshold

    # Test ambiguity replacement
    # ambig_seq: ATGRYMAAATAG -> Cleaned: RYMAAA -> NNNAAA (N=3, Len=6, 50% > 15%) -> Removed
    ambig_seq = SeqRecord(Seq("ATGRYMAAATAG"), id="ambig")
    cleaned_ambig = utils.clean_and_filter_sequences([ambig_seq], max_ambiguity_pct=15.0)
    # --- Expect length 0 because N% > 15% ---
    assert len(cleaned_ambig) == 0
    # Test with higher threshold allows it
    cleaned_ambig_high_thresh = utils.clean_and_filter_sequences([ambig_seq], max_ambiguity_pct=60.0)
    assert len(cleaned_ambig_high_thresh) == 1
    assert str(cleaned_ambig_high_thresh[0].seq) == "NNNAAA"


def test_clean_filter_start_stop_logic():
    """Test conditional start/stop removal."""
    seq_nostart = SeqRecord(Seq("TTGAAATTTTAG"), id="no_start") # TTG not in START
    seq_nostop = SeqRecord(Seq("ATGAAATTTCGC"), id="no_stop") # CGC not in STOP
    seq_both = SeqRecord(Seq("ATGAAATTTTAG"), id="both") # Has ATG and TAG
    seq_neither = SeqRecord(Seq("TTGAAATTTCGC"), id="neither") # Neither start nor stop

    cleaned = utils.clean_and_filter_sequences([seq_nostart, seq_nostop, seq_both, seq_neither])
    cleaned_map = {rec.id: str(rec.seq) for rec in cleaned}

    # --- Expected sequences based on logic ---
    assert "no_start" in cleaned_map and cleaned_map["no_start"] == "TTGAAATTT" # Start TTG kept, Stop TAG removed
    assert "no_stop" in cleaned_map and cleaned_map["no_stop"] == "AAATTTCGC"   # Start ATG removed, Stop CGC kept
    assert "both" in cleaned_map and cleaned_map["both"] == "AAATTT"       # Both removed
    assert "neither" in cleaned_map and cleaned_map["neither"] == "TTGAAATTTCGC" # Neither removed

def test_clean_filter_length_checks():
    """Test filtering based on length."""
    seq_ok = SeqRecord(Seq("ATGCGTTAG"), id="ok")
    seq_short_orig = SeqRecord(Seq("ATGTA"), id="short1")
    seq_short_after_gap = SeqRecord(Seq("AT---GTAA"), id="short2") # ATG TAA -> '' -> removed
    seq_short_after_ss = SeqRecord(Seq("ATGCCCTAG"), id="short3") # ATG CCC TAG -> CCC -> ok
    seq_not_mult3 = SeqRecord(Seq("ATGCCGCG"), id="notmult3")
    seq_not_mult3_gap = SeqRecord(Seq("ATGCC---GCG"), id="notmult3_gap") # ATGCCGCG -> len 8 -> removed

    cleaned = utils.clean_and_filter_sequences([
        seq_ok, seq_short_orig, seq_short_after_gap,
        seq_short_after_ss, seq_not_mult3, seq_not_mult3_gap
    ])
    cleaned_ids = {rec.id for rec in cleaned}
    assert cleaned_ids == {"ok", "short3"}
    assert str([r.seq for r in cleaned if r.id=='ok'][0]) == "CGT"
    assert str([r.seq for r in cleaned if r.id=='short3'][0]) == "CCC"


# --- Tests for reference loading ---

def test_load_reference_ok(tmp_path, standard_genetic_code_dict):
    """Test loading a valid reference file and calculating weights."""
    # Create dummy TSV file with more complete data
    ref_content = (
        "Codon\tFrequency\tAminoAcid\tBlah\n"
        # Lysine (K) - AAA preferred
        "AAA\t0.6\tK\tX1\n"
        "AAG\t0.4\tK\tX2\n"
        # Asparagine (N) - AAT preferred
        "AAC\t0.3\tN\tY1\n"
        "AAT\t0.7\tN\tY2\n"
        # Glycine (G) - GGC most preferred, GGG least
        "GGA\t0.25\tG\tZ1\n"
        "GGC\t0.40\tG\tZ2\n"
        "GGG\t0.10\tG\tZ3\n"
        "GGT\t0.25\tG\tZ4\n"
        # Methionine (M) - Single codon
        "ATG\t1.0\tM\tW1\n"
        # Stop Codon (Example) - Should be excluded
        "TAA\t0.5\t*\tS1\n"
    )
    
    ref_file = tmp_path / "dummy_ref_ok_v2.tsv" # Changed filename slightly
    ref_file.write_text(ref_content)

    ref_data = utils.load_reference_usage(str(ref_file), standard_genetic_code_dict, 1)
    assert ref_data is not None
    assert isinstance(ref_data, pd.DataFrame)
    assert 'Weight' in ref_data.columns
    assert ref_data.index.name == 'Codon'

    # --- Recalculated Expected Weights based on new ref_content ---
    # K: Max RSCU=1.2 (AAA). W_AAA=1.0, W_AAG=0.8/1.2=0.667
    # N: Max RSCU=1.4 (AAT). W_AAC=0.6/1.4=0.429, W_AAT=1.0
    # G: Max RSCU=1.6 (GGC). W_GGA=1.0/1.6=0.625, W_GGC=1.0, W_GGG=0.4/1.6=0.25, W_GGT=1.0/1.6=0.625
    # M: W_ATG=1.0

    # --- Assert against recalculated expected weights ---
    # K weights
    assert np.isclose(ref_data.loc['AAA', 'Weight'], 1.0, atol=0.01)
    assert np.isclose(ref_data.loc['AAG', 'Weight'], 0.667, atol=0.001)
    # N weights
    assert np.isclose(ref_data.loc['AAC', 'Weight'], 0.429, atol=0.001) # This is the key check for the original bug
    assert np.isclose(ref_data.loc['AAT', 'Weight'], 1.0, atol=0.01)
    # G weights
    assert np.isclose(ref_data.loc['GGA', 'Weight'], 0.625, atol=0.001)
    assert np.isclose(ref_data.loc['GGC', 'Weight'], 1.0, atol=0.01)
    assert np.isclose(ref_data.loc['GGG', 'Weight'], 0.25, atol=0.001)
    assert np.isclose(ref_data.loc['GGT', 'Weight'], 0.625, atol=0.001)
    # M weight
    assert np.isclose(ref_data.loc['ATG', 'Weight'], 1.0, atol=0.01)
    # Ensure stop codon was excluded
    assert 'TAA' not in ref_data.index




def test_load_reference_rscu_direct(tmp_path, standard_genetic_code_dict):
    """Test loading file where RSCU is provided directly."""
    ref_content = "Codon\tRSCU\tAA\nAAA\t1.1\tK\nAAC\t0.8\tN\nAAT\t1.2\tN\n"
    ref_file = tmp_path / "dummy_ref_rscu.tsv"
    ref_file.write_text(ref_content)

    ref_data = utils.load_reference_usage(str(ref_file), standard_genetic_code_dict, 1)
    assert ref_data is not None
    # Check weights (N: AAC=0.8, AAT=1.2 -> max=1.2 -> w_AAC=0.666, w_AAT=1.0)
    assert np.isclose(ref_data.loc['AAC', 'Weight'], 0.8 / 1.2)
    assert np.isclose(ref_data.loc['AAT', 'Weight'], 1.0)
    assert np.isclose(ref_data.loc['AAA', 'Weight'], 1.0) # Single codon gets weight 1

def test_load_reference_bad_file(tmp_path, standard_genetic_code_dict):
    """Test loading missing or incorrectly formatted files."""
    # File not found
    assert utils.load_reference_usage("nonexistent_file.tsv", standard_genetic_code_dict, 1) is None

    # File missing required columns
    ref_content_bad = "Codon\tSomethingElse\nAAA\tX\nAAC\tY"
    ref_file_bad = tmp_path / "dummy_ref_bad.tsv"
    ref_file_bad.write_text(ref_content_bad)
    assert utils.load_reference_usage(str(ref_file_bad), standard_genetic_code_dict, 1) is None

# --- Add tests for get_genetic_code, get_synonymous_codons if needed ---