# tests/conftest.py
import pytest
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

@pytest.fixture
def simple_seq_records():
    """Provides a list of simple SeqRecord objects for testing."""
    return [
        SeqRecord(Seq("ATGCGTAGA---"), id="SeqA"),
        SeqRecord(Seq("ATGCCCTAACCC"), id="SeqB"),
        SeqRecord(Seq("ATGAAAGGGNNN---TGA"), id="SeqC_with_N"),
        SeqRecord(Seq("ATGTTT---"), id="SeqD_short"), # Too short after cleaning
        SeqRecord(Seq("ATGCGATAG"), id="SeqE_nostop_ok"),
        SeqRecord(Seq("ACG---ACG"), id="SeqF_no_start"),
        SeqRecord(Seq("---"), id="SeqG_gaps_only"),
        SeqRecord(Seq(""), id="SeqH_empty"),
        SeqRecord(Seq("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"), id="SeqI_all_N"),
    ]

@pytest.fixture
def standard_genetic_code_dict():
    """Provides the standard genetic code dictionary."""
    # Avoid direct import from src if possible, define here or load reliably
    # This ensures tests are independent of potential errors in src/utils.py import path
    return {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S',
        'TCA': 'S', 'TCG': 'S', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CTT': 'L', 'CTC': 'L',
        'CTA': 'L', 'CTG': 'L', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGT': 'R', 'CGC': 'R',
        'CGA': 'R', 'CGG': 'R', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'AAT': 'N', 'AAC': 'N',
        'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A',
        'GCA': 'A', 'GCG': 'A', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

@pytest.fixture
def dummy_ref_weights():
    """Provides a simple dictionary of reference weights."""
    return {'AAA': 1.0, 'AAC': 0.42, 'AAT': 0.57, 'GGG': 0.1, 'GGC': 1.0, 'ATG': 1.0}