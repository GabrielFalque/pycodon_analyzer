# tests/conftest.py
import pytest
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation, CompoundLocation

# Ensure the src directory is in the Python path for tests
# This allows tests to import modules from src.pycodon_analyzer
# Note: Pytest often handles this automatically if run from the project root.
# However, explicit addition can be more robust in some CI/IDE environments.
# Comment out if pytest already discovers your src package without issues.
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# --- Reset Logging Configuration Fixture ---
@pytest.fixture(autouse=True)
def reset_logging_config(monkeypatch):
    """Reset logging configuration before each test to ensure proper log capture.
    
    This fixture runs automatically before each test and ensures that pytest's
    log capture mechanism works correctly by completely resetting the logging system
    and redirecting all output to pytest's log capture.
    """
    import logging
    import io
    
    # Completely reset the logging system
    logging.shutdown()
    logging.root.handlers.clear()
    
    # Monkeypatch the basicConfig to do nothing
    monkeypatch.setattr(logging, 'basicConfig', lambda **kwargs: None)
    
    # Ensure all loggers propagate to the root logger
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.disabled = False
        logger.setLevel(logging.NOTSET)
    
    # Reset the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    root_logger.handlers = []
    
    # Add a NullHandler to avoid "No handlers could be found" warnings
    root_logger.addHandler(logging.NullHandler())
    
    yield

# --- General Utility Fixtures ---

@pytest.fixture
def standard_genetic_code_dict() -> Dict[str, str]:
    """Provides the standard genetic code dictionary (NCBI table 1)."""
    return {
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

@pytest.fixture
def reference_cai_weights_dict() -> Dict[str, float]:
    """Provides a sample dictionary of CAI reference weights."""
    return {
        "TTT": 0.80, "TTC": 1.00, # F
        "TTA": 0.50, "TTG": 1.00, "CTT": 0.70, "CTC": 0.60, "CTA": 0.30, "CTG": 0.90, # L
        "ATT": 0.75, "ATC": 1.00, "ATA": 0.20, # I
        "ATG": 1.00, # M
        "GTT": 0.60, "GTC": 0.80, "GTA": 0.30, "GTG": 1.00, # V
        "TCT": 0.50, "TCC": 0.70, "TCA": 0.40, "TCG": 0.30, "AGT": 0.25, "AGC": 0.45, #S
        # Add more for other amino acids as needed by tests
        "GCT": 1.0, "GCC": 0.8, "GCA":0.6, "GCG":0.4, # A
        "CCT": 1.0, # P
        "ACT": 1.0, # T
        "GGT": 1.0, # G
        "CGT": 1.0, # R
        "TAT": 1.0, "TAC":0.9, # Y
        "TGT": 1.0, "TGC":0.7, # C
        "TGG": 1.0, # W
        "CAA": 1.0, "CAG":0.8, # Q
        "AAT": 1.0, "AAC":0.7, # N
        "CAT": 1.0, "CAC":0.6, # H
        "GAA": 1.0, "GAG":0.9, # E
        "GAT": 1.0, "GAC":0.5, # D
        "AAA": 1.0, "AAG":0.4, # K
    }

# --- Fixtures for Sequence Data (used in test_utils.py, test_analysis.py) ---

@pytest.fixture
def simple_seq_records() -> List[SeqRecord]:
    """Provides a list of simple SeqRecord objects for testing cleaning and analysis."""
    return [
        SeqRecord(Seq("ATGCGTAGA---"), id="SeqA"),              # Valid after cleaning -> CGT (TAA stop removed)
        SeqRecord(Seq("ATGCCCTAACCC"), id="SeqB"),            # Valid after cleaning -> CCC (ATG start, TAA stop removed)
        SeqRecord(Seq("ATGAAAGGGNNN---TGA"), id="SeqC_with_N"),# High N%, removed
        SeqRecord(Seq("ATGTTT---"), id="SeqD_short"),         # Valid after cleaning -> TTT (ATG start removed)
        SeqRecord(Seq("ATGCGATAG"), id="SeqE_nostop_ok"),      # Valid after cleaning -> CGA (ATG start, TAG stop removed)
        SeqRecord(Seq("ACG---ACG"), id="SeqF_no_start"),       # Valid, no standard start/stop -> ACGACG
        SeqRecord(Seq("---"), id="SeqG_only_gaps"),           # Empty after cleaning, removed
        SeqRecord(Seq(""), id="SeqH_empty_str"),              # Empty, removed
        SeqRecord(Seq("NNNNNNNNN"), id="SeqI_all_N"),          # All N, removed by ambiguity filter
    ]

@pytest.fixture
def sample_seq_record_1() -> SeqRecord:
    """A sample SeqRecord for analysis tests."""
    return SeqRecord(Seq("ATGTTTGGCTAATGC"), id="S1", description="Sample sequence 1")
    # Codons: ATG, TTT, GGC, TAA, TGC -> M, F, G, *, C

@pytest.fixture
def sample_seq_record_2() -> SeqRecord:
    """Another sample SeqRecord for analysis tests."""
    return SeqRecord(Seq("ATGCCCAAGTAGCCC"), id="S2", description="Sample sequence 2")
    # Codons: ATG, CCC, AAG, TAG, CCC -> M, P, K, *, P

@pytest.fixture
def sample_sequences_list(sample_seq_record_1, sample_seq_record_2) -> List[SeqRecord]:
    """A list of SeqRecords for run_full_analysis tests."""
    return [sample_seq_record_1, sample_seq_record_2]


# --- Fixtures for I/O Tests (test_io.py) ---

@pytest.fixture
def dummy_fasta_file_valid(tmp_path: Path) -> Path:
    """Creates a temporary valid FASTA file."""
    content = ">Seq1\nATGCGT\n>Seq2_with_description description text\nNNNCGTA\n"
    fasta_file = tmp_path / "valid_io.fasta"
    fasta_file.write_text(content)
    return fasta_file

@pytest.fixture
def dummy_fasta_file_invalid_type(tmp_path: Path) -> Path: # Name matches test_io.py usage
    """Creates a FASTA file with characters that might be considered 'invalid type' by some tools,
    though FASTA itself is lenient."""
    content = ">InvalidChars\nATGXXXCGT\n" # XXX are not standard DNA/RNA
    fasta_file = tmp_path / "invalid_chars_io.fasta"
    fasta_file.write_text(content)
    return fasta_file

# --- Fixtures for Plotting Tests (test_plotting.py) ---
# Some of these were defined locally in test_plotting.py, centralizing them here.

@pytest.fixture
def sample_rscu_df_for_plot() -> pd.DataFrame:
    """DataFrame with RSCU values for plotting."""
    data = {
        'Codon': ['TTT', 'TTC', 'TTA', 'TTG', 'ATG'],
        'AminoAcid': ['F', 'F', 'L', 'L', 'M'],
        'RSCU': [1.2, 0.8, 1.5, 0.5, 1.0],
        'SequenceID': ['Seq1'] * 5
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_per_sequence_df_for_plots() -> pd.DataFrame:
    """DataFrame similar to per_sequence_metrics for various plots."""
    data = {
        'ID': ['Seq1', 'Seq2', 'Seq3', 'Seq4'],
        'Gene': ['GeneA', 'GeneA', 'GeneB', 'GeneB'],
        'GC': [0.4, 0.5, 0.6, 0.7],
        'GC1': [0.3, 0.4, 0.5, 0.6],
        'GC2': [0.4, 0.5, 0.6, 0.7],
        'GC3': [0.5, 0.6, 0.7, 0.8],
        'ENC': [40.0, 45.0, 50.0, 55.0],
        'Length': [300, 330, 360, 390],
        'CAI': [0.5, 0.6, 0.7, 0.8], # Added for correlation plots
        'TotalCodons': [100, 110, 120, 130], # Added for correlation plots
        'Fop': [0.4,0.5,0.6,0.7],
        'RCDI': [0.9,1.0,1.1,0.8],
        'ProteinLength': [99,109,119,129],
        'GRAVY': [-0.1,0.0,0.1,0.2],
        'Aromaticity': [0.05,0.06,0.07,0.08]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_rscu_df_for_ca_plot_fixture() -> dict: # Renamed to avoid conflict with test_analysis.py's local fixture
    """Provides RSCU data and groups for CA plotting tests.
       Used by sample_ca_results_for_plot fixture.
    """
    data = {
        "Seq1": {"TTT": 1.2, "TTC": 0.8, "CTG": 1.5, "CTA": 0.5, "ATG":1.0, "GGG":0.9},
        "Seq2": {"TTT": 0.7, "TTC": 1.3, "CTG": 0.6, "CTA": 1.4, "ATG":1.0, "GGG":1.1},
        "Seq3": {"TTT": 1.0, "TTC": 1.0, "CTG": 1.0, "CTA": 1.0, "ATG":1.0, "GGG":1.0},
        "Seq4": {"TTT": 1.8, "TTC": 0.2, "CTG": 1.7, "CTA": 0.3, "ATG":1.0, "GGG":1.3},
    }
    rscu_df = pd.DataFrame.from_dict(data, orient='index')
    rscu_df.index.name = "ID"
    rscu_df.columns.name = "Codon" # Important for prince
    # Ensure there's enough variance for CA to run
    if rscu_df.shape[0] < 2 or rscu_df.shape[1] < 2: # pragma: no cover
        # Add dummy data to ensure CA can run if initial data is too small
        rscu_df.loc["Seq5", "GGC"] = 1.0 
        rscu_df = rscu_df.fillna(0)
    groups = pd.Series(['GroupA', 'GroupA', 'GroupB', 'GroupB'] + (['GroupC'] if "Seq5" in rscu_df.index else []), index=rscu_df.index, name="Group")
    return {"rscu_df": rscu_df, "groups": groups}


# --- Fixtures for Extraction Tests (test_extraction.py) ---

@pytest.fixture
def ref_genome_seq_for_extraction() -> Seq: # Renamed to avoid conflict
    return Seq("---ATGCATTATTGCGCG---TAGCATTAGCAT---")

@pytest.fixture
def sample_alignment_dict_for_extraction(ref_genome_seq_for_extraction) -> Dict[str, SeqRecord]:
    return {
        "RefGenome": SeqRecord(ref_genome_seq_for_extraction, id="RefGenome"),
        "Org1": SeqRecord(Seq("NNNATGCATTATTGCGCGXXXTAGCATTAGCATNNN"), id="Org1"),
        "Org2": SeqRecord(Seq("---ATGGATTATTACGCG---TAGGATTAGGAT---"), id="Org2"),
        "OrgShort": SeqRecord(Seq("ATGC"), id="OrgShort")
    }

@pytest.fixture
def gene_feature_fwd_for_extraction() -> SeqFeature:
    location = FeatureLocation(3, 12, strand=1) # ATGCATTAT
    return SeqFeature(location, type="CDS", qualifiers={"gene": ["GeneFwd"]})

@pytest.fixture
def gene_feature_rev_for_extraction() -> SeqFeature:
    location = FeatureLocation(21, 30, strand=-1) # TAGCATTAG -> revcomp CTAATGCTA
    return SeqFeature(location, type="CDS", qualifiers={"locus_tag": ["GeneRev"]})

@pytest.fixture
def gene_annotation_record_for_extraction(ref_genome_seq_for_extraction) -> SeqRecord:
    return SeqRecord(ref_genome_seq_for_extraction, id="RefGenome", description="Reference for extraction tests")


# --- Fixtures for CLI Tests (test_cli.py) ---

@pytest.fixture
def dummy_gene_fasta_content_for_cli() -> str: # Renamed
    return ">GeneA_Seq1\nATGCGTTAA\n>GeneA_Seq2\nATGCCCTAG\n"

@pytest.fixture
def dummy_gene_dir_for_cli(tmp_path: Path, dummy_gene_fasta_content_for_cli: str) -> Path: # Renamed
    gene_dir = tmp_path / "cli_genes_input"
    gene_dir.mkdir()
    (gene_dir / "gene_GeneA.fasta").write_text(dummy_gene_fasta_content_for_cli)
    (gene_dir / "gene_GeneB.fasta").write_text(">GeneB_Seq1\nTTTCCCGGG\n")
    return gene_dir

@pytest.fixture
def dummy_ref_usage_file_for_cli(tmp_path: Path) -> Path: # Renamed
    content = "Codon\tFrequency\nAAA\t0.6\nAAG\t0.4\nATG\t1.0\n"
    ref_file = tmp_path / "cli_dummy_ref.tsv"
    ref_file.write_text(content)
    return ref_file

@pytest.fixture
def dummy_metadata_file_for_cli(tmp_path: Path) -> Path: # Renamed
    content = "seq_id,Category,Value\nGeneA_Seq1,X,10\nGeneA_Seq2,Y,20\nGeneB_Seq1,X,30\n"
    meta_file = tmp_path / "cli_metadata.csv"
    meta_file.write_text(content)
    return meta_file

@pytest.fixture
def dummy_annotation_gb_file_for_cli(tmp_path: Path) -> Path: # Renamed, using GenBank format
    gb_content = """LOCUS       RefAnnotationSeq    50 bp    DNA     linear   UNK 01-JAN-1980
DEFINITION  Reference sequence for annotation.
ACCESSION   RefAnnotationSeq
VERSION     RefAnnotationSeq.1
SOURCE      synthetic
  ORGANISM  synthetic
FEATURES             Location/Qualifiers
     source          1..50
     gene            10..18
                     /gene="GeneX"
     CDS             10..18
                     /gene="GeneX"
     gene            complement(30..38)
                     /gene="GeneY"
     CDS             complement(30..38)
                     /gene="GeneY"
ORIGIN
        1 aaaaaaaaag attacagabb bbbbbbbbg attacaccdd ddddddddee eeeeeeeeee
//
"""
    anno_file = tmp_path / "cli_annotations.gb"
    anno_file.write_text(gb_content)
    return anno_file

@pytest.fixture
def dummy_alignment_fasta_file_for_cli(tmp_path: Path) -> Path: # Renamed
    ref_seq_str =   "AAAAAAAAAGATTACAGABBBBBBBBBBGATTACACCDDDDDDDDDDEEEEEEEEEE" # GeneX: GATTACAGA, GeneY: GATTACACC
    org_a_seq_str = "AAAAAAAAAGATTACAGABBBBBBBBBBGATTACACCEEEEEEEEEFFFFFFFFFF"
    org_b_seq_str = "AAAAAAAAATTTTTTTTABBBBBBBBBBCCCCCCCCCCGGGGGGGGGHHHHHHHHHH" # Different for GeneY region

    content = f""">RefAnnotationSeq
{ref_seq_str}
>OrgA
{org_a_seq_str}
>OrgB
{org_b_seq_str}
"""
    align_file = tmp_path / "cli_alignment.fasta"
    align_file.write_text(content)
    return align_file