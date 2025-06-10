# tests/test_extraction.py
import pytest
import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Ensure the src directory is in the Python path for tests
try:
    from pycodon_analyzer import extraction, utils
except ImportError: # pragma: no cover
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from pycodon_analyzer import extraction, utils


# --- Fixtures for Extraction Tests ---

@pytest.fixture
def sample_location_strings() -> List[str]:
    """Sample GenBank location strings for testing."""
    return [
        "100..200",
        "complement(300..400)",
        "join(100..200,300..400)",
        "complement(join(100..200,300..400))",
        "<100..>200",
        "100",
        "invalid_location"
    ]

@pytest.fixture
def sample_annotation_fasta_content() -> str:
    """Sample FASTA content with annotation headers."""
    return (
        ">seq1 [gene=GeneA] [location=1..3]\n"
        "ATGCATGCATGCATGCATGC\n"
        ">seq2 [locus_tag=LocusB] [location=complement(5..10)]\n"
        "GCATGCATGCATGCATGCAT\n"
        ">seq3 [gene=GeneC] [location=join(2..4,6..8)]\n"
        "TGCATGCATGCATGCATGCA\n"
        ">seq4 [gene=GeneD]\n"  # Missing location
        "CATGCATGCATGCATGCATG\n"
        ">seq5 [location=9..12]\n"  # Missing gene/locus_tag
        "ATGCATGCATGCATGCATGC\n"
        ">seq6 [gene=GeneE] [location=invalid_location]\n"  # Invalid location
        "GCATGCATGCATGCATGCAT\n"
    )

@pytest.fixture
def sample_alignment_fasta_content() -> str:
    """Sample FASTA content with aligned sequences."""
    return (
        ">RefSeq\n"
        "ATG---CATGCATGC---ATGC\n"
        ">Seq1\n"
        "ATGCATCATGCATGCAAATGC\n"
        ">Seq2\n"
        "ATG---CATGCATGC---ATGC\n"
    )

@pytest.fixture
def sample_annotation_fasta_file(tmp_path: Path, sample_annotation_fasta_content: str) -> Path:
    """Create a temporary FASTA file with annotation headers."""
    annotation_file = tmp_path / "annotations.fasta"
    annotation_file.write_text(sample_annotation_fasta_content)
    return annotation_file

@pytest.fixture
def sample_alignment_fasta_file(tmp_path: Path, sample_alignment_fasta_content: str) -> Path:
    """Create a temporary FASTA file with aligned sequences."""
    alignment_file = tmp_path / "alignment.fasta"
    alignment_file.write_text(sample_alignment_fasta_content)
    return alignment_file

@pytest.fixture
def sample_ref_aligned_record() -> SeqRecord:
    """Sample reference aligned sequence record."""
    return SeqRecord(
        Seq("ATG---CATGCATGC---ATGC"),
        id="RefSeq",
        description="Reference sequence"
    )


# --- Test Class for parse_genbank_location_string ---
class TestParseGenbankLocationString:
    
    def test_standard_location(self):
        """Test parsing standard location string."""
        start, end, strand = extraction.parse_genbank_location_string("100..200")
        assert start == 100
        assert end == 200
        assert strand == '+'
    
    def test_complement_location(self):
        """Test parsing complement location string."""
        start, end, strand = extraction.parse_genbank_location_string("complement(300..400)")
        assert start == 300
        assert end == 400
        assert strand == '-'
    
    def test_join_location(self):
        """Test parsing join location string."""
        start, end, strand = extraction.parse_genbank_location_string("join(100..200,300..400)")
        assert start == 100
        assert end == 400
        assert strand == '+'
    
    def test_complement_join_location(self):
        """Test parsing complement of join location string."""
        start, end, strand = extraction.parse_genbank_location_string("complement(join(100..200,300..400))")
        assert start == 100
        assert end == 400
        assert strand == '-'
    
    def test_boundary_markers(self):
        """Test parsing location with boundary markers."""
        start, end, strand = extraction.parse_genbank_location_string("<100..>200")
        assert start == 100
        assert end == 200
        assert strand == '+'
    
    def test_single_position(self):
        """Test parsing single position."""
        start, end, strand = extraction.parse_genbank_location_string("100")
        assert start == 100
        assert end == 100
        assert strand == '+'
    
    def test_invalid_location(self, caplog):
        """Test parsing invalid location string."""
        caplog.set_level(logging.WARNING)
        start, end, strand = extraction.parse_genbank_location_string("invalid_location")
        assert start is None
        assert end is None
        assert strand == '+'
        assert "Could not parse coordinates from location string" in caplog.text


# --- Test Class for parse_annotation_fasta_for_extraction ---
class TestParseAnnotationFastaForExtraction:
    
    def test_parse_valid_annotations(self, sample_annotation_fasta_file):
        """Test parsing valid annotations from FASTA file."""
        gene_annotations = extraction.parse_annotation_fasta_for_extraction(sample_annotation_fasta_file)
        assert len(gene_annotations) == 3  # Only 3 valid entries with both gene/locus_tag and valid location
        
        # Check first annotation (GeneA)
        assert gene_annotations[0]['GeneName'] == 'GeneA'
        assert gene_annotations[0]['Start'] == 1
        assert gene_annotations[0]['End'] == 3
        assert gene_annotations[0]['Strand'] == '+'
        
        # Check second annotation (LocusB)
        assert gene_annotations[1]['GeneName'] == 'LocusB'
        assert gene_annotations[1]['Start'] == 5
        assert gene_annotations[1]['End'] == 10
        assert gene_annotations[1]['Strand'] == '-'
        
        # Check third annotation (GeneC)
        assert gene_annotations[2]['GeneName'] == 'GeneC'
        assert gene_annotations[2]['Start'] == 2
        assert gene_annotations[2]['End'] == 8
        assert gene_annotations[2]['Strand'] == '+'
    
    def test_parse_nonexistent_file(self, caplog):
        """Test parsing a non-existent file."""
        caplog.set_level(logging.ERROR)
        non_existent_file = Path("non_existent_file.fasta")
        
        with pytest.raises(FileNotFoundError):
            extraction.parse_annotation_fasta_for_extraction(non_existent_file)
        
        assert f"Annotation file not found: {non_existent_file}" in caplog.text
    
    def test_parse_empty_file(self, tmp_path, caplog):
        """Test parsing an empty file."""
        caplog.set_level(logging.WARNING)
        empty_file = tmp_path / "empty.fasta"
        empty_file.write_text("")
        
        gene_annotations = extraction.parse_annotation_fasta_for_extraction(empty_file)
        assert len(gene_annotations) == 0
        assert "No annotations with gene name/locus_tag and parsable location found" in caplog.text


# --- Test Class for map_coordinates_to_alignment_for_extraction ---
class TestMapCoordinatesToAlignmentForExtraction:
    
    @pytest.fixture
    def sample_genes_info_list(self) -> List[Dict]:
        """Sample gene info list for testing coordinate mapping."""
        return [
            {
                'GeneName': 'GeneA',
                'Start': 1,  # 1-based
                'End': 3,    # 1-based
                'Strand': '+',
                'OriginalLocationStr': '1..3'
            },
            {
                'GeneName': 'GeneB',
                'Start': 5,  # 1-based
                'End': 10,   # 1-based
                'Strand': '-',
                'OriginalLocationStr': 'complement(5..10)'
            },
            {
                'GeneName': 'GeneC',
                'Start': 100,  # Out of bounds
                'End': 110,    # Out of bounds
                'Strand': '+',
                'OriginalLocationStr': '100..110'
            }
        ]
    
    def test_map_coordinates_standard(self, sample_genes_info_list, sample_ref_aligned_record):
        """Test mapping coordinates to alignment."""
        # Reference sequence: "ATG---CATGCATGC---ATGC"
        # Ungapped positions:  123    456789012    3456
        # Aligned positions:   012345678901234567890123
        
        mapped_genes_info = extraction.map_coordinates_to_alignment_for_extraction(
            sample_genes_info_list, sample_ref_aligned_record
        )
        
        assert len(mapped_genes_info) == 2  # GeneC is out of bounds and should be skipped
        
        # Check GeneA mapping (1-3 -> 0-2)
        assert mapped_genes_info[0]['GeneName'] == 'GeneA'
        assert mapped_genes_info[0]['Aln_Start_0based'] == 0  # ATG (start at position 0)
        assert mapped_genes_info[0]['Aln_End_Exclusive'] == 3  # End after G (position 3)
        
        # Check GeneB mapping (5-10 -> 7-15)
        assert mapped_genes_info[1]['GeneName'] == 'GeneB'
        assert mapped_genes_info[1]['Aln_Start_0based'] == 7  # C (position 7)
        assert mapped_genes_info[1]['Aln_End_Exclusive'] == 13  # End after C (position 13)
    
    def test_map_coordinates_empty_genes_list(self, sample_ref_aligned_record):
        """Test mapping with empty genes list."""
        mapped_genes_info = extraction.map_coordinates_to_alignment_for_extraction(
            [], sample_ref_aligned_record
        )
        assert len(mapped_genes_info) == 0
    
    def test_map_coordinates_missing_ref_record(self, sample_genes_info_list):
        """Test mapping with missing reference record."""
        with pytest.raises(ValueError, match="Reference sequence record is missing for coordinate mapping"):
            extraction.map_coordinates_to_alignment_for_extraction(sample_genes_info_list, None)


# --- Test Class for extract_gene_alignments_from_genome_msa ---
class TestExtractGeneAlignmentsFromGenomeMSA:
    
    def test_extract_gene_alignments_basic(self, sample_annotation_fasta_file, sample_alignment_fasta_file, tmp_path, caplog):
        """Test basic gene alignment extraction."""
        output_dir = tmp_path / "output"
        
        # Execute the function and check that it runs without errors
        extraction.extract_gene_alignments_from_genome_msa(
            sample_annotation_fasta_file,
            sample_alignment_fasta_file,
            "RefSeq",
            output_dir
        )
        
        # Check if output directory was created
        assert output_dir.exists()
        
        # Check if gene files were created - we expect at least one
        gene_files = list(output_dir.glob("gene_*.fasta"))
        assert len(gene_files) > 0
        
        # Check for expected log messages
        assert "Starting gene extraction" in caplog.text
        assert "Found 3 potential gene annotations" in caplog.text
        assert "Read 3 genome sequences" in caplog.text
    
    def test_extract_gene_alignments_nonexistent_annotation(self, sample_alignment_fasta_file, tmp_path):
        """Test extraction with non-existent annotation file."""
        output_dir = tmp_path / "output_nonexistent_anno"
        
        with pytest.raises(FileNotFoundError):
            extraction.extract_gene_alignments_from_genome_msa(
                Path("non_existent_annotations.fasta"),
                sample_alignment_fasta_file,
                "RefSeq",
                output_dir
            )
    
    def test_extract_gene_alignments_nonexistent_alignment(self, sample_annotation_fasta_file, tmp_path):
        """Test extraction with non-existent alignment file."""
        output_dir = tmp_path / "output_nonexistent_aln"
        
        with pytest.raises(ValueError):
            extraction.extract_gene_alignments_from_genome_msa(
                sample_annotation_fasta_file,
                Path("non_existent_alignment.fasta"),
                "RefSeq",
                output_dir
            )
    
    def test_extract_gene_alignments_nonexistent_ref_id(self, sample_annotation_fasta_file, sample_alignment_fasta_file, tmp_path):
        """Test extraction with non-existent reference ID."""
        output_dir = tmp_path / "output_nonexistent_ref"
        
        with pytest.raises(ValueError, match="Reference sequence ID 'NonExistentRef' not found in alignment file"):
            extraction.extract_gene_alignments_from_genome_msa(
                sample_annotation_fasta_file,
                sample_alignment_fasta_file,
                "NonExistentRef",
                output_dir
            )
    
    def test_extract_gene_alignments_no_valid_annotations(self, tmp_path, sample_alignment_fasta_file):
        """Test extraction with no valid annotations."""
        output_dir = tmp_path / "output_no_valid_anno"
        
        # Create annotation file with no valid annotations
        invalid_anno_file = tmp_path / "invalid_annotations.fasta"
        invalid_anno_file.write_text(">seq1\nATGC\n")
        
        with pytest.raises(ValueError, match="No valid annotations parsed from the annotation file"):
            extraction.extract_gene_alignments_from_genome_msa(
                invalid_anno_file,
                sample_alignment_fasta_file,
                "RefSeq",
                output_dir
            )