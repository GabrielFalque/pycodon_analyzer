# tests/test_io.py

import pytest
import os
import logging
from pathlib import Path # ADDED
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Ensure the src directory is in the Python path for tests
try:
    from pycodon_analyzer import io
except ImportError: # pragma: no cover
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from pycodon_analyzer import io


# --- Test Class for read_fasta ---
class TestReadFasta:

    # Test for VALID_DNA_CHARS default value
    def test_valid_dna_chars_default_value(self):
        """Test that VALID_DNA_CHARS has the expected default values."""
        # We can't easily test the import fallback directly, but we can verify
        # that the module has the expected VALID_DNA_CHARS set with correct values
        assert hasattr(io, 'VALID_DNA_CHARS')
        assert isinstance(io.VALID_DNA_CHARS, set)
        assert 'A' in io.VALID_DNA_CHARS
        assert 'T' in io.VALID_DNA_CHARS
        assert 'C' in io.VALID_DNA_CHARS
        assert 'G' in io.VALID_DNA_CHARS
        assert 'N' in io.VALID_DNA_CHARS
        assert '-' in io.VALID_DNA_CHARS

    # Using fixtures from conftest.py
    def test_read_valid_fasta(self, dummy_fasta_file_valid, caplog):
        """Test reading a correctly formatted FASTA file."""
        caplog.set_level(logging.INFO) # Changed to INFO
        sequences = io.read_fasta(str(dummy_fasta_file_valid))
        assert len(sequences) == 2
        assert isinstance(sequences[0], SeqRecord)
        assert sequences[0].id == "Seq1"
        assert str(sequences[0].seq) == "ATGCGT"
        assert sequences[1].id == "Seq2_with_description"
        assert str(sequences[1].seq) == "NNNCGTA"
        assert sequences[1].description == "Seq2_with_description description text"
        assert f"Successfully read {len(sequences)} sequences from {str(dummy_fasta_file_valid)}" in caplog.text # Updated log message to full path

    def test_read_empty_fasta(self, tmp_path: Path, caplog):
        """Test reading an empty FASTA file."""
        caplog.set_level(logging.WARNING)
        empty_file = tmp_path / "empty.fasta"
        empty_file.write_text("")
        sequences = io.read_fasta(str(empty_file))
        assert len(sequences) == 0
        assert f"FASTA file '{str(empty_file)}' is empty." in caplog.text # Updated log message to match io.py
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_read_fasta_file_not_found(self, caplog):
        """Test reading a non-existent FASTA file."""
        caplog.set_level(logging.ERROR)
        non_existent_file = "non_existent.fasta"
        sequences = io.read_fasta(non_existent_file)
        assert len(sequences) == 0
        assert f"FASTA file not found: '{non_existent_file}'" in caplog.text # Updated log message
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_read_fasta_malformed_entry(self, tmp_path: Path, caplog):
        """Test reading a FASTA file with a malformed entry (e.g., no header)."""
        caplog.set_level(logging.INFO) # Changed to INFO to capture success message
        malformed_content = "ATGCGT\n>Seq2\nNNNCGT" # Missing header for the first sequence
        malformed_file = tmp_path / "malformed.fasta"
        malformed_file.write_text(malformed_content)
        
        sequences = io.read_fasta(str(malformed_file))
        # Biopython's SeqIO.parse will parse Seq2 successfully and skip the malformed first part.
        assert len(sequences) == 1
        assert sequences[0].id == "Seq2"
        assert str(sequences[0].seq) == "NNNCGT"
        assert f"Successfully read {len(sequences)} sequences from {str(malformed_file)}" in caplog.text # Check for success message with full path
        # Also check for potential Biopython warnings/errors if they occur, though our function now handles them.
        # Biopython's SeqIO.parse silently skips lines that don't start with '>' if they are not part of a sequence.
        # Therefore, no specific error/warning is expected from io.read_fasta for the initial "ATGCGT" line.
        # The function should only log a warning if no valid records are found at all, which is not the case here.
        # We ensure no unexpected errors were logged.
        assert not any(record.levelno >= logging.ERROR for record in caplog.records)


    def test_read_fasta_with_non_fasta_content(self, tmp_path: Path, caplog):
        """Test reading a file that is not in FASTA format."""
        caplog.set_level(logging.ERROR) # Expecting a parsing error or Biopython error
        not_fasta_content = "This is just some plain text, not FASTA."
        not_fasta_file = tmp_path / "not_fasta.txt"
        not_fasta_file.write_text(not_fasta_content)
        
        # Biopython's SeqIO.parse might raise a ValueError or return an empty list
        # if the format cannot be determined or if it's strictly FASTA.
        # Our function catches general exceptions.
        sequences = io.read_fasta(str(not_fasta_file))
        assert len(sequences) == 0
        # The function now logs a warning if no valid records are found.
        assert f"FASTA file '{str(not_fasta_file)}' contains content but no valid FASTA records could be parsed." in caplog.text
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_read_fasta_with_mixed_case_and_spaces_in_header(self, tmp_path: Path):
        """Test FASTA headers with mixed case and leading/trailing spaces."""
        content = ">  SeqOne mixedCase  \nATGC\n>SeqTwo \nCGTA"
        fasta_file = tmp_path / "mixed_header.fasta"
        fasta_file.write_text(content)
        sequences = io.read_fasta(str(fasta_file))
        assert len(sequences) == 2
        # Biopython's Fasta parser usually takes the first word of the header as ID
        assert sequences[0].id == "SeqOne"
        # Biopython's description field for FASTA headers typically includes the full header line after '>'
        # but strips leading/trailing whitespace.
        assert sequences[0].description == "  SeqOne mixedCase" # Biopython strips trailing whitespace from description
        assert str(sequences[0].seq) == "ATGC"
        assert sequences[1].id == "SeqTwo"
        assert sequences[1].description == "SeqTwo" # Biopython strips trailing whitespace from description
        assert str(sequences[1].seq) == "CGTA"

    def test_read_fasta_with_numerical_ids(self, tmp_path: Path):
        """Test FASTA records where IDs are purely numerical."""
        content = ">12345\nATGC\n>007\nCGTA"
        fasta_file = tmp_path / "numerical_ids.fasta"
        fasta_file.write_text(content)
        sequences = io.read_fasta(str(fasta_file))
        assert len(sequences) == 2
        assert sequences[0].id == "12345"
        assert sequences[1].id == "007"
        
    def test_read_fasta_with_missing_sequence_data(self, tmp_path: Path, monkeypatch, caplog):
        """Test reading a FASTA file with a record that has missing sequence data."""
        caplog.set_level(logging.WARNING)
        
        # Create a test file
        fasta_file = tmp_path / "missing_seq.fasta"
        fasta_file.write_text(">NoSeq\nATGC\n")
        
        # Create a patched version of the hasattr function that returns False for 'seq'
        original_hasattr = hasattr
        def mock_hasattr(obj, name):
            if name == 'seq' and getattr(obj, 'id', None) == 'NoSeq':
                return False
            return original_hasattr(obj, name)
        
        # Apply the monkeypatch
        with monkeypatch.context() as m:
            m.setattr('builtins.hasattr', mock_hasattr)
            sequences = io.read_fasta(str(fasta_file))
        
        # Check that no sequences were returned
        assert len(sequences) == 0
        # The warning message should be in the log
        assert "has missing sequence data" in caplog.text
        
    def test_read_fasta_with_empty_sequence(self, tmp_path: Path, caplog):
        """Test reading a FASTA file with a record that has an empty sequence."""
        caplog.set_level(logging.WARNING)
        content = ">EmptySeq\n\n>ValidSeq\nATGC\n"
        fasta_file = tmp_path / "empty_seq.fasta"
        fasta_file.write_text(content)
        
        sequences = io.read_fasta(str(fasta_file))
        
        # Check that only the valid sequence was returned
        assert len(sequences) == 1
        assert sequences[0].id == "ValidSeq"
        assert "Sequence 'EmptySeq' in empty_seq.fasta is empty" in caplog.text
        
    def test_read_fasta_with_attribute_error(self, tmp_path: Path, monkeypatch, caplog):
        """Test handling of AttributeError during record processing."""
        caplog.set_level(logging.WARNING)
        
        # Create a test file with a sequence that will trigger an AttributeError
        # We'll use a malformed FASTA file that will cause Biopython to skip records
        # but still process one record that we can check
        fasta_file = tmp_path / "attr_error.fasta"
        fasta_file.write_text("Not a proper FASTA header\nATGC\n>ValidSeq\nATGC\n")
        
        # Read the file - the malformed header should be skipped
        sequences = io.read_fasta(str(fasta_file))
        
        # Check that only the valid sequence was returned
        assert len(sequences) == 1
        assert sequences[0].id == "ValidSeq"
        
        # Create a more specific test for the AttributeError handling
        # by creating a custom fixture that will be used in the test
        class MockSeqRecord:
            """A mock SeqRecord that raises AttributeError when seq is accessed."""
            def __init__(self, id):
                self.id = id
                
            @property
            def seq(self):
                raise AttributeError("Test attribute error")
                
            @property
            def description(self):
                return self.id
        
        # Create a test file
        attr_error_file = tmp_path / "attr_error_custom.fasta"
        attr_error_file.write_text(">AttrErrorSeq\nATGC\n")
        
        # Patch the SeqIO.parse function to return our mock record
        def mock_parse(handle, format):
            yield MockSeqRecord("AttrErrorSeq")
            
        # Use monkeypatch to replace SeqIO.parse with our mock
        with monkeypatch.context() as m:
            m.setattr(SeqIO, 'parse', mock_parse)
            
            # This should handle the AttributeError and log a warning
            sequences = io.read_fasta(str(attr_error_file))
            
            # Check that no sequences were returned and the warning was logged
            assert len(sequences) == 0
            assert "missing sequence data" in caplog.text.lower()
        
    def test_read_fasta_with_processing_exception(self, tmp_path: Path, monkeypatch, caplog):
        """Test handling of other exceptions during record processing."""
        caplog.set_level(logging.ERROR)
        
        # Create a test file with invalid content that will cause an exception
        fasta_file = tmp_path / "proc_error.fasta"
        fasta_file.write_text(">ProcErrorSeq\nATGC\n")
        
        # Create a mock version of SeqIO.parse that raises an exception
        def mock_parse(handle, format):
            raise Exception("Test processing error")
        
        # Apply the monkeypatch
        with monkeypatch.context() as m:
            m.setattr(SeqIO, 'parse', mock_parse)
            
            # This should catch the Exception and log an error
            sequences = io.read_fasta(str(fasta_file))
            
            # Check that no sequences were returned
            assert len(sequences) == 0
            assert "An unexpected error occurred" in caplog.text
        
    def test_read_fasta_with_value_error(self, tmp_path: Path, monkeypatch, caplog):
        """Test handling of ValueError during file parsing."""
        caplog.set_level(logging.ERROR)
        
        # Create a test file that would cause a ValueError
        fasta_file = tmp_path / "value_error.fasta"
        fasta_file.write_text("This is not a valid FASTA file format")
        
        # Mock function that raises ValueError during parsing
        def mock_seqio_parse(handle, format):
            raise ValueError("Test value error during parsing")
        
        # Use monkeypatch to replace SeqIO.parse with our mock
        with monkeypatch.context() as m:
            m.setattr(SeqIO, 'parse', mock_seqio_parse)
            sequences = io.read_fasta(str(fasta_file))
        
        # Check that the error was logged and no sequences were returned
        assert len(sequences) == 0
        assert "Error parsing FASTA file" in caplog.text
        
    # Note: We're removing the test_read_fasta_specific_attribute_error test
    # because the io.py implementation doesn't actually raise an AttributeError
    # when accessing record.description - Biopython's SeqRecord has a default
    # implementation that returns the ID if description is not explicitly set.

    # This test reuses a fixture from conftest.py
    def test_read_fasta_invalid_type_in_fixture(self, dummy_fasta_file_invalid_type, caplog):
        """
        Test reading a FASTA file that contains sequences with invalid characters
        (as defined by the 'dna' alphabet in Biopython if it were strict, but FASTA is lenient).
        Our io.read_fasta doesn't do sequence validation itself, Bio.SeqIO.parse does.
        This test primarily ensures it reads what Biopython provides.
        """
        caplog.set_level(logging.DEBUG)
        sequences = io.read_fasta(str(dummy_fasta_file_invalid_type))
        assert len(sequences) == 1
        assert sequences[0].id == "InvalidChars"
        # FASTA format itself allows any characters in the sequence part.
        # Validation happens later (e.g., in utils.clean_and_filter_sequences)
        caplog.set_level(logging.INFO) # Changed to INFO
        assert str(sequences[0].seq) == "ATGXXXCGT"
        assert f"Successfully read {len(sequences)} sequences from {str(dummy_fasta_file_invalid_type)}" in caplog.text # Updated log message to full path