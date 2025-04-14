# tests/test_io.py
from typing import List
from Bio.SeqRecord import SeqRecord
import pytest # type: ignore
import os
import logging

# Adjust import based on how pytest is run (from root or tests dir)
try:
    from src.pycodon_analyzer import io # type: ignore
    from src.pycodon_analyzer.utils import VALID_DNA_CHARS # type: ignore # Import needed constant
except ImportError:
     import sys
     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
     from pycodon_analyzer import io
     from pycodon_analyzer.utils import VALID_DNA_CHARS


# --- Helper function to create dummy FASTA ---
def create_dummy_fasta(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)

# --- Test error handling and logging in read_fasta ---

def test_read_fasta_file_not_found(caplog):
    """Test FileNotFoundError is raised and logged."""
    caplog.set_level(logging.ERROR)
    filepath = "nonexistent_fasta.fa"
    with pytest.raises(FileNotFoundError):
        io.read_fasta(filepath)
    assert f"Input FASTA file not found: '{filepath}'" in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)

def test_read_fasta_empty_file(tmp_path, caplog):
    """Test ValueError is raised for an empty file."""
    caplog.set_level(logging.ERROR)
    filepath = tmp_path / "empty.fasta"
    create_dummy_fasta(filepath, "")
    with pytest.raises(ValueError, match="No valid sequences found"):
        io.read_fasta(str(filepath))
    assert f"No valid sequences found in FASTA file: '{filepath}'" in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)

# --- Test for specific malformed FASTA (leading sequence data) ---
def test_read_fasta_malformed_leading_sequence(tmp_path, caplog):
    """Test FASTA reading when sequence data precedes the first header.
       Biopython's parser should ignore leading data and parse subsequent valid records.
    """
    caplog.set_level(logging.INFO) # Capture INFO and WARNING for checks
    filepath = tmp_path / "malformed_leading_seq.fasta"
    # Malformed content: sequence data before the first header, followed by valid record
    create_dummy_fasta(filepath, "ACGTACGT\n>Seq1\nATGC")

    # Expect the function to succeed by ignoring leading data and parsing Seq1
    # Therefore, DO NOT expect a ValueError here.
    try:
        records: List[SeqRecord] = io.read_fasta(str(filepath))
    except ValueError as e:
        # Fail the test if a ValueError *is* raised unexpectedly
        pytest.fail(f"io.read_fasta raised unexpected ValueError: {e}")

    # Assert that one valid record was found
    assert len(records) == 1, "Expected one record to be parsed successfully."
    assert records[0].id == "Seq1", "Parsed record ID does not match."
    # Check sequence content (should be converted to upper by read_fasta)
    assert str(records[0].seq) == "ATGC", "Parsed record sequence does not match."

    # Check that no ERROR messages were logged for this case
    error_logs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert not error_logs, f"Expected no ERROR logs, but found: {[rec.message for rec in error_logs]}"
    # Ensure the specific "No valid sequences found" error was NOT logged
    assert "No valid sequences found" not in caplog.text

def test_read_fasta_empty_sequences(tmp_path, caplog):
    """Test skipping empty sequences and logging warnings."""
    caplog.set_level(logging.WARNING)
    filepath = tmp_path / "empty_seqs.fasta"
    content = ">Seq1\n\n>Seq2\nACGT\n>Seq3\n\n>Seq4\nTGCA"
    create_dummy_fasta(filepath, content)

    records = io.read_fasta(str(filepath))

    # Should only return Seq2 and Seq4
    assert len(records) == 2
    record_ids = {rec.id for rec in records}
    assert record_ids == {"Seq2", "Seq4"}
    # Check that warnings were logged for Seq1 and Seq3
    assert f"Sequence 'Seq1' in {os.path.basename(filepath)} is empty. Skipping." in caplog.text
    assert f"Sequence 'Seq3' in {os.path.basename(filepath)} is empty. Skipping." in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)

def test_read_fasta_invalid_chars(tmp_path, caplog):
    """Test warning for invalid characters (not in VALID_DNA_CHARS)."""
    caplog.set_level(logging.WARNING)
    filepath = tmp_path / "invalid_chars.fasta"
    # Assuming 'X' and '?' are not in VALID_DNA_CHARS set in utils.py
    content = ">Seq1\nACGT-N\n>Seq2\nACGXT?\n>Seq3\nTGCA"
    create_dummy_fasta(filepath, content)

    records = io.read_fasta(str(filepath))

    assert len(records) == 3 # Should still read all records
    # Check for the warning specifically about Seq2
    assert f"Sequence 'Seq2' in {os.path.basename(filepath)} contains potentially invalid characters" in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    # Ensure no warning for Seq1 or Seq3 if they only contain valid chars
    assert f"Sequence 'Seq1' in {os.path.basename(filepath)} contains potentially invalid characters" not in caplog.text
    assert f"Sequence 'Seq3' in {os.path.basename(filepath)} contains potentially invalid characters" not in caplog.text