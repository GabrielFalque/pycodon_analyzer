# tests/test_cli.py
from pathlib import Path
from typing import Any, List, Literal, Tuple
import pytest # type: ignore
import os
import sys
import subprocess # For running as command-line
import pandas as pd
import logging # <-- Add logging
from unittest.mock import patch, MagicMock # For mocking sys.argv and sys.exit

# Adjust import path
try:
    from src.pycodon_analyzer import cli, utils # type: ignore
    MP_AVAILABLE = cli.MP_AVAILABLE
    # RICH_AVAILABLE is part of the cli module, we will patch it
except ImportError:
     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
     from pycodon_analyzer import cli, utils
     MP_AVAILABLE = cli.MP_AVAILABLE

@pytest.mark.parametrize(
    "filename, expected_name",
    [
        ("gene_ORF1ab.fasta", "ORF1ab"),
        ("gene_S.fa", "S"),
        ("gene_N.fna", "N"),
        ("gene_complex-name_v2.fas", "complex-name_v2"),
        ("gene_E.faa", "E"),
        ("gene_1.2.fasta", "1.2"),
        # --- Regex should match '_' ---
        ("gene_.fasta", None),
        ("not_a_gene.fasta", None),
        ("gene_NoExt", None),
        ("prefix_gene_A.fasta", None),
        ("gene_trailing_dot.fasta.", None), # Doesn't end with valid extension
        ("gene_name.with.dots.fasta", "name.with.dots"),
        ("gene_name-with-hyphens.fasta", "name-with-hyphens"),
    ]
)

# --- Helper function to assert log messages ---
def assert_log_message(caplog_records: List[logging.LogRecord], level: int, message_substring: str):
    """
    Helper to assert specific log message and level from caplog.records.
    """
    found = any(
        message_substring in rec.message and rec.levelno == level
        for rec in caplog_records
    )
    if not found:
        all_captured_messages = "\n".join([f"[{logging.getLevelName(rec.levelno)}] {rec.name}: {rec.message}" for rec in caplog_records])
        pytest.fail(
            f"Expected log (level {logging.getLevelName(level)}) containing '{message_substring}' not found. "
            f"Captured logs:\n{all_captured_messages}\n--- End caplog ---"
        )

# --- Integration tests for cli.main ---

# Helper to create dummy gene files
def create_dummy_gene_file(dir_path, gene_name, seq_dict):
    filepath = dir_path / f"gene_{gene_name}.fasta"
    with open(filepath, "w") as f:
        for seq_id, seq_str in seq_dict.items():
            f.write(f">{seq_id}\n{seq_str}\n")
    return filepath

@pytest.fixture
def setup_input_dir(tmp_path: Path):
    """Creates a temporary input directory with dummy gene files."""
    input_dir = tmp_path / "input_genes"
    input_dir.mkdir()
    # Gene A: Simple valid sequences
    create_dummy_gene_file(input_dir, "A", {
        "Seq1": "ATGCCGCGTTAG", # Len 12 -> CCGCGT (OK)
        "Seq2": "ATGAAACCCTGA", # Len 12 -> AAACCC (OK)
    })
    # Gene B: Mixed sequences (some will be filtered)
    create_dummy_gene_file(input_dir, "B", {
        "Seq1": "ATGCGTAGA---",  # Len 12 -> CGTAGA (OK)
        "Seq2": "ATGNNNTTTTAG",  # Len 12 -> NNNTTT (OK if ambiguity <= 25%)
        "Seq3": "ATGTAG",        # Len 6 -> '' (Removed - too short after trim)
        "Seq4": "ACGTACGT",      # Len 8 -> (Removed - not multiple of 3)
    })
    # Gene C: All sequences invalid after cleaning
    create_dummy_gene_file(input_dir, "C", {
        "Seq1": "ATGC",          # Removed - not multiple of 3
        "Seq2": "NNNNNNNNNNNN",  # Removed - ambiguity
    })
    return input_dir


def run_cli(args_list: List[str], monkeypatch) -> Tuple[int, str, str]:
    """
    Helper function to run cli.main with patched sys.argv and sys.exit.
    """
    full_args = ["pycodon_analyzer"] + args_list
    exit_code = -999 # Sentinel
    captured_stdout, captured_stderr = "", ""

    monkeypatch.setattr(sys, "argv", full_args)
    def mock_exit(code=0):
        nonlocal exit_code
        if exit_code == -999: exit_code = code # Set only if not already set by previous SystemExit
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", mock_exit)

    try:
        cli.main() # This will execute your cli.py main function, which configures RichHandler
        if exit_code == -999: exit_code = 0 # Success if no SystemExit caught by mock_exit
    except SystemExit as e:
        if exit_code == -999: exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        # This part is for UNEXPECTED errors in cli.main, not for sys.exit
        logging.getLogger("test_cli_runner").exception(f"Unexpected error during CLI run helper: {e}")
        captured_stderr = str(e)
        exit_code = 1 # Indicate failure
    
    if not isinstance(exit_code, int): exit_code = 1 # Default to 1 if exit_code is weird
    return exit_code, captured_stdout, captured_stderr

# --- Integration Test Cases ---

def test_cli_integration_success_basic(setup_input_dir: Path, tmp_path: Path, monkeypatch, caplog: Any): # [Source 7]
    """Test a basic successful run of the 'analyze' subcommand."""
    # caplog automatically captures logs from the root logger and its children.
    # The level set here applies to what caplog itself records.
    caplog.set_level(logging.INFO) # [Source 179]

    input_dir = setup_input_dir
    output_dir = tmp_path / "cli_output_analyze_basic"
    args = ["analyze", "-d", str(input_dir), "-o", str(output_dir), "--ref", "none", "--skip_ca", "--skip_plots"] # [Source 180]
    # The run_cli helper no longer takes caplog and desired_log_level
    exit_code, _, errors = run_cli(args, monkeypatch) # [Source 181]

    assert exit_code == 0, f"CLI 'analyze' exited with {exit_code}. Errors: {errors}" # [Source 181]
    assert output_dir.exists(), "Output directory was not created."
    assert (output_dir / "per_sequence_metrics_all_genes.csv").is_file(), "Combined metrics CSV missing."
    assert (output_dir / "mean_features_per_gene.csv").is_file(), "Mean features CSV missing."
    assert (output_dir / "gene_comparison_stats.csv").is_file(), "Stats comparison CSV missing."

    assert_log_message(caplog.records, logging.INFO, "PyCodon Analyzer - Command: analyze") # [Source 182]
    assert_log_message(caplog.records, logging.INFO, "Running 'analyze' command with input directory")
    assert_log_message(caplog.records, logging.INFO, "Found 3 potential gene files")
    assert_log_message(caplog.records, logging.INFO, "Expecting data for 3 genes:")
    assert_log_message(caplog.records, logging.WARNING, "No valid sequences remaining after cleaning for gene C") # [Source 200]
    assert_log_message(caplog.records, logging.WARNING, "Processed 2 genes out of 3 expected.") # [Source 201]
    assert_log_message(caplog.records, logging.INFO, "Running analysis on 1 valid 'complete' sequence record") # [Source 202]
    assert_log_message(caplog.records, logging.INFO, "Skipping combined plot generation as requested.") # [Source 214]
    assert_log_message(caplog.records, logging.INFO, "Skipping combined Correspondence Analysis as requested.") # [Source 212]
    assert_log_message(caplog.records, logging.INFO, "PyCodon Analyzer run finished successfully.") # [Source 215]


def test_cli_integration_parallel(setup_input_dir: Path, tmp_path: Path, monkeypatch, caplog: Any): # [Source 41]
    """Test 'analyze' subcommand running with multiple threads."""
    if not cli.MP_AVAILABLE: pytest.skip("multiprocessing not available")

    with caplog.at_level(logging.INFO, logger="pycodon_analyzer"):
        caplog.set_level(logging.INFO) # [Source 217]

        input_dir = setup_input_dir
        output_dir = tmp_path / "cli_output_analyze_parallel"
        args = ["analyze", "-d", str(input_dir), "-o", str(output_dir), "--ref", "none", "--skip_ca", "--skip_plots", "-t", "0"] # [Source 218]
        exit_code, _, errors = run_cli(args, monkeypatch) # [Source 218]
    # Removed duplicate run_cli call

    assert exit_code == 0, f"CLI 'analyze' parallel exited with {exit_code}. Errors: {errors}" # [Source 219]
    assert output_dir.exists()
    assert_log_message(caplog.records, logging.INFO, "processes for parallel gene file analysis") # [Source 225]


def test_cli_integration_input_dir_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any): # [Source 81]
    """Test 'analyze' error handling for non-existent input directory."""
    with caplog.at_level(logging.ERROR, logger="pycodon_analyzer"):
        input_dir = tmp_path / "nonexistent_dir_for_analyze"
        output_dir = tmp_path / "cli_output_analyze_error"
        args = ["analyze", "-d", str(input_dir), "-o", str(output_dir)] # [Source 281]
        exit_code, _, _ = run_cli(args, monkeypatch)

    assert exit_code != 0
    assert_log_message(caplog.records, logging.ERROR, f"Input directory for analysis not found: {input_dir}") # [Source 82]

def test_cli_integration_no_gene_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any): # [Source 84]
    """Test 'analyze' error handling when no gene files are found."""
    with caplog.at_level(logging.ERROR, logger="pycodon_analyzer"):
        input_dir = tmp_path / "empty_input_for_analyze"
        input_dir.mkdir()
        output_dir = tmp_path / "cli_output_analyze_error_no_files"
        args = ["analyze", "-d", str(input_dir), "-o", str(output_dir)] # [Source 296]
        exit_code, _, _ = run_cli(args, monkeypatch)

    assert exit_code != 0
    assert_log_message(caplog.records, logging.ERROR, "No gene alignment files") # [Source 296]
    assert_log_message(caplog.records, logging.ERROR, f"found in directory: {input_dir}")



# --- Tests for subcommand 'extract' ---
@pytest.fixture
def setup_extract_files(tmp_path: Path) -> Tuple[Path, Path]: # [Source 950]
    annotations_file = tmp_path / "ref_annotations.fasta"
    alignment_file = tmp_path / "genome_alignment.fasta"
    annotations_content = (
        ">RefSeq1_feature1 [gene=GeneX] [location=4..12]\n"
        "ATGCATGCAT\n"
        ">RefSeq1_feature2 [locus_tag=GeneY] [location=complement(15..20)]\n"
        "CGTACG\n"
        ">RefSeq1_feature3 [gene=GeneZ_skipped] [location=25..35]\n" # Changed name for clarity
        "AAAA\n"
    )
    annotations_file.write_text(annotations_content)
    alignment_content = ( # [Source 951]
        ">Ref_ID_in_MSA\n"
        "---ATGCCG---TTAG----C--G--T--A--GCCATT---\n"
        ">Sample1\n"
        "---ATGNNG---TTAG----C--G--T--A--GCCANN---\n"
        ">Sample2\n"
        "---ATGCCG---TTAG----C--G--T--A--NNNATT---\n"
    )
    alignment_file.write_text(alignment_content)
    return annotations_file, alignment_file

def test_cli_integration_extract_success(setup_extract_files: Tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any): # [Source 88]
    """Test a basic successful run of the 'extract' subcommand."""
    with caplog.at_level(logging.INFO, logger="pycodon_analyzer"):
        annotations_file, alignment_file = setup_extract_files
        output_dir_extract = tmp_path / "extracted_genes_output_success"
        args = ["extract", "-a", str(annotations_file), "-g", str(alignment_file), "-r", "Ref_ID_in_MSA", "-o", str(output_dir_extract)] # [Source 313]
        exit_code, _, errors = run_cli(args, monkeypatch)

    assert exit_code == 0, f"CLI 'extract' exited with code {exit_code}. Errors: {errors}" # [Source 313]
    # ... (file assertions as before) ...

    with caplog.at_level(logging.INFO, logger="src.pycodon_analyzer.extraction"):
         pass
    assert_log_message(caplog.records, logging.INFO, "PyCodon Analyzer - Command: extract") # [Source 310]
    assert_log_message(caplog.records, logging.INFO, "'extract' command finished successfully.")
    assert_log_message(caplog.records, logging.INFO, "Gene alignments written: 2")


def test_cli_integration_extract_ref_id_not_found(setup_extract_files: Tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any): # [Source 112]
    """Test 'extract' error handling when ref_id is not in alignment."""
    with caplog.at_level(logging.ERROR, logger="pycodon_analyzer"): # Pour les logs de handle_extract_command
        with caplog.at_level(logging.ERROR, logger="src.pycodon_analyzer.extraction"):
            annotations_file, alignment_file = setup_extract_files
            output_dir_extract = tmp_path / "extracted_genes_error_ref_id"
            args = ["extract", "-a", str(annotations_file), "-g", str(alignment_file), "-r", "WRONG_Ref_ID", "-o", str(output_dir_extract)] # [Source 330]
            exit_code, _, _ = run_cli(args, monkeypatch)
    assert exit_code != 0
    # The actual error logged by extraction.py is more specific
    assert_log_message(caplog.records, logging.ERROR, "Reference sequence ID 'WRONG_Ref_ID' not found in alignment") # [Source 330]
    # And the handler in cli.py logs this:
    assert_log_message(caplog.records, logging.ERROR, "Extraction error: Reference sequence ID 'WRONG_Ref_ID' not found")


def test_cli_integration_extract_annotations_not_found(setup_extract_files: Tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any): # [Source 121]
    """Test 'extract' error handling when annotations file is not found."""
    with caplog.at_level(logging.ERROR, logger="pycodon_analyzer"):
        with caplog.at_level(logging.ERROR, logger="src.pycodon_analyzer.extraction"):
            _, alignment_file = setup_extract_files
            annotations_file_non_existent = tmp_path / "non_existent_annotations.fasta"
            output_dir_extract = tmp_path / "extracted_genes_error_annot"
            args = ["extract", "-a", str(annotations_file_non_existent), "-g", str(alignment_file), "-r", "Ref_ID_in_MSA", "-o", str(output_dir_extract)] # [Source 345]
            exit_code, _, _ = run_cli(args, monkeypatch)
    assert exit_code != 0
    assert_log_message(caplog.records, logging.ERROR, f"Annotation file not found: {annotations_file_non_existent}") # [Source 345]
    assert_log_message(caplog.records, logging.ERROR, "Extraction error: Annotation file not found")