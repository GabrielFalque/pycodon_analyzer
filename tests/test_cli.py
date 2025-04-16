# tests/test_cli.py
import pytest # type: ignore
import os
import sys
import subprocess # For running as command-line
import pandas as pd
import logging # <-- Add logging
from unittest.mock import patch # For mocking sys.argv and sys.exit

# Adjust import path
try:
    from pycodon_analyzer import cli
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from pycodon_analyzer import cli


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
def test_extract_gene_name_from_file(filename, expected_name):
    """Test extracting gene names from various filename patterns."""
    # Re-check regex in cli.py: r'gene_([\w\-.]+?)\.(fasta|fa|fna|fas|faa)$'
    # [\w\-.]+? does match '_'. The issue might be how pytest reports the failure
    # or a subtle environment thing. Let's trust the regex and keep the test.
    # If it still fails, add print(f"Match object: {re.match(...)}") inside the helper.
    assert cli.extract_gene_name_from_file(filename) == expected_name

# --- Integration tests for cli.main ---

# Helper to create dummy gene files
def create_dummy_gene_file(dir_path, gene_name, seq_dict):
    filepath = dir_path / f"gene_{gene_name}.fasta"
    with open(filepath, "w") as f:
        for seq_id, seq_str in seq_dict.items():
            f.write(f">{seq_id}\n{seq_str}\n")
    return filepath

@pytest.fixture
def setup_input_dir(tmp_path):
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


def run_cli(args_list, monkeypatch):
    """Helper function to run cli.main with patched sys.argv and sys.exit."""
    # Prepend script name (can be anything, often sys.executable or module path)
    full_args = ["pycodon_analyzer"] + args_list
    exit_code = None
    output = ""
    errors = ""

    # Patch sys.argv
    monkeypatch.setattr(sys, "argv", full_args)

    # Patch sys.exit to capture exit code
    def mock_exit(code=0):
        nonlocal exit_code
        exit_code = code
        # Raise an exception to stop execution cleanly after exit is called
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", mock_exit)

    try:
        # Call the main function directly (ensure imports allow this)
        cli.main()
        # If main completes without SystemExit, assume exit code 0
        if exit_code is None: exit_code = 0
    except SystemExit as e:
        # Capture the exit code from the raised exception
        if exit_code is None: exit_code = e.code
    except Exception as e:
         # Capture unexpected errors during execution
         errors = str(e)
         exit_code = -1 # Indicate failure

    return exit_code, output, errors # Output/errors might need capturing stdout/stderr


# --- Integration Test Cases ---


def test_cli_integration_success_basic(setup_input_dir, tmp_path, monkeypatch, caplog):
    """Test a basic successful run with default options (excluding ref)."""
    caplog.set_level(logging.INFO) # Capture INFO level messages
    input_dir = setup_input_dir
    output_dir = tmp_path / "cli_output"
    # Basic arguments for a successful run - CORRECTED underscores
    args = ["-d", str(input_dir), "-o", str(output_dir), "--ref", "none", "--skip_ca", "--skip_plots"] # <-- Utiliser --skip_ca et --skip_plots

    exit_code, _, errors = run_cli(args, monkeypatch)

    assert exit_code == 0, f"CLI exited with non-zero code {exit_code}. Errors: {errors}"
    assert output_dir.exists(), "Output directory was not created."
    # Check expected output files are created
    assert (output_dir / "per_sequence_metrics_all_genes.csv").is_file(), "Combined metrics CSV missing."
    assert (output_dir / "mean_features_per_gene.csv").is_file(), "Mean features CSV missing."
    assert (output_dir / "gene_comparison_stats.csv").is_file(), "Stats comparison CSV missing."
    
    assert "Starting PyCodon Analyzer run." in caplog.text
    assert "Found 3 potential gene alignment files" in caplog.text
    assert "Expecting data for 3 genes:" in caplog.text and "A" in caplog.text and "B" in caplog.text and "C" in caplog.text
    
    assert "Processing 3 gene files using 1 processes..." in caplog.text or "Using 1 process (sequential gene file analysis)." in caplog.text
    assert "Starting processing for gene: A" in caplog.text
    assert "Starting processing for gene: B" in caplog.text
    assert "Starting processing for gene: C" in caplog.text

    # Check for the key part of the warning message, more robust
    assert any(record.levelname == 'WARNING' for record in caplog.records if "cleaning for gene C" in record.message)
    
    assert "Processed 2 genes out of 3 expected." in caplog.text
    
    assert "Running analysis on 1 valid 'complete' sequence records" in caplog.text
    assert "Combining final results..." in caplog.text
    assert "Calculating mean features per gene..." in caplog.text
    assert "Performing statistical comparison between genes..." in caplog.text
    assert "Skipping combined plot generation as requested." in caplog.text # <-- Ajout de " as requested."
    assert "Skipping combined Correspondence Analysis as requested." in caplog.text

    assert "PyCodon Analyzer run finished successfully." in caplog.text

    # Check content of one output file (basic check)
    try:
        metrics_df = pd.read_csv(output_dir / "per_sequence_metrics_all_genes.csv")
        assert 'ID' in metrics_df.columns
        assert 'Gene' in metrics_df.columns
        # Expected sequences surviving: A/Seq1, A/Seq2, B/Seq1, B/Seq2, complete/Seq1, complete/Seq2
        assert len(metrics_df) == 4, f"Expected 4 rows in combined metrics, found {len(metrics_df)}"
        assert set(metrics_df['Gene'].unique()) == {'A', 'B', 'complete'}, "Unexpected genes in combined metrics"
    except FileNotFoundError:
        pytest.fail("Output CSV file not found for validation.")
    except Exception as e:
        pytest.fail(f"Failed to read or validate output CSV: {e}")


def test_cli_integration_parallel(setup_input_dir, tmp_path, monkeypatch, caplog):
    """Test running with multiple threads."""
    # Only run if multiprocessing is available
    if not cli.MP_AVAILABLE:
        pytest.skip("multiprocessing not available, skipping parallel test")

    caplog.set_level(logging.INFO)
    input_dir = setup_input_dir
    output_dir = tmp_path / "cli_output_parallel"
    # Use -t 0 to request all cores (or at least > 1)
    args = ["-d", str(input_dir), "-o", str(output_dir), "--ref", "none", "--skip_ca", "--skip_plots", "-t", "0"]

    exit_code, _, _ = run_cli(args, monkeypatch)

    assert exit_code == 0
    assert output_dir.exists()
    assert (output_dir / "per_sequence_metrics_all_genes.csv").is_file()
    # Check log indicates parallel execution (number might vary)
    assert "processes for parallel gene file analysis" in caplog.text

def test_cli_integration_input_dir_not_found(tmp_path, monkeypatch, caplog):
    """Test error handling when input directory does not exist."""
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "nonexistent_dir"
    output_dir = tmp_path / "cli_output_error"
    args = ["-d", str(input_dir), "-o", str(output_dir)]

    exit_code, _, _ = run_cli(args, monkeypatch)

    assert exit_code != 0 # Expect non-zero exit code
    assert f"Input directory not found: {input_dir}" in caplog.text

def test_cli_integration_no_gene_files(tmp_path, monkeypatch, caplog):
    """Test error handling when no gene files are found."""
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "empty_input"
    input_dir.mkdir() # Create empty dir
    output_dir = tmp_path / "cli_output_error"
    args = ["-d", str(input_dir), "-o", str(output_dir)]

    exit_code, _, _ = run_cli(args, monkeypatch)

    assert exit_code != 0
    assert "No gene alignment files" in caplog.text
    assert f"found in directory: {input_dir}" in caplog.text

# Add more integration tests:
# - Test with a valid reference file (--ref path/to/ref) and check CAI etc. columns
# - Test --skip-plots (check plot files NOT created, check log message)
# - Test --skip-ca (check CA files NOT created, check log message)
# - Test invalid --max-ambiguity value
# - Test case where NO sequences survive cleaning in ANY gene file