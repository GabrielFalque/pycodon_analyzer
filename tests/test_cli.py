# tests/test_cli.py
import pytest
import os
import sys
import io
import subprocess # For testing the CLI entry point directly
from src.pycodon_analyzer import utils # Import utils module for sanitize_filename
from pathlib import Path
import logging
import shutil # For cleaning up test directories if needed, though tmp_path handles most
import pandas as pd # For creating dummy metadata/ref files

# Ensure the src directory is in the Python path for tests
# This allows running pytest from the root directory
try:
    from pycodon_analyzer import cli
except ImportError: # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.pycodon_analyzer import cli # Assuming src is the root for the package pycodon_analyzer


# --- Helper to run CLI commands ---
def run_cli_command(cmd_args: list[str], expect_success: bool = True) -> subprocess.CompletedProcess:
    """Runs the pycodon_analyzer CLI command as a subprocess."""
    # For testing, we'll use a direct call to cli.main with monkeypatched sys.argv
    # This is more reliable than subprocess for unit tests
    with pytest.MonkeyPatch.context() as mp:
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        mp.setattr(sys, "stdout", stdout_buffer)
        mp.setattr(sys, "stderr", stderr_buffer)
        mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
        
        # Run the command and catch SystemExit
        returncode = 0
        try:
            cli.main()
        except SystemExit as e:
            returncode = e.code if isinstance(e.code, int) else 1
        
        # Create a CompletedProcess-like object for compatibility
        process = subprocess.CompletedProcess(
            args=["pycodon_analyzer"] + cmd_args,
            returncode=returncode,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue()
        )
        
    if expect_success and process.returncode != 0:
        print(f"CLI command failed! Args: {cmd_args}")
        print("STDOUT:")
        print(process.stdout)
        print("STDERR:")
        print(process.stderr)
        assert process.returncode == 0, "CLI command was expected to succeed."
    elif not expect_success and process.returncode == 0:
        print(f"CLI command succeeded but was expected to fail! Args: {cmd_args}")
        print("STDOUT:")
        print(process.stdout)
        assert process.returncode != 0, "CLI command was expected to fail."
    return process

# --- Fixtures for CLI Tests ---

@pytest.fixture
def dummy_gene_fasta_content() -> str:
    return ">GeneA_Seq1\nATGCGTTAA\n>GeneA_Seq2\nATGCCCTAG\n"

@pytest.fixture
def dummy_gene_dir(tmp_path: Path, dummy_gene_fasta_content: str) -> Path:
    gene_dir = tmp_path / "genes_input"
    gene_dir.mkdir()
    (gene_dir / "gene_GeneA.fasta").write_text(dummy_gene_fasta_content)
    (gene_dir / "gene_GeneB.fasta").write_text(">GeneB_Seq1\nTTTCCCGGG\n")
    return gene_dir

@pytest.fixture
def dummy_ref_usage_file(tmp_path: Path) -> Path:
    content = "Codon\tFrequency\nAAA\t0.6\nAAG\t0.4\nATG\t1.0\n"
    ref_file = tmp_path / "dummy_ref.tsv"
    ref_file.write_text(content)
    return ref_file

@pytest.fixture
def dummy_metadata_file(tmp_path: Path) -> Path:
    content = "seq_id,Category,Value\nGeneA_Seq1,X,10\nGeneA_Seq2,Y,20\nGeneB_Seq1,X,30\n"
    meta_file = tmp_path / "metadata.csv"
    meta_file.write_text(content)
    return meta_file

@pytest.fixture
def dummy_annotation_file_for_extract(tmp_path: Path) -> Path:
    # Create a FASTA file with annotations in the description that can be parsed by extraction.py
    content = """>RefAnnotationSeq [gene=GeneX] [location=10..18]
GATCCTCCATATACAACCTACCTGTCTACCTAATCCTTCTTCTCCTCCATCCTCAATCCT
>RefAnnotationSeq [gene=GeneY] [location=complement(30..38)]
GATCCTCCATATACAACCTACCTGTCTACCTAATCCTTCTTCTCCTCCATCCTCAATCCT
"""
    anno_file = tmp_path / "annotations.fasta"
    anno_file.write_text(content)
    return anno_file


@pytest.fixture
def dummy_alignment_file_for_extract(tmp_path: Path) -> Path:
    content = """>RefAnnotationSeq
GATTACAGATTACAGATTACACCDSXXXYYYGATTACAGATTACACCDSXXXYYYGATTACAGATTACACCDSXXXYYYGATTACAGATTACAGATTACA
>OrgA
GATTACAGATTACAGATTACACCDSAAAYYYGATTACAGATTACACCDSAAAYYYGATTACAGATTACACCDSAAAYYYGATTACAGATTACAGATTACA
>OrgB
GATTACAGATTACAGATTACACCDSCCCYYYGATTACAGATTACACCDSCCCYYYGATTACAGATTACACCDSCCCYYYGATTACAGATTACAGATTACA
"""
    # Replace CDSXXXYYY with actual sequences based on GenBank locations
    # GeneX: 10..18 (9bp) -> Ref: GATTACAGA, OrgA: GATTACAGA, OrgB: GATTACAGA
    # GeneY: 30..38 (9bp) -> Ref: TTACACCDS, OrgA: TTACACCDS, OrgB: TTACACCDS (example)
    # The sequence needs to be long enough and have the genes.
    # Ref: ---GATTACAGA---GATTACACCDS---
    #      (9)  (9)    (9)   (9)     (9)
    #      0        10        20        30        40
    # GeneX: 10-18 => index 9 to 17
    # GeneY: 30-38 => index 29 to 37
    ref_seq_str = "AAAAAAAAAGATTACAGABBBBBBBBBBGATTACACCDDDDDDDDDDEEEEEEEEEE" # Len 50
    org_a_seq_str = "AAAAAAAAAGATTACAGABBBBBBBBBBGATTACACCEEEEEEEEEFFFFFFFFFF"
    org_b_seq_str = "AAAAAAAAAGATTACAGABBBBBBBBBBTAGCATTAGGGGGGGGGGHHHHHHHHHH"

    content = f""">RefAnnotationSeq
{ref_seq_str}
>OrgA
{org_a_seq_str}
>OrgB
{org_b_seq_str}
"""
    align_file = tmp_path / "alignment.fasta"
    align_file.write_text(content)
    return align_file


# --- Test Class for `analyze` subcommand ---
class TestAnalyzeCommand:

    def test_analyze_basic_run(self, dummy_gene_dir: Path, tmp_path: Path, caplog):
        """Test a basic successful run of the analyze command."""
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "analyze_output_basic"
        log_file_name = "test_run.log"

        cmd_args = [
            "--log-file", log_file_name,
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--ref", "human" # Use built-in human reference
        ]
        # Using direct cli.main call for easier debugging and no subprocess overhead for now
        # Patch sys.argv for the cli.main() call
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()

        assert output_dir.is_dir()
        assert (output_dir / log_file_name).exists()
        assert (output_dir / "README.txt").exists()
        assert (output_dir / "report.html").exists()
        assert (output_dir / "html").is_dir()
        assert (output_dir / "data").is_dir()
        assert (output_dir / "images").is_dir()

        # Check for some key output files
        assert (output_dir / "data" / "per_sequence_metrics_all_genes.csv").exists()
        assert (output_dir / "data" / "mean_features_per_gene.csv").exists()
        # One plot example (format might vary based on defaults, assume svg if not specified)
        # Assuming default plot format is svg as per recent cli.py changes
        plot_files = list((output_dir / "images").glob("*.svg")) # Default plot format is svg
        assert len(plot_files) > 0, "No SVG plot files found in images directory"

        # Check the log file for the success message
        log_file = output_dir / log_file_name
        if log_file.exists():
            log_content = log_file.read_text()
            assert "PyCodon Analyzer 'analyze' command finished successfully." in log_content

    def test_analyze_no_html_report(self, dummy_gene_dir: Path, tmp_path: Path):
        output_dir = tmp_path / "analyze_output_no_html"
        cmd_args = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--no-html-report" # Key flag to test
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()
        
        assert not (output_dir / "report.html").exists()
        assert not (output_dir / "html").exists()
        assert (output_dir / "README.txt").exists() # README should still be there
        assert (output_dir / "data").is_dir() # Data should still be generated

    def test_analyze_skip_plots(self, dummy_gene_dir: Path, tmp_path: Path, caplog):
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "analyze_output_no_plots"
        cmd_args = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--skip-plots"
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()

        # The directory might be created but should be empty or only contain placeholder files
        if (output_dir / "images").exists():
            # Check if the directory is empty or only contains placeholder files
            image_files = list((output_dir / "images").glob("*.svg"))
            assert len(image_files) == 0, "Images directory should not contain any plot files"
        
        # Check the log file for the skip plots message
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Skipping generation of all combined plots as requested (--skip_plots)." in log_content
        # HTML report might still be generated but without plot links/images
        assert (output_dir / "report.html").exists()


    def test_analyze_invalid_input_dir(self, tmp_path: Path, caplog):
        output_dir = tmp_path / "analyze_output_invalid_input"
        non_existent_input_dir = tmp_path / "non_existent_genes"
        cmd_args = [
            "analyze",
            "--directory", str(non_existent_input_dir),
            "--output", str(output_dir)
        ]
        
        # We expect the command to exit with an error
        with pytest.raises(SystemExit) as e:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
                cli.main()
            assert e.value.code != 0  # Check for non-zero exit code
        
        # The error should be logged to the log file if it was created
        if output_dir.exists() and (output_dir / "pycodon_analyzer.log").exists():
            log_content = (output_dir / "pycodon_analyzer.log").read_text()
            assert "error" in log_content.lower() and "directory" in log_content.lower()


    def test_analyze_with_metadata(self, dummy_gene_dir: Path, dummy_metadata_file: Path, tmp_path: Path, caplog):
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "analyze_output_with_meta"
        cmd_args = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--metadata", str(dummy_metadata_file),
            "--metadata_id_col", "seq_id", # Matches fixture
            "--color_by_metadata", "Category" # Matches fixture
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()

        assert (output_dir / "data" / "per_sequence_metrics_all_genes.csv").exists()
        
        # Check if the output CSV exists and can be read
        results_df = pd.read_csv(output_dir / "data" / "per_sequence_metrics_all_genes.csv")
        
        # Check for metadata-specific plot directory - it should exist if metadata was processed
        metadata_plot_subdir = output_dir / "images" / f"{utils.sanitize_filename('Category')}_per_gene_plots"
        assert metadata_plot_subdir.exists(), "Metadata plot directory should exist"
        
        # Check for log messages indicating metadata processing
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "metadata" in log_content.lower(), "Log should mention metadata processing"

# --- Test Class for `analyze` subcommand ---

    def test_analyze_with_custom_ref_file(self, dummy_gene_dir: Path, tmp_path: Path):
        """Test analyze command with a custom reference file."""
        output_dir = tmp_path / "analyze_output_custom_ref"
        custom_ref_content = "Codon\tFrequency\nAAA\t0.5\nAAG\t0.5\nTTT\t1.0\n"
        custom_ref_file = tmp_path / "custom_ref.tsv"
        custom_ref_file.write_text(custom_ref_content)

        cmd_args = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--ref", str(custom_ref_file),
            "--ref_delimiter", "\\t" # Explicitly set delimiter
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()

        assert (output_dir / "data" / "per_sequence_metrics_all_genes.csv").exists()
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert f"Using user-specified reference file: {custom_ref_file}" in log_content
            assert "Reference data loaded and weights successfully extracted." in log_content

    def test_analyze_with_invalid_ref_file(self, dummy_gene_dir: Path, tmp_path: Path):
        """Test analyze command with a non-existent or malformed reference file."""
        output_dir = tmp_path / "analyze_output_invalid_ref"
        non_existent_ref = tmp_path / "non_existent_ref.tsv"
        malformed_ref = tmp_path / "malformed_ref.tsv"
        malformed_ref.write_text("Codon,Freq\nAAA\n") # Missing value for AAA

        # Test non-existent file
        cmd_args_non_existent = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--ref", str(non_existent_ref)
        ]
        with pytest.raises(SystemExit) as e:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args_non_existent)
                cli.main()
        assert e.value.code != 0
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert f"Specified reference file not found: {non_existent_ref}. Exiting." in log_content

        # Test malformed file
        output_dir_malformed = tmp_path / "analyze_output_malformed_ref"
        cmd_args_malformed = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir_malformed),
            "--ref", str(malformed_ref)
        ]
        with pytest.raises(SystemExit) as e:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args_malformed)
                cli.main()
        assert e.value.code != 0
        log_file_malformed = output_dir_malformed / "pycodon_analyzer.log"
        if log_file_malformed.exists():
            log_content_malformed = log_file_malformed.read_text()
            assert "Fatal: Failed to load or process reference data" in log_content_malformed

    def test_analyze_with_threads_option(self, dummy_gene_dir: Path, tmp_path: Path, caplog):
        """Test analyze command with --threads > 1."""
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "analyze_output_threads"
        cmd_args = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--threads", "2" # Request 2 threads
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Using 2 process(es) for gene file analysis." in log_content or \
                   "Starting parallel analysis with 2 processes..." in log_content
        assert (output_dir / "data" / "per_sequence_metrics_all_genes.csv").exists()

    def test_analyze_with_max_ambiguity_filter(self, tmp_path: Path, caplog):
        """Test analyze command with --max_ambiguity to filter sequences."""
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "analyze_output_ambiguity"
        
        # Create a gene file with one high-ambiguity sequence and one low-ambiguity
        gene_dir = tmp_path / "genes_input_ambiguity"
        gene_dir.mkdir()
        (gene_dir / "gene_Ambiguous.fasta").write_text(
            ">Seq1_HighAmbiguity\nATGCGTTAANNNNNNNNNN\n" # ~50% N
            ">Seq2_LowAmbiguity\nATGCGTTAA\n" # 0% N
        )

        cmd_args = [
            "analyze",
            "--directory", str(gene_dir),
            "--output", str(output_dir),
            "--max_ambiguity", "10.0" # Only allow <= 10% N
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()
        
        # Check that only the low-ambiguity sequence is processed
        metrics_file = output_dir / "data" / "per_sequence_metrics_all_genes.csv"
        assert metrics_file.exists()
        df = pd.read_csv(metrics_file)
        # Expect 2 rows: one for the individual gene, one for the 'complete' concatenated sequence
        assert len(df) == 2
        assert "Seq2_LowAmbiguity" in df['ID'].values[0] # ID is gene__originalID
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Removed 1 out of 2 sequences during cleaning/filtering." in log_content

    def test_analyze_skip_ca(self, dummy_gene_dir: Path, tmp_path: Path, caplog):
        """Test analyze command with --skip-ca flag."""
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "analyze_output_skip_ca"
        cmd_args = [
            "analyze",
            "--directory", str(dummy_gene_dir),
            "--output", str(output_dir),
            "--skip-ca"
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()
        
        # Check that CA-related files are NOT created
        assert not (output_dir / "data" / "ca_row_coordinates.csv").exists()
        assert not (output_dir / "data" / "ca_col_coordinates.csv").exists()
        assert not (output_dir / "data" / "ca_col_contributions.csv").exists()
        assert not (output_dir / "data" / "ca_eigenvalues.csv").exists()
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Skipping combined Correspondence Analysis as requested (--skip_ca)." in log_content

    def test_analyze_empty_input_directory(self, tmp_path: Path, caplog):
        """Test analyze command with an empty input directory."""
        output_dir = tmp_path / "analyze_output_empty_input"
        empty_gene_dir = tmp_path / "empty_genes"
        empty_gene_dir.mkdir()

        cmd_args = [
            "analyze",
            "--directory", str(empty_gene_dir),
            "--output", str(output_dir)
        ]
        with pytest.raises(SystemExit) as e:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
                cli.main()
            assert e.value.code != 0
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert f"No gene alignment files matching 'gene_*' with extensions" in log_content
            assert f"found in directory: {empty_gene_dir}. Exiting." in log_content

    def test_analyze_input_dir_no_gene_prefix(self, tmp_path: Path, caplog):
        """Test analyze command with files not matching 'gene_*.fasta' pattern."""
        output_dir = tmp_path / "analyze_output_no_prefix"
        gene_dir = tmp_path / "genes_no_prefix"
        gene_dir.mkdir()
        (gene_dir / "my_gene.fasta").write_text(">Seq1\nATGCGTTAA\n")
        (gene_dir / "another_file.txt").write_text("some text")

        cmd_args = [
            "analyze",
            "--directory", str(gene_dir),
            "--output", str(output_dir)
        ]
        with pytest.raises(SystemExit) as e:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
                cli.main()
            assert e.value.code != 0
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert f"No gene alignment files matching 'gene_*' with extensions" in log_content
            assert f"No gene alignment files matching 'gene_*' with extensions" in log_content

    def test_analyze_with_metadata_too_many_categories(self, dummy_gene_dir: Path, tmp_path: Path, caplog):
        """Test analyze command with metadata having more categories than metadata_max_categories."""
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "analyze_output_many_meta_cats"
        
        # Create metadata with more categories than default (15)
        meta_content = "seq_id,Category\n"
        for i in range(20):
            meta_content += f"GeneA_Seq{i+1},Cat{i%10}\n" # 10 categories
        meta_file = tmp_path / "metadata_many_cats.csv"
        meta_file.write_text(meta_content)

        # Create dummy gene files matching the metadata IDs
        gene_dir = tmp_path / "genes_many_meta_cats"
        gene_dir.mkdir()
        for i in range(20):
            (gene_dir / f"gene_GeneA_Seq{i+1}.fasta").write_text(f">GeneA_Seq{i+1}\nATGCGTTAA\n")

        cmd_args = [
            "analyze",
            "--directory", str(gene_dir),
            "--output", str(output_dir),
            "--metadata", str(meta_file),
            "--metadata_id_col", "seq_id",
            "--color_by_metadata", "Category",
            "--metadata_max_categories", "5" # Limit to 5 categories
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main()
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Limiting to top 5 for plotting, others grouped as 'Other'." in log_content
        
        # Check that metadata-specific plot directory exists
        metadata_plot_subdir = output_dir / "images" / f"{utils.sanitize_filename('Category')}_per_gene_plots"
        assert metadata_plot_subdir.is_dir()
        # Check for plots within the gene-specific subdirectory
        gene_specific_meta_plots_dir = metadata_plot_subdir / utils.sanitize_filename('GeneA_Seq1') # Example gene
        assert gene_specific_meta_plots_dir.is_dir()
        plot_files = list(gene_specific_meta_plots_dir.glob("*.svg"))
        assert len(plot_files) > 0 # At least one plot should be generated

# --- Test Class for `extract` subcommand ---
class TestExtractCommand:

    def test_extract_basic_run(self, dummy_annotation_file_for_extract: Path,
                               dummy_alignment_file_for_extract: Path, tmp_path: Path, caplog):
        """Test the extract command with our test fixtures."""
        caplog.set_level(logging.INFO)
        output_dir = tmp_path / "extract_output_basic"
        log_file_name = "extract_run.log"

        cmd_args = [
            "--log-file", log_file_name,
            "extract",
            "--annotations", str(dummy_annotation_file_for_extract),
            "--alignment", str(dummy_alignment_file_for_extract),
            "--ref_id", "RefAnnotationSeq", # Matches fixture
            "--output_dir", str(output_dir)
        ]
        
        # Now, this should succeed
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            cli.main() # Expect success, no SystemExit

        assert output_dir.is_dir()
        assert (output_dir / log_file_name).exists()
        
        # Check for expected output files (extracted gene fasta files)
        assert (output_dir / "gene_GeneX.fasta").exists()
        assert (output_dir / "gene_GeneY.fasta").exists()
        
        # Check content of extracted files (simple check for non-empty)
        assert (output_dir / "gene_GeneX.fasta").read_text().strip() != ""
        assert (output_dir / "gene_GeneY.fasta").read_text().strip() != ""

        # Check the log file for success message
        log_file = output_dir / log_file_name
        if log_file.exists():
            log_content = log_file.read_text()
            assert "'extract' command finished successfully." in log_content


    def test_extract_invalid_annotation_file(self, dummy_alignment_file_for_extract: Path, tmp_path: Path, caplog):
        output_dir = tmp_path / "extract_output_invalid_anno"
        non_existent_anno = tmp_path / "non_existent_anno.gb"
        cmd_args = [
            "extract",
            "--annotations", str(non_existent_anno),
            "--alignment", str(dummy_alignment_file_for_extract),
            "--ref_id", "RefAnnotationSeq",
            "--output_dir", str(output_dir)
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            with pytest.raises(SystemExit) as e:
                cli.main()
            assert e.value.code != 0
        
        # Check the log file if it was created
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Annotation file not found" in log_content

    def test_extract_invalid_ref_id(self, dummy_annotation_file_for_extract: Path,
                                  dummy_alignment_file_for_extract: Path, tmp_path: Path, caplog):
        output_dir = tmp_path / "extract_output_invalid_ref_id"
        cmd_args = [
            "extract",
            "--annotations", str(dummy_annotation_file_for_extract),
            "--alignment", str(dummy_alignment_file_for_extract),
            "--ref_id", "NonExistentRef", # Invalid ref_id
            "--output_dir", str(output_dir)
        ]
        # This should ideally be caught by extraction.py and logged,
        # cli.py might complete if extraction.py returns None gracefully for all genes.
        # The `handle_extract_command` should log if no genes are extracted.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
            with pytest.raises(SystemExit) as e:
                cli.main()
            assert e.value.code != 0

        # Check that the output directory exists but no gene files were created
        assert output_dir.exists()
        assert not list(output_dir.glob("*.fasta")) # No fasta files should be created
        
        # We'll check the log file directly since caplog might not capture logs from all handlers
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Reference sequence ID 'NonExistentRef' not found in alignment file." in log_content


    def test_extract_invalid_alignment_file(self, dummy_annotation_file_for_extract: Path, tmp_path: Path, caplog):
        """Test extract command with a non-existent alignment file."""
        output_dir = tmp_path / "extract_output_invalid_align"
        non_existent_align = tmp_path / "non_existent_align.fasta"
        cmd_args = [
            "extract",
            "--annotations", str(dummy_annotation_file_for_extract),
            "--alignment", str(non_existent_align),
            "--ref_id", "RefAnnotationSeq",
            "--output_dir", str(output_dir)
        ]
        with pytest.raises(SystemExit) as e:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
                cli.main()
            assert e.value.code != 0
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "Alignment file not found" in log_content

    def test_extract_no_genes_in_annotation(self, dummy_alignment_file_for_extract: Path, tmp_path: Path, caplog):
        """Test extract command when annotation file contains no parsable genes."""
        output_dir = tmp_path / "extract_output_no_genes_anno"
        # Create an annotation file with no gene/location tags
        empty_anno_content = """>Seq1_NoGeneInfo\nATGCGTTAA\n"""
        empty_anno_file = tmp_path / "empty_annotations.fasta"
        empty_anno_file.write_text(empty_anno_content)

        cmd_args = [
            "extract",
            "--annotations", str(empty_anno_file),
            "--alignment", str(dummy_alignment_file_for_extract),
            "--ref_id", "RefAnnotationSeq",
            "--output_dir", str(output_dir)
        ]
        with pytest.raises(SystemExit) as e:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args)
                cli.main()
            assert e.value.code != 0
        
        log_file = output_dir / "pycodon_analyzer.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "No valid annotations parsed from the annotation file." in log_content
            assert "Cannot proceed with extraction." in log_content

# --- Tests for general CLI behavior ---
class TestGeneralCLI:
    def test_help_message(self, caplog):
        # Test main help
        process_main = run_cli_command(["--help"], expect_success=True)
        assert "PyCodon Analyzer" in process_main.stdout
        assert "analyze" in process_main.stdout
        assert "extract" in process_main.stdout

        # Test analyze help
        process_analyze = run_cli_command(["analyze", "--help"], expect_success=True)
        assert "directory" in process_analyze.stdout
        assert "output" in process_analyze.stdout
        assert "no-html-report" in process_analyze.stdout

        # Test extract help
        process_extract = run_cli_command(["extract", "--help"], expect_success=True)
        assert "annotations" in process_extract.stdout
        assert "alignment" in process_extract.stdout

    def test_version_command(self):
        # Import version directly to avoid subprocess issues
        from pycodon_analyzer import __version__ as pkg_version
        
        # Use our improved run_cli_command that directly calls cli.main
        process = run_cli_command(["--version"], expect_success=True)
        assert f"pycodon_analyzer {pkg_version}" in process.stdout

    def test_invalid_subcommand(self, caplog):
        # Using direct cli.main and checking stderr for argparse errors
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer", "invalidcommand"])
            # Argparse error messages go to stderr and cause SystemExit
            with pytest.raises(SystemExit) as e:
                # Capture stderr
                from io import StringIO
                stderr_catcher = StringIO()
                mp.setattr(sys, "stderr", stderr_catcher)
                cli.main()
            assert e.value.code != 0
            assert "invalid choice: 'invalidcommand'" in stderr_catcher.getvalue()


    def test_verbose_logging_console(self, dummy_gene_dir: Path, tmp_path: Path):
        """Test if -v increases console log verbosity to DEBUG."""
        output_dir_no_verbose = tmp_path / "analyze_output_no_verbose"
        output_dir_verbose = tmp_path / "analyze_output_verbose"
        
        # Run without verbose
        cmd_args_no_verbose = [
            "analyze", "--directory", str(dummy_gene_dir), "--output", str(output_dir_no_verbose),
            "--skip-plots", "--no-html-report"
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args_no_verbose)
            cli.main()
        
        # Run with verbose
        cmd_args_verbose = [
            "-v", "analyze", "--directory", str(dummy_gene_dir), "--output", str(output_dir_verbose),
            "--skip-plots", "--no-html-report"
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["pycodon_analyzer"] + cmd_args_verbose)
            cli.main()
        
        # Check log files
        log_file_no_verbose = output_dir_no_verbose / "pycodon_analyzer.log"
        log_file_verbose = output_dir_verbose / "pycodon_analyzer.log"
        
        assert log_file_no_verbose.exists()
        assert log_file_verbose.exists()
        
        log_content_no_verbose = log_file_no_verbose.read_text()
        log_content_verbose = log_file_verbose.read_text()
        
        # Count DEBUG messages in each log
        debug_count_no_verbose = log_content_no_verbose.count("DEBUG")
        debug_count_verbose = log_content_verbose.count("DEBUG")
        
        # Verbose mode should have DEBUG messages
        assert debug_count_verbose > 0, "Verbose mode should have DEBUG messages"
        
        # Check for specific debug message that should only appear in verbose mode
        assert "Full command-line arguments:" in log_content_verbose, "Verbose log should contain debug message with full arguments"