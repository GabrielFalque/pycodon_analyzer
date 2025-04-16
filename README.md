# PyCodon Analyzer (pycodon_analyzer)

A Python tool for comprehensive codon usage and sequence property analysis from multiple gene alignments.

[![PyPI version](https://badge.fury.io/py/pycodon-analyzer.svg)](https://badge.fury.io/py/pycodon-analyzer) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Overview

`pycodon_analyzer` reads multiple FASTA files (each containing sequences for a single gene aligned across different samples/genomes) from a specified directory. It performs a series of cleaning steps on the sequences and then calculates a wide range of codon usage indices and sequence properties for each gene individually, as well as for concatenated sequences representing the "complete" coding sequence per original genome ID.

The tool aggregates results across all genes, performs Correspondence Analysis (CA) on the combined RSCU data, computes basic statistics comparing genes, and generates various publication-ready plots to visualize codon usage patterns, sequence properties, and relationships between calculated metrics. It leverages multiprocessing for faster analysis of multiple gene files and uses logging for informative feedback.

## Features

* **Input:** Reads gene alignments from multiple FASTA files within a directory (requires `gene_GENENAME.fasta` naming convention).
* **Sequence Cleaning:** Performs robust cleaning before analysis:
    * Removes gap characters (`-`).
    * Validates sequence length (multiple of 3 after gap removal).
    * Conditionally removes standard START (`ATG`) and STOP (`TAA`, `TAG`, `TGA`) codons.
    * Replaces IUPAC ambiguous DNA characters with 'N'.
    * Filters sequences exceeding a defined ambiguity threshold (default: 15% 'N').
* **Calculated Metrics:** Computes the following for each gene and for concatenated "complete" sequences:
    * Codon Counts & Frequencies
    * RSCU (Relative Synonymous Codon Usage)
    * GC Content: Overall GC%, GC1, GC2, GC3, GC12
    * ENC (Effective Number of Codons) - with configurable codon count threshold.
    * CAI (Codon Adaptation Index) - Requires reference file (`--ref`).
    * RCDI (Relative Codon Deoptimization Index) - Requires reference file (`--ref`).
    * Fop (Frequency of Optimal Codons) - Requires reference file (`--ref`).
    * Protein Properties: GRAVY (Grand Average of Hydropathicity) & Aromaticity %.
    * Nucleotide & Dinucleotide Frequencies.
* **Statistical Analysis:**
    * Performs Kruskal-Wallis H-test (default) or ANOVA to compare key metrics between different genes.
* **Multivariate Analysis:**
    * Performs Correspondence Analysis (CA) on combined RSCU data from all genes.
* **Output Tables (CSV):**
    * `per_sequence_metrics_all_genes.csv`: Detailed metrics for every valid sequence from every processed gene/complete set.
    * `mean_features_per_gene.csv`: Mean values for key metrics per gene/complete set.
    * `per_sequence_rscu_wide.csv`: RSCU value for every codon for every sequence (wide format).
    * `gene_comparison_stats.csv`: Results of statistical tests comparing features between genes.
    * `ca_row_coordinates.csv`, `ca_col_coordinates.csv`, `ca_col_contributions.csv`, `ca_eigenvalues.csv`: Detailed results from the combined Correspondence Analysis (if run).
* **Output Plots:** Generates various plots (default: PNG, other formats available):
    * `RSCU_boxplot_GENENAME.(fmt)`: RSCU distribution per codon for each gene/complete set, grouped by AA, with highlighted codon labels.
    * `gc_means_barplot_by_Gene.(fmt)`: Mean GC values grouped by gene.
    * `neutrality_plot_grouped_by_Gene.(fmt)`: GC12 vs GC3, colored by gene, with **adjusted labels** to reduce overlap.
    * `enc_vs_gc3_plot_grouped_by_Gene.(fmt)`: ENC vs GC3, colored by gene, with Wright's curve and **adjusted labels**.
    * `relative_dinucleotide_abundance.(fmt)`: O/E ratio for dinucleotides, lines colored by gene (consistent colors).
    * `ca_biplot_compXvY_combined_by_gene.(fmt)`: Combined CA biplot, sequence points colored by gene (consistent colors) with **adjusted labels**.
    * `ca_variance_explained_topN.(fmt)`: Variance explained by top CA dimensions.
    * `ca_contribution_dimX_topN.(fmt)`: Top codons contributing to CA dimensions 1 & 2.
    * `feature_correlation_heatmap_METHOD.(fmt)`: Heatmap showing correlations between calculated metrics.
    * **New:** `ca_axes_feature_corr_METHOD.(fmt)`: Heatmap showing correlations between CA axes (Dim1, Dim2) and other features (metrics & RSCU per codon), highlighting significant correlations.
* **Performance:** Uses multiprocessing to process **gene files in parallel**.
* **Logging:** Provides informative console output using Python's standard `logging` module (use `-v` for DEBUG level).
* **Code Quality:** Includes type hints and refined error handling.

## Prerequisites

1.  **Python:** Python 3.8 or higher recommended.
2.  **Input Directory:** Requires an input directory containing gene alignments as separate FASTA files.
    * Each file must contain sequences for the **same gene** aligned across different samples/genomes.
    * Files must be named following the pattern `gene_GENENAME.fasta` (or `.fa`, `.fna`, `.fas`, `.faa`), where `GENENAME` is the identifier for the gene (e.g., `gene_S.fasta`, `gene_N.fasta`).
    * Sequence IDs within each file should correspond to the original sample/genome ID (e.g., `EPI_ISL_XXXXXX`). IDs should ideally be consistent across different gene files for the same sample/genome to enable the "complete" sequence analysis.
    * *(Recommendation)* Use a script like the one previously discussed (`extract_genes_aln.py` or similar GFF/GenBank parsers) to prepare files in this format from your original genome alignments or annotation files.

## Dependencies

The tool requires the following Python libraries:

* biopython (>=1.79)
* pandas (>=1.3.0)
* matplotlib (>=3.4.0)
* seaborn (>=0.11.0)
* numpy (>=1.21.0)
* scipy (>=1.6.0)
* prince (>=0.12.1)
* adjustText (>=0.8)
* importlib-resources (>=1.0) ; *only for Python < 3.9*

These will be installed automatically when using `pip`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd pycodon_analyzer
    ```
2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows (cmd/powershell)
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
3.  **Install the tool and its dependencies:**
    ```bash
    pip install .
    ```
4.  **(Optional) Install development dependencies** (for running tests, linting, building):
    ```bash
    pip install -e .[dev]
    ```
    *(Note: The `-e` installs in editable mode)*

## Usage

Run the analysis directly from the command line (ensure your virtual environment is active if you used one):

```bash
pycodon_analyzer --directory <path/to/gene_alignments_dir/> --output <results_dir> [OPTIONS]
```

**Example:**

```bash
# Analyze genes in 'extracted_genes/', use 4 processes, human reference, verbose logging
pycodon_analyzer -d extracted_genes/ -o combined_analysis --ref human -t 4 -v

# Analyze using 8 processes (if available), disable reference, skip plots and CA
pycodon_analyzer -d extracted_genes/ -o analysis_no_ref --ref none -t 8 --skip_plots --skip_ca
```

**Command-Line Options:**

```bash
pycodon_analyzer --help
```
Displays all available command-line arguments. Key options include:

* `-d, --directory DIR`: Path to input directory with `gene_*.fasta` files (Required).
* `-o, --output DIR`: Path to output directory (Default: `codon_analysis_results`).
* `--genetic_code INT`: NCBI genetic code ID (Default: 1).
* `--ref FILE | human | none`: Path to codon usage reference table for CAI/Fop/RCDI. Use 'human' for bundled default, 'none' to disable. (Default: 'human').
* `-t, --threads INT`: Number of processes for parallel **gene file** analysis (Default: 1, 0=all cores).
* `--max_ambiguity FLOAT`: Max allowed 'N' percentage per sequence (Default: 15.0).
* `--plot_formats FMT [FMT ...]`: Output format(s) for plots (Default: png).
* `--skip_plots`: Flag to disable all plot generation.
* `--skip_ca`: Flag to disable combined Correspondence Analysis calculation and plotting.
* `--ca_dims X Y`: Components (0-indexed) for combined CA plot (Default: 0 1).
* `-v, --verbose`: Increase output verbosity (sets logging to DEBUG).

## Reference File Format (`--ref`)

Required for CAI, Fop, RCDI calculations. Should be CSV or TSV with columns for 'Codon' and one of 'Frequency', 'Count', 'RSCU', 'Freq', or 'Frequency (per thousand)'. The tool prioritizes finding an 'RSCU' column if present. For meaningful CAI/RCDI interpretation regarding translational adaptation, using a reference set based on *highly expressed genes* of the target organism is recommended.

## Development

* **Running Tests:**
    * Install development dependencies: `pip install -e .[dev]`
    * Navigate to the project root directory.
    * Run: `pytest`
* **Type Checking:**
    * Install `mypy` (included in `dev` dependencies).
    * Run: `mypy src`
* **Linting/Formatting (Example using Ruff):**
    * Install `ruff` (included in `dev` dependencies).
    * Check: `ruff check src tests`
    * Format: `ruff format src tests`
* **Building:**
    * Install `build` (`pip install build`).
    * Run: `python -m build`

## TODO / Future Improvements

* Implement tAI calculation (requires tRNA data input).
* Integrate input data preparation (from GFF/GenBank).
* Add more statistical comparison options (e.g., pairwise tests, grouping by metadata).
* Support providing sequence metadata for advanced analysis.
* Generate combined HTML report.
* Add progress bars (e.g., `tqdm`).
* Implement CI/CD pipeline (e.g., GitHub Actions).
* Consider interactive plots (Plotly/Bokeh).
* Publish package to PyPI.