# PyCodon Analyzer (pycodon_analyzer)

A Python tool for comprehensive codon usage and sequence property analysis. It processes multiple gene alignments provided as separate FASTA files within an input directory.

## Overview

`pycodon_analyzer` reads multiple FASTA files (each containing sequences for a single gene aligned across different samples/genomes) from a specified directory. It performs a series of cleaning steps on the sequences and then calculates a wide range of codon usage indices and sequence properties for each gene individually, as well as for concatenated sequences representing the "complete" coding sequence per original genome ID.

The tool aggregates results across all genes, performs Correspondence Analysis (CA) on the combined RSCU data, computes basic statistics comparing genes, and generates various publication-ready plots to visualize codon usage patterns, sequence properties, and relationships between calculated metrics.

## Features

* **Input:** Reads gene alignments from multiple FASTA files within a directory (e.g., `gene_GENENAME.fasta`).
* **Sequence Cleaning:** Performs the following steps automatically before analysis:
    * Removes gap characters (`-`).
    * Validates sequence length (must be multiple of 3 after gap removal).
    * Conditionally removes standard START (`ATG`) and STOP (`TAA`, `TAG`, `TGA`) codons if present at the ends.
    * Replaces IUPAC ambiguous DNA characters with 'N'.
    * Filters out sequences exceeding a defined ambiguity threshold (default: 15% 'N').
* **Calculated Metrics:** Computes the following for each gene and for concatenated "complete" sequences:
    * Codon Counts & Frequencies
    * RSCU (Relative Synonymous Codon Usage)
    * GC Content: Overall GC%, GC1, GC2, GC3, GC12
    * ENC (Effective Number of Codons)
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
    * `per_sequence_metrics_all_genes.csv`: Detailed metrics for every valid sequence from every processed gene (including 'complete'). Row ID format: `SequenceID_GeneName`.
    * `mean_features_per_gene.csv`: Mean values for key metrics, calculated per gene (including 'complete').
    * `per_sequence_rscu_wide.csv`: RSCU value for every codon for every sequence (wide format: sequences x codons). Index format: `GeneName__SequenceID`.
    * `gene_comparison_stats.csv`: Results of statistical tests comparing features between genes.
    * `ca_row_coordinates.csv`, `ca_col_coordinates.csv`, `ca_col_contributions.csv`, `ca_eigenvalues.csv`: Detailed results from the Correspondence Analysis.
* **Output Plots:** Generates various plots (default: PNG):
    * `RSCU_boxplot_GENENAME.(fmt)`: RSCU distribution per codon for each gene (and 'complete'), grouped by AA, with highlighted codon labels.
    * `gc_means_barplot_by_Gene.(fmt)`: Mean GC values (GC, GC1-3, GC12) grouped by gene.
    * `neutrality_plot_grouped_by_Gene.(fmt)`: GC12 vs GC3, colored by gene.
    * `enc_vs_gc3_plot_grouped_by_Gene.(fmt)`: ENC vs GC3, colored by gene, with Wright's curve.
    * `relative_dinucleotide_abundance.(fmt)`: O/E ratio for dinucleotides, lines colored by gene.
    * `ca_biplot_compXvY_combined_by_gene.(fmt)`: CA biplot (sequences/codons), sequence points colored by gene.
    * `ca_variance_explained_topN.(fmt)`: Variance explained by top CA dimensions.
    * `ca_contribution_dimX_topN.(fmt)`: Top codons contributing to CA dimensions 1 & 2.
    * `feature_correlation_heatmap_METHOD.(fmt)`: Heatmap showing correlations between calculated metrics.
* **Performance:** Uses multiprocessing for parallel analysis within each gene file.

## Prerequisite: Input Directory

This tool requires an input directory containing gene alignments as separate FASTA files.
* Each file should contain sequences for the **same gene** aligned across different samples/genomes.
* Files must be named following the pattern `gene_GENENAME.fasta` (or `.fa`, `.fna`, `.fas`, `.faa`), where `GENENAME` is the identifier for the gene (e.g., `gene_S.fasta`, `gene_N.fasta`).
* Sequence IDs within each file should correspond to the original sample/genome ID (e.g., `EPI_ISL_XXXXXX`). These IDs should ideally be consistent across different gene files for the same original sample/genome to enable the "complete" sequence analysis.

You can generate these files using the separate `extract_gene_alignments.py` script (provided previously or available [here](https://github.com/GabrielFalque/fasta_tools)) or similar tools.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd pycodon_analyzer
    ```
2.  **Install dependencies:**
    *(Recommended: Use a virtual environment)*
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    # Install the tool and its dependencies
    pip install .
    # For development (includes pytest):
    # pip install -e .[dev]
    ```

## Usage

Run the analysis directly from the command line:

```bash
pycodon_analyzer --directory <path/to/gene_alignments_dir/> --output <results_dir> [OPTIONS]
```

**Example:**

```bash
# Analyze all gene alignments found in 'extracted_genes/' directory
# using 4 threads and the default human codon usage reference.
pycodon_analyzer -d extracted_genes/ -o combined_analysis --ref human -t 4

# Analyze using 8 threads, disabling reference-based calculations (CAI/Fop/RCDI)
# and skipping plots. Set max ambiguity to 10%.
pycodon_analyzer -d extracted_genes/ -o analysis_no_ref --ref none -t 8 --skip_plots --max_ambiguity 10.0
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
* `-t, --threads INT`: Number of threads for parallel processing (Default: 1, 0=all cores).
* `--max_ambiguity FLOAT`: Max allowed 'N' percentage per sequence (Default: 15.0).
* `--plot_formats FMT [FMT ...]`: Output format(s) for plots (Default: png).
* `--skip_plots`: Flag to disable all plot generation.
* `--skip_ca`: Flag to disable Correspondence Analysis calculation and plotting.
* `--ca_dims X Y`: Components (0-indexed) for CA plot (Default: 0 1).

## Reference File Format (`--ref`)

Required for CAI, Fop, RCDI. Should be CSV or TSV with columns for 'Codon' and one of 'Frequency', 'Count', 'RSCU', or 'Frequency (per thousand)'. The tool prioritizes finding an 'RSCU' column if present. Using a reference set based on *highly expressed genes* of the target organism is recommended for meaningful CAI/RCDI interpretation regarding translational adaptation.

## Development

* **Running Tests:**
    * Install development dependencies: `pip install -e .[dev]` (includes `pytest`).
    * Navigate to the project root directory.
    * Run: `pytest`
* **Building:** Install `build` (`pip install build`) and run `python -m build`.

## TODO / Future Improvements

* Implement tAI calculation (requires tRNA data input).
* Add more statistical comparison options (e.g., between sequence groups if metadata is provided).
* Add option to save aggregate results per gene.
* Generate combined summary statistics table.
* Support GFF3/GTF input for the separate extraction script.
* Option for interactive plots (Plotly/Bokeh).
* Formal logging instead of print statements.
* Publish package to PyPI.