# PyCodon Analyzer (pycodon_analyzer)

A Python tool for comprehensive codon usage analysis and gene alignment extraction.

[![PyPI version](https://badge.fury.io/py/pycodon-analyzer.svg)](https://badge.fury.io/py/pycodon-analyzer) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`pycodon_analyzer` is a command-line tool with two main functionalities:

1.  **`extract`**: Extracts individual gene alignments from a whole genome multiple sequence alignment (MSA) based on a reference annotation file.
2.  **`analyze`**: Performs codon usage analysis on a directory of pre-extracted and aligned gene FASTA files. This can be enhanced by providing an optional metadata file to generate more contextual and stratified analyses.

The `analyze` command calculates a wide range of codon usage indices and sequence properties for each gene, as well as for concatenated sequences ("complete" coding sequence per original genome ID). It aggregates results, performs Correspondence Analysis (CA), computes statistics, and generates various plots. When metadata is provided, it can generate additional sets of plots colored by specified metadata categories.

The tool leverages multiprocessing for faster analysis and uses Python's standard `logging` module (enhanced with `rich` for better console output and progress bars) for informative feedback.

## Features

### Common Features
* **Logging:** Provides informative console output using Python's standard `logging` module, enhanced with `rich` for better readability and progress bars (use `-v` for DEBUG level).
* **Code Quality:** Includes type hints, structured error handling, and refactored modules for better maintainability.
* **Robust Input Handling:** Improved parsing for reference files (delimiter detection) and more resilient processing pipelines.

### `extract` Subcommand
* **Input:**
    * Whole Genome Multiple Sequence Alignment (FASTA format).
    * Reference Annotation File: Currently supports a multi-FASTA format where sequence headers contain GenBank-style feature tags like `[gene=GENE_NAME]` or `[locus_tag=LOCUS_TAG]` and `[location=START..END]`.
    * ID of the reference sequence within the alignment.
* **Processing:**
    * Parses gene coordinates (start, end, strand) from the annotation file.
    * Maps these ungapped coordinates to the gapped positions in the aligned reference sequence.
    * Extracts the alignment columns corresponding to each gene for all sequences in the MSA.
    * Handles reverse-complementation for genes on the negative strand.
* **Output:**
    * Writes a separate FASTA alignment file for each successfully extracted gene, in the format `gene_GENENAME.fasta`. These files are suitable for direct input into the `analyze` subcommand.

### `analyze` Subcommand
* **Input:**
    * Reads pre-extracted gene alignments from multiple FASTA files within a directory (requires `gene_GENENAME.fasta` naming convention).
    * **Optional Metadata:** Accepts a CSV/TSV file (`--metadata`) to associate sequences with external categorical or numerical data. Sequence identifiers in the metadata (`--metadata_id_col`) must match original FASTA sequence IDs.
* **Sequence Cleaning:** Performs robust cleaning before analysis:
    * Removes gap characters (`-`).
    * Validates sequence length (multiple of 3 after gap removal).
    * Conditionally removes standard START (`ATG`) and STOP (`TAA`, `TAG`, `TGA`) codons.
    * Replaces IUPAC ambiguous DNA characters with 'N'.
    * Filters sequences exceeding a defined ambiguity threshold (default: 15% 'N', configurable).
* **Calculated Metrics:** Computes the following for each gene and for concatenated "complete" sequences:
    * Codon Counts & Frequencies (aggregate and per-sequence).
    * RSCU (Relative Synonymous Codon Usage) (aggregate and per-sequence for CA input).
    * GC Content: Overall GC%, GC1, GC2, GC3, GC12.
    * ENC (Effective Number of Codons).
    * CAI (Codon Adaptation Index) - Requires reference file (`--ref`).
    * RCDI (Relative Codon Deoptimization Index) - Requires reference file (`--ref`).
    * Fop (Frequency of Optimal Codons) - Requires reference file (`--ref`).
    * Protein Properties: GRAVY (Grand Average of Hydropathicity) & Aromaticity %.
    * Nucleotide & Dinucleotide Frequencies (aggregate per gene/complete set, and **per-sequence** for metadata-driven dinucleotide plots).
    * Relative Dinucleotide Abundance (O/E ratios, aggregate per gene/complete set, and **per-sequence** for metadata-driven plots).
* **Metadata Integration:**
    * Merges provided metadata with per-sequence analysis results.
    * Allows for generation of plots colored by a specified metadata column (`--color_by_metadata`).
    * Limits the number of categories shown in metadata-colored plots (`--metadata_max_categories`, default 15, others grouped as "Other").
* **Statistical Analysis:**
    * Performs Kruskal-Wallis H-test (default) or ANOVA to compare key metrics between different genes.
* **Multivariate Analysis:**
    * Performs Correspondence Analysis (CA) on combined RSCU data from all genes.
    * Performs Correspondence Analysis (CA) on RSCU data *for each gene individually* if metadata-based coloring is requested.
    * Generates a correlation heatmap between CA axes (Dim1, Dim2 of combined CA) and other features.
* **Output Tables (CSV):**
    * `per_sequence_metrics_all_genes.csv`: Detailed metrics for every valid sequence, **including merged metadata if provided**.
    * `mean_features_per_gene.csv`: Mean values for key metrics per gene.
    * `per_sequence_rscu_wide.csv`: RSCU value for every codon for every sequence (input for combined CA).
    * `gene_comparison_stats.csv`: Results of statistical tests between genes.
    * `ca_row_coordinates.csv`, `ca_col_coordinates.csv`, `ca_col_contributions.csv`, `ca_eigenvalues.csv`: Detailed results from the **combined CA**.
    * `ca_axes_vs_metadata_correlation.csv`: (If implemented) Correlations between CA axes and numerical metadata.
* **Output Plots (Default and Metadata-Driven):**
    * **Standard Combined Plots (in main output directory):**
        * `RSCU_boxplot_GENENAME.(fmt)`: RSCU distribution per codon for each gene/complete set.
        * `gc_means_barplot_by_Gene.(fmt)`: Mean GC values grouped by gene.
        * `neutrality_plot_grouped_by_Gene.(fmt)`: GC12 vs GC3, colored by gene.
        * `enc_vs_gc3_plot_grouped_by_Gene.(fmt)`: ENC vs GC3, colored by gene, with Wright's curve.
        * `relative_dinucleotide_abundance.(fmt)`: Aggregate O/E ratio for dinucleotides, lines colored by gene.
        * `ca_biplot_compXvY_combined_by_gene.(fmt)`: Combined CA biplot, points colored by gene.
        * CA diagnostics: `ca_variance_explained_topN.(fmt)`, `ca_contribution_dimX_topN.(fmt)`.
        * `feature_correlation_heatmap_METHOD.(fmt)`: Correlation between calculated metrics.
        * `ca_axes_feature_corr_METHOD.(fmt)`: Correlation between combined CA axes and other features.
    * **Per-Gene Plots Colored by Metadata (if `--color_by_metadata <COL>` is used):**
        * Saved in: `output_dir/<METADATA_COL>_per_gene_plots/<GENE_NAME>/`
        * For each gene (and "complete" set):
            * `enc_vs_gc3_plot_<GENE>_by_<META_COL>.(fmt)`: Sequences colored by metadata categories.
            * `neutrality_plot_<GENE>_by_<META_COL>.(fmt)`: Sequences colored by metadata categories.
            * `ca_biplot_compXvY_<GENE>_by_<META_COL>.(fmt)`: CA on sequences of this gene only, points colored by metadata.
            * `dinucl_abundance_<GENE>_by_<META_COL>.(fmt)`: Mean per-sequence dinucleotide O/E ratios, lines colored by metadata categories.
* **Performance:** Uses multiprocessing to process gene files in parallel.

## Prerequisites

* Python 3.8 or higher.
* Git (for cloning).

## Dependencies

The tool requires the following Python libraries:

* `biopython >= 1.79`
* `pandas >= 1.3.0`
* `matplotlib >= 3.4.0`
* `seaborn >= 0.11.0`
* `numpy >= 1.21.0`
* `scipy >= 1.6.0`
* `prince >= 0.12.1`
* `adjustText >= 0.8` (Recommended for better label placement in plots)
* `rich >= 10.0` (For enhanced console logging and progress bars)
* `importlib-resources >= 1.0 ; python_version<"3.9"`

These will be installed automatically when using `pip install .`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GabrielFalque/pycodon_analyzer.git
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
    pip install -e ".[dev]"
    ```
    *(Note: The `-e` installs in editable mode, useful for development. Quotes around `.[dev]` can be necessary in some shells like zsh.)*

## Usage

`pycodon_analyzer` now operates using subcommands: `extract` and `analyze`.

```bash
pycodon_analyzer <subcommand> --help
```

This will show help for the specific subcommand.

### 1\. `extract` Subcommand
(This section remains largely the same as your provided version)
...

### 2\. `analyze` Subcommand

Use this command to perform codon usage analysis on a directory of pre-extracted gene alignment files.

**Synopsis:**

```bash
pycodon_analyzer analyze --directory <PATH_TO_GENE_FASTA_DIR> \
                         --output <RESULTS_OUTPUT_DIR> \
                         [OPTIONS]
```

**Key Arguments for `analyze`:**

  * `-d, --directory DIR`: Path to the input directory containing `gene_GENENAME.fasta` files (Required).
  * `-o, --output DIR`: Path to the output directory for analysis results (Default: `codon_analysis_results`).
  * `--ref FILE | human | none`: Path to codon usage reference table. (Default: 'human' using bundled file).
  * `--ref_delimiter DELIM`: Delimiter for the reference file (e.g., ',', '\\t'). Auto-detects if not provided.
  * `-t, --threads INT`: Number of processes for parallel gene file analysis (Default: 1, 0 or negative for all cores).
  * `--max_ambiguity FLOAT`: Max allowed 'N' percentage per sequence (0-100, Default: 15.0).
  * `--metadata FILE`: Optional path to a metadata file (CSV or TSV).
  * `--metadata_id_col NAME`: Column name in metadata for sequence IDs (Default: "seq_id").
  * `--metadata_delimiter DELIM`: Optional delimiter for metadata file. Auto-detects if not provided.
  * `--color_by_metadata NAME`: Metadata column to use for coloring per-gene plots.
  * `--metadata_max_categories INT`: Max metadata categories for plotting (Default: 15).
  * `--plot_formats FMT [FMT ...]`: Output plot format(s) (Default: png; choices: png, svg, pdf, jpg).
  * `--skip_plots`: Flag to disable all plot generation.
  * `--skip_ca`: Flag to disable combined Correspondence Analysis.
  * `--ca_dims X Y`: Indices for CA components in combined plot (Default: 0 1).
  * `-v, --verbose`: Increase output verbosity (DEBUG level logging).
  * *(Run `pycodon_analyzer analyze --help` for all options.)*

**Example for `analyze`:**

```bash
# Basic analysis using 4 cores and human reference
pycodon_analyzer analyze \
    -d extracted_gene_alignments/ \
    -o codon_analysis_output \
    --ref human \
    -t 4 \
    -v

# Analysis with metadata, generating per-gene plots colored by 'Clade'
pycodon_analyzer analyze \
    -d viral_genes/ \
    -o viral_analysis_with_clades \
    --metadata virus_metadata.csv \
    --metadata_id_col Sequence_Accession \
    --color_by_metadata Clade \
    -t 0 \
    -v
```

## Workflow Example

1.  **Prepare Annotations:** Ensure your reference annotation file is in the expected multi-FASTA format with `[gene=...]` and `[location=...]` tags, or adapt the `extraction.py` module to parse your format (e.g., GenBank, GFF3).
2.  **Extract Gene Alignments:**
    ```bash
    pycodon_analyzer extract -a my_ref_annotations.gb.fasta -g my_genome_msa.fasta -r ref_genome_id -o ./gene_alignments
    ```
3.  **Run Codon Analysis:**
    ```bash
    pycodon_analyzer analyze -d ./gene_alignments -o ./codon_analysis_results --ref human -t 0 -v
    ```

## Reference File Format (`--ref` for `analyze`)

Required for CAI, Fop, RCDI calculations. Should be CSV or TSV with columns for 'Codon' and one of 'Frequency', 'Count', 'RSCU', 'Freq', or 'Frequency (per thousand)'. The tool prioritizes finding an 'RSCU' column if present. For meaningful CAI/RCDI interpretation, using a reference set based on *highly expressed genes* of the target organism is recommended.

## Development

  * **Running Tests:**
    ```bash
    pip install -e .[dev]  # Ensure dev dependencies like pytest are installed
    pytest
    ```
  * **Type Checking:**
    ```bash
    mypy src
    ```
  * **Linting/Formatting (Example using Ruff):**
    ```bash
    ruff check src tests
    ruff format src tests
    ```
  * **Building:**
    ```bash
    python -m build
    ```

## TODO / Future Improvements

  * **`extract` subcommand:**
      * Add support for standard GenBank and GFF3/GTF annotation file formats.
      * Option to directly process unaligned CDS files (perform alignment if requested).
  * **`analyze` subcommand:**
      * Implement tAI calculation (requires tRNA data input).
      * **Further metadata integration:**
          * Allow statistical comparisons grouped by metadata categories.
          * Correlate CA axes with numerical metadata columns.
          * Option to filter analysis based on metadata.
      * Add more statistical comparison options (e.g., pairwise tests).
  * **General:**
      * Generate combined HTML report.
      * Implement CI/CD pipeline (e.g., GitHub Actions).
      * Consider interactive plots (Plotly/Bokeh).
      * Publish package to PyPI.