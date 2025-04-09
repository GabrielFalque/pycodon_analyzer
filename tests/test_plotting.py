# tests/test_plotting.py
import traceback
import pytest # type: ignore
import os
import pandas as pd
import numpy as np

# Adjust import path
try:
    from src.pycodon_analyzer import plotting # type: ignore
except ImportError:
     import sys
     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
     from pycodon_analyzer import plotting

# --- Mock Data for Plotting ---

@pytest.fixture
def sample_agg_usage_df():
    """Dummy aggregate usage DataFrame."""
    data = {
        'Codon': ['AAA', 'AAG', 'GGG', 'GGC', 'TTT', 'TTC'],
        'AminoAcid': ['K', 'K', 'G', 'G', 'F', 'F'],
        'Count': [10, 30, 5, 45, 20, 20],
        'Frequency': [0.09, 0.27, 0.045, 0.409, 0.09, 0.09], # Approx
        'RSCU': [0.5, 1.5, 0.18, 1.81, 1.0, 1.0] # Approx
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_long_rscu_df():
    """Dummy long format RSCU DataFrame."""
    data = {
        'SequenceID': ['Seq1', 'Seq1', 'Seq1', 'Seq2', 'Seq2', 'Seq2', 'Seq1', 'Seq2'],
        'Codon': ['AAA', 'GGG', 'TTT', 'AAG', 'GGC', 'TTC', 'AAG', 'GGG'],
        'RSCU': [0.4, 0.2, 1.1, 1.6, 1.7, 0.9, 1.55, 0.15],
        'AminoAcid': ['K', 'G', 'F', 'K', 'G', 'F', 'K', 'G'],
        'Gene': ['GeneX', 'GeneX', 'GeneX', 'GeneX', 'GeneX', 'GeneX', 'GeneX', 'GeneX']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_per_sequence_df():
    """Dummy combined per-sequence metrics DataFrame."""
    data = {
        'ID': ['S1_G1', 'S2_G1', 'S1_G2', 'S2_G2', 'S1_comp', 'S2_comp'],
        'Gene': ['Gene1', 'Gene1', 'Gene2', 'Gene2', 'complete', 'complete'],
        'GC': [50, 55, 40, 45, 45, 50],
        'GC1': [51, 56, 41, 46, 46, 51],
        'GC2': [52, 57, 42, 47, 47, 52],
        'GC3': [47, 52, 37, 42, 42, 47],
        'GC12': [51.5, 56.5, 41.5, 46.5, 46.5, 51.5],
        'ENC': [55, 58, 45, 48, 50, 53],
        'CAI': [0.7, 0.75, 0.6, 0.65, 0.68, 0.72],
        'RCDI': [1.4, 1.3, 1.6, 1.5, 1.5, 1.4],
        'Aromaticity': [0.1, 0.11, 0.12, 0.13, 0.11, 0.12],
        'GRAVY': [-0.1, -0.2, 0.1, 0.0, -0.05, -0.15],
        'Length': [300, 300, 600, 600, 900, 900],
        'TotalCodons': [100, 100, 200, 200, 300, 300]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_ca_results():
    """Creates a mock Prince CA object with minimal necessary attributes."""
    try:
        import prince
        # Create dummy data for fitting
        # Needs >= 2 rows, >= 2 cols
        dummy_ca_data = pd.DataFrame({
            'AAA': [1.1, 0.9, 1.0], 'AAG': [0.9, 1.1, 1.0], 'CCC': [1.0, 1.0, 1.0]
        }, index=['G1__S1', 'G1__S2', 'G2__S1'])

        ca = prince.CA(n_components=2, random_state=42)
        ca.fit(dummy_ca_data)
        return ca, dummy_ca_data
    except ImportError:
        pytest.skip("Prince library not installed, skipping CA plot tests.")
    except Exception as e: # Catch potential errors during fit with dummy data
         pytest.skip(f"Skipping CA plot tests due to CA fit error: {e}")


# --- Basic Plotting Tests (Check if they run without error and create file) ---

PLOT_FUNCTIONS = [
    plotting.plot_rscu,
    plotting.plot_codon_frequency,
    plotting.plot_gc_means_barplot,
    plotting.plot_neutrality,
    plotting.plot_enc_vs_gc3,
    plotting.plot_relative_dinucleotide_abundance,
    plotting.plot_correlation_heatmap,
    plotting.plot_ca_variance,
    plotting.plot_ca_contribution,
    plotting.plot_ca,
    plotting.plot_rscu_boxplot_per_gene,
]

@pytest.mark.parametrize("plot_func", PLOT_FUNCTIONS)
def test_plotting_functions_run(plot_func, tmp_path, sample_agg_usage_df, sample_per_sequence_df, sample_long_rscu_df, sample_ca_results):
    """Test if plotting functions execute without raising errors and create a file."""
    output_dir = str(tmp_path)
    file_format = "png"
    plot_name = plot_func.__name__
    print(f"\nTesting plot function: {plot_name}")

    # Prepare args based on function name (this is brittle, consider better approach if complex)
    args = {"output_dir": output_dir, "file_format": file_format}
    ran_successfully = False

    try:
        if plot_name in ["plot_rscu", "plot_codon_frequency"]:
             args["rscu_df"] = sample_agg_usage_df
             plot_func(**args)
        elif plot_name in ["plot_gc_means_barplot", "plot_neutrality", "plot_enc_vs_gc3", "plot_correlation_heatmap"]:
             # --- Argument name for heatmap is 'df' ---
             if plot_name == "plot_correlation_heatmap":
                  args["df"] = sample_per_sequence_df # Use 'df' key
                  args["features"] = ['GC', 'ENC', 'CAI', 'RCDI'] # Sample features
             else:
                  args["per_sequence_df"] = sample_per_sequence_df
                  args["group_by"] = 'Gene'
             plot_func(**args)
        elif plot_name == "plot_relative_dinucleotide_abundance":
            # Need dummy rel_abund_df
            rel_data = {'Gene': ['G1','G1','G2','G2'], 'Dinucleotide': ['AA','AC','AA','AC'], 'RelativeAbundance': [1.1,0.9,1.0,1.0]}
            args["rel_abund_df"] = pd.DataFrame(rel_data)
            plot_func(**args)
        elif plot_name == "plot_rscu_boxplot_per_gene":
             args["long_rscu_df"] = sample_long_rscu_df
             args["agg_rscu_df"] = sample_agg_usage_df # Needs agg data too
             args["gene_name"] = "SampleGene"
             plot_func(**args)
        elif plot_name in ["plot_ca_variance", "plot_ca_contribution", "plot_ca"]:
            if sample_ca_results is None: pytest.skip("Skipping CA plot test.")
            ca_res, ca_input = sample_ca_results
            args["ca_results"] = ca_res
            if plot_name == "plot_ca_variance":
                 args["n_dims"] = 2
            elif plot_name == "plot_ca_contribution":
                 args["dimension"] = 0
                 args["n_top"] = 5
            elif plot_name == "plot_ca":
                 args["ca_input_df"] = ca_input
                 args["groups"] = pd.Series(['G1', 'G1', 'G2'], index=ca_input.index, name='Gene') # Dummy groups matching fixture
            plot_func(**args)
        else:
             pytest.skip(f"Argument setup not implemented for {plot_name}")
             return # Skip if args not set

        ran_successfully = True

    except Exception as e:
         pytest.fail(f"{plot_name} raised an exception: {e}\n{traceback.format_exc()}")

    # Check if at least one file was created (filename might vary)
    output_files = os.listdir(output_dir)
    assert len(output_files) > 0, f"{plot_name} did not create an output file in {output_dir}"
    print(f"  {plot_name} ran and created files: {output_files}")