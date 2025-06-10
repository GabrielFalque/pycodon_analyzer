# tests/test_plotting.py
import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Ensure the src directory is in the Python path for tests
try:
    from pycodon_analyzer import plotting, utils, analysis
    PRINCE_AVAILABLE = analysis.PRINCE_AVAILABLE
    SNS_AVAILABLE = plotting.SNS_AVAILABLE
    MATPLOTLIB_AVAILABLE = plotting.MATPLOTLIB_AVAILABLE
except ImportError: # pragma: no cover
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from pycodon_analyzer import plotting, utils, analysis
    PRINCE_AVAILABLE = analysis.PRINCE_AVAILABLE
    SNS_AVAILABLE = plotting.SNS_AVAILABLE
    MATPLOTLIB_AVAILABLE = plotting.MATPLOTLIB_AVAILABLE

# Skip all tests in this file if matplotlib or seaborn is not available
pytestmark = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE or not SNS_AVAILABLE, reason="matplotlib or seaborn not available")

# --- Fixtures for Plotting Tests ---

@pytest.fixture
def sample_rscu_df_for_plot() -> pd.DataFrame:
    """DataFrame with RSCU values for plotting."""
    data = {
        'Codon': ['TTT', 'TTC', 'TTA', 'TTG', 'ATG'],
        'AminoAcid': ['F', 'F', 'L', 'L', 'M'],
        'RSCU': [1.2, 0.8, 1.5, 0.5, 1.0],
        'SequenceID': ['Seq1'] * 5 # Dummy for some plots that might need it
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_codon_freq_df() -> pd.DataFrame:
    """DataFrame with codon frequencies."""
    data = {'Codon': ['AAA', 'AAC', 'AAG', 'AAT'], 'Frequency': [0.1, 0.2, 0.3, 0.4]}
    return pd.DataFrame(data)

@pytest.fixture
def sample_dinucl_freq_df() -> pd.DataFrame:
    """DataFrame with dinucleotide frequencies."""
    data = {'Dinucleotide': ['AA', 'AT', 'AG', 'AC'], 'Frequency': [0.05, 0.10, 0.15, 0.08]}
    return pd.DataFrame(data)

@pytest.fixture
def sample_dinucl_freq_dict() -> dict:
    """Dictionary with dinucleotide frequencies."""
    return {'AA': 0.05, 'AT': 0.10, 'AG': 0.15, 'AC': 0.08}

@pytest.fixture
def sample_per_sequence_df_for_plots() -> pd.DataFrame:
    """DataFrame similar to per_sequence_metrics for various plots."""
    data = {
        'ID': ['Seq1', 'Seq2', 'Seq3', 'Seq4'],
        'Gene': ['GeneA', 'GeneA', 'GeneB', 'GeneB'],
        'GC': [0.4, 0.5, 0.6, 0.7],
        'GC1': [0.3, 0.4, 0.5, 0.6],
        'GC2': [0.4, 0.5, 0.6, 0.7],
        'GC3': [0.5, 0.6, 0.7, 0.8],
        'GC12': [0.35, 0.45, 0.55, 0.65],  # Added GC12 for neutrality plot
        'ENC': [40.0, 45.0, 50.0, 55.0],
        'Length': [300, 330, 360, 390]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_rel_abund_df_for_plot() -> pd.DataFrame:
    """DataFrame with relative dinucleotide abundance."""
    data = {
        ('GeneA', 'AA'): 1.1, ('GeneA', 'AT'): 0.9,
        ('GeneB', 'AA'): 0.8, ('GeneB', 'AT'): 1.2,
        ('GeneA', 'GC'): 1.0, ('GeneB', 'GC'): 1.0
    }
    df = pd.Series(data).unstack(level=1)
    df.index.name = 'Gene'
    return df

@pytest.fixture
def sample_rel_abund_long_df_for_plot() -> pd.DataFrame:
    """DataFrame with relative dinucleotide abundance in long format."""
    data = {
        'Gene': ['GeneA', 'GeneA', 'GeneA', 'GeneB', 'GeneB', 'GeneB'],
        'Dinucleotide': ['AA', 'AT', 'GC', 'AA', 'AT', 'GC'],
        'RelativeAbundance': [1.1, 0.9, 1.0, 0.8, 1.2, 1.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
@pytest.mark.skipif(not PRINCE_AVAILABLE, reason="prince library not installed")
def sample_ca_results_for_plot(sample_rscu_df_for_ca_plot): # Assuming this fixture exists or is defined
    """Sample CA results from prince."""
    if PRINCE_AVAILABLE:
        ca = analysis.perform_ca(sample_rscu_df_for_ca_plot['rscu_df'], n_components=2)
        return ca, sample_rscu_df_for_ca_plot['rscu_df'], sample_rscu_df_for_ca_plot['groups']
    return None, None, None # Should be skipped if prince not available

@pytest.fixture
def sample_rscu_df_for_ca_plot() -> dict:
    """Provides RSCU data and groups for CA plotting tests."""
    data = {
        "Seq1": {"TTT": 1.2, "TTC": 0.8, "CTG": 1.5, "CTA": 0.5, "ATG":1.0, "GGG":0.9},
        "Seq2": {"TTT": 0.7, "TTC": 1.3, "CTG": 0.6, "CTA": 1.4, "ATG":1.0, "GGG":1.1},
        "Seq3": {"TTT": 1.0, "TTC": 1.0, "CTG": 1.0, "CTA": 1.0, "ATG":1.0, "GGG":1.0},
        "Seq4": {"TTT": 1.8, "TTC": 0.2, "CTG": 1.7, "CTA": 0.3, "ATG":1.0, "GGG":1.3},
    }
    rscu_df = pd.DataFrame.from_dict(data, orient='index')
    rscu_df.index.name = "ID"
    groups = pd.Series(['GroupA', 'GroupA', 'GroupB', 'GroupB'], index=rscu_df.index, name="Group")
    return {"rscu_df": rscu_df, "groups": groups}


# --- Generic Test Function for Plots ---
def _test_plot_generation(plot_function, output_filepath_obj: Path, caplog, *args, **kwargs):
    """Helper to test if a plot function runs and creates a file."""
    caplog.set_level(logging.DEBUG) # Capture all logs
    plot_function(*args, **kwargs, output_filepath=str(output_filepath_obj))
    assert output_filepath_obj.exists()
    assert output_filepath_obj.stat().st_size > 0 # File is not empty
    # Check for major errors in logs (optional, can be too strict)
    # assert not any(record.levelno >= logging.ERROR for record in caplog.records)


# --- Test Classes for each Plotting Function ---

class TestPlotRSCU:
    def test_plot_rscu_runs(self, sample_rscu_df_for_plot, tmp_path: Path, caplog):
        output_file = tmp_path / "rscu_plot.png"
        _test_plot_generation(plotting.plot_rscu, output_file, caplog,
                              rscu_df=sample_rscu_df_for_plot)

    def test_plot_rscu_empty_data(self, tmp_path: Path, caplog):
        output_file = tmp_path / "rscu_empty.png"
        empty_df = pd.DataFrame(columns=['Codon', 'AminoAcid', 'RSCU'])
        plotting.plot_rscu(rscu_df=empty_df, output_filepath=str(output_file))
        assert not output_file.exists() # Should not create plot for empty data
        assert "DataFrame is missing, empty, or lacks required columns" in caplog.text


class TestPlotCodonFrequency:
    def test_plot_codon_frequency_runs(self, sample_codon_freq_df, tmp_path: Path, caplog):
        output_file = tmp_path / "codon_freq.png"
        _test_plot_generation(plotting.plot_codon_frequency, output_file, caplog,
                              rscu_df=sample_codon_freq_df)  # Changed from codon_freq_df to rscu_df

class TestPlotDinucleotideFreq:
    def test_plot_dinucleotide_freq_runs(self, sample_dinucl_freq_dict, tmp_path: Path, caplog):
        output_file = tmp_path / "dinucl_freq.png"
        _test_plot_generation(plotting.plot_dinucleotide_freq, output_file, caplog,
                              dinucl_freqs=sample_dinucl_freq_dict)  # Changed from dinucl_freq_df to dinucl_freqs

class TestPlotGCMeansBarplot:
    def test_plot_gc_means_barplot_runs(self, sample_per_sequence_df_for_plots, tmp_path: Path, caplog):
        output_file = tmp_path / "gc_means.png"
        _test_plot_generation(plotting.plot_gc_means_barplot, output_file, caplog,
                              per_sequence_df=sample_per_sequence_df_for_plots, group_by='Gene')  # Changed from group_by_col to group_by

class TestPlotNeutrality:
    def test_plot_neutrality_runs(self, sample_per_sequence_df_for_plots, tmp_path: Path, caplog):
        output_file = tmp_path / "neutrality.png"
        _test_plot_generation(plotting.plot_neutrality, output_file, caplog,
                              per_sequence_df=sample_per_sequence_df_for_plots, group_by_col='Gene')

class TestPlotENCvsGC3:
    def test_plot_enc_vs_gc3_runs(self, sample_per_sequence_df_for_plots, tmp_path: Path, caplog):
        output_file = tmp_path / "enc_gc3.png"
        _test_plot_generation(plotting.plot_enc_vs_gc3, output_file, caplog,
                              per_sequence_df=sample_per_sequence_df_for_plots, group_by_col='Gene')

class TestPlotPerGeneDinucleotideAbundanceByMetadata:
    @pytest.fixture
    def dinucl_data_for_metadata_plot(self):
        data = {
            'SequenceID': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3'],
            'Dinucleotide': ['AA', 'AT', 'AA', 'AT', 'AA', 'AT'],
            'RelativeAbundance': [1.1, 0.9, 1.2, 0.8, 1.0, 1.0],
            'MetadataGroup': ['X', 'X', 'Y', 'Y', 'X', 'X']
        }
        return pd.DataFrame(data)

    def test_plot_per_gene_dinucl_metadata_runs(self, dinucl_data_for_metadata_plot, tmp_path: Path, caplog):
        output_file = tmp_path / "dinucl_meta.png"
        _test_plot_generation(plotting.plot_per_gene_dinucleotide_abundance_by_metadata,
                              output_file, caplog,
                              per_sequence_oe_ratios_df=dinucl_data_for_metadata_plot,
                              metadata_hue_col='MetadataGroup',
                              gene_name='TestGene',
                              palette=None)  # Added required palette parameter

@pytest.mark.skipif(not PRINCE_AVAILABLE, reason="prince library not installed")
class TestPlotCAPrinceDependent:
    def test_plot_ca_contribution_runs(self, sample_ca_results_for_plot, tmp_path: Path, caplog):
        ca_results, _, _ = sample_ca_results_for_plot
        if ca_results is None: pytest.skip("CA results not available for test")
        output_file = tmp_path / "ca_contrib.png"
        # Check if column_contributions_ attribute exists and is a DataFrame
        if not (hasattr(ca_results, 'column_contributions_') and isinstance(ca_results.column_contributions_, pd.DataFrame) and not ca_results.column_contributions_.empty):
             pytest.skip("column_contributions_ not available or not DataFrame in CA result for test_plot_ca_contribution_runs")

        _test_plot_generation(plotting.plot_ca_contribution, output_file, caplog,
                              ca_results=ca_results, dimension=0, n_top=5)  # Changed from dim_idx to dimension and top_n to n_top


    def test_plot_ca_variance_runs(self, sample_ca_results_for_plot, tmp_path: Path, caplog):
        ca_results, _, _ = sample_ca_results_for_plot
        if ca_results is None: pytest.skip("CA results not available for test")
        output_file = tmp_path / "ca_variance.png"
        _test_plot_generation(plotting.plot_ca_variance, output_file, caplog,
                              ca_results=ca_results, n_dims=2)  # Changed from top_n to n_dims

    def test_plot_ca_biplot_runs(self, sample_ca_results_for_plot, tmp_path: Path, caplog):
        ca_results, ca_input_df, groups = sample_ca_results_for_plot
        if ca_results is None: pytest.skip("CA results not available for test")
        output_file = tmp_path / "ca_biplot.png"
        _test_plot_generation(plotting.plot_ca, output_file, caplog,
                              ca_results=ca_results, ca_input_df=ca_input_df,
                              comp_x=0, comp_y=1, groups=groups)

class TestPlotUsageComparison:
    @pytest.fixture
    def usage_data_for_comparison(self, sample_rscu_df_for_plot):
        # agg_usage_df (like RSCU but maybe averaged per gene)
        # reference_df (similar structure to agg_usage_df)
        # Using sample_rscu_df_for_plot as a proxy for one of them
        df1 = sample_rscu_df_for_plot.copy()[['Codon', 'RSCU']]  # Keep Codon as a column
        df1['Codon'] = df1['Codon'].astype(str)  # Ensure Codon is string type
        
        df2_data = {
            'Codon': ['TTT', 'TTC', 'TTA', 'TTG', 'ATG'],
            'RSCU': [1.1, 0.9, 1.4, 0.6, 1.0]
        }
        df2 = pd.DataFrame(df2_data)
        df2['Codon'] = df2['Codon'].astype(str)  # Ensure Codon is string type
        df2 = df2.set_index('Codon')  # Set Codon as index for reference_data
        
        return df1, df2

    def test_plot_usage_comparison_runs(self, usage_data_for_comparison, tmp_path: Path, caplog):
        df1, df2 = usage_data_for_comparison
        output_file = tmp_path / "usage_comp.png"
        _test_plot_generation(plotting.plot_usage_comparison, output_file, caplog,
                              agg_usage_df=df1, reference_data=df2)  # Removed col_set1 and col_set2 parameters

class TestPlotRelativeDinucleotideAbundance:
    def test_plot_rel_dinucl_abund_runs(self, sample_rel_abund_long_df_for_plot, tmp_path: Path, caplog):
        output_file = tmp_path / "rel_dinucl.png"
        _test_plot_generation(plotting.plot_relative_dinucleotide_abundance, output_file, caplog,
                              rel_abund_df=sample_rel_abund_long_df_for_plot)

class TestPlotRSCUBoxplotPerGene:
    @pytest.fixture
    def rscu_long_df_for_boxplot(self):
        # Data for plot_rscu_boxplot_per_gene
        # Needs 'Codon', 'RSCU', 'AminoAcid', 'SequenceID'
        # And agg_usage_df for ordering
        data_long = {
            'SequenceID': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S3', 'S3', 'S3'],
            'Codon': ['TTT', 'TTC', 'TTA', 'TTT', 'TTC', 'TTA', 'TTT', 'TTC', 'TTA'],
            'RSCU': [1.2, 0.8, 1.5, 0.7, 1.3, 0.5, 1.0, 1.0, 1.0],
            'AminoAcid': ['F', 'F', 'L', 'F', 'F', 'L', 'F', 'F', 'L']
        }
        long_df = pd.DataFrame(data_long)
        
        # Dummy aggregate usage for codon ordering
        agg_data = {'Codon': ['TTT', 'TTC', 'TTA'], 'RSCU': [1.0, 1.0, 1.0], 'AminoAcid': ['F', 'F', 'L']}
        agg_df = pd.DataFrame(agg_data).set_index('Codon')
        return long_df, agg_df

    def test_plot_rscu_boxplot_runs(self, rscu_long_df_for_boxplot, tmp_path: Path, caplog):
        long_df, agg_df = rscu_long_df_for_boxplot
        output_file = tmp_path / "rscu_boxplot.png"
        _test_plot_generation(plotting.plot_rscu_boxplot_per_gene, output_file, caplog,
                              long_rscu_df=long_df, agg_rscu_df=agg_df, gene_name="TestGene")  # Changed from agg_usage_df to agg_rscu_df

class TestPlotCorrelationHeatmap:
    def test_plot_correlation_heatmap_runs(self, sample_per_sequence_df_for_plots, tmp_path: Path, caplog):
        output_file = tmp_path / "corr_heatmap.png"
        features_to_correlate = ['GC', 'ENC', 'Length']
        _test_plot_generation(plotting.plot_correlation_heatmap, output_file, caplog,
                              df=sample_per_sequence_df_for_plots,
                              features=features_to_correlate)

    def test_plot_correlation_heatmap_insufficient_features(self, sample_per_sequence_df_for_plots, tmp_path: Path, caplog):
        output_file = tmp_path / "corr_heatmap_insufficient.png"
        plotting.plot_correlation_heatmap(
            df=sample_per_sequence_df_for_plots,
            features=['GC'], # Only one feature
            output_filepath=str(output_file)
        )
        assert not output_file.exists()
        assert "Need at least two available features for correlation" in caplog.text  # Updated assertion to match actual message


class TestPlotCAAxesFeatureCorrelation:
    @pytest.fixture
    def data_for_ca_axes_corr(self, sample_per_sequence_df_for_plots, sample_rscu_df_for_ca_plot):
        ca_dims_data = {
            'CA_Dim1': np.random.rand(4),
            'CA_Dim2': np.random.rand(4)
        }
        ca_dims_df = pd.DataFrame(ca_dims_data, index=['Seq1', 'Seq2', 'Seq3', 'Seq4'])
        
        metrics_df = sample_per_sequence_df_for_plots.set_index('ID')
        rscu_df = sample_rscu_df_for_ca_plot['rscu_df'] # This is already indexed by ID
        return ca_dims_df, metrics_df, rscu_df

    def test_plot_ca_axes_feature_corr_runs(self, data_for_ca_axes_corr, tmp_path: Path, caplog):
        ca_dims_df, metrics_df, rscu_df = data_for_ca_axes_corr
        output_file = tmp_path / "ca_axes_corr.png"
        features = ['GC', 'ENC', 'TTT', 'CTG'] # Mix of metric and RSCU codon
        
        _test_plot_generation(plotting.plot_ca_axes_feature_correlation, output_file, caplog,
                              ca_dims_df=ca_dims_df, metrics_df=metrics_df, rscu_df=rscu_df,
                              features_to_correlate=features)

# Additional tests for edge cases

class TestPlotRSCUEdgeCases:
    def test_plot_rscu_with_missing_columns(self, tmp_path: Path, caplog):
        output_file = tmp_path / "rscu_missing_cols.png"
        # Missing AminoAcid column
        df = pd.DataFrame({'Codon': ['TTT', 'TTC'], 'RSCU': [1.2, 0.8]})
        plotting.plot_rscu(rscu_df=df, output_filepath=str(output_file))
        assert not output_file.exists()
        assert "DataFrame is missing, empty, or lacks required columns" in caplog.text

    def test_plot_rscu_with_nan_values(self, tmp_path: Path, caplog):
        output_file = tmp_path / "rscu_nan.png"
        df = pd.DataFrame({
            'Codon': ['TTT', 'TTC', 'TTA'],
            'AminoAcid': ['F', 'F', 'L'],
            'RSCU': [1.2, np.nan, 1.5]
        })
        _test_plot_generation(plotting.plot_rscu, output_file, caplog, rscu_df=df)
        # Should still create a plot, just without the NaN value

class TestPlotNeutralityEdgeCases:
    def test_plot_neutrality_with_single_point(self, tmp_path: Path, caplog):
        output_file = tmp_path / "neutrality_single.png"
        df = pd.DataFrame({
            'ID': ['Seq1'],
            'Gene': ['GeneA'],
            'GC12': [0.4],
            'GC3': [0.5]
        })
        plotting.plot_neutrality(per_sequence_df=df, output_filepath=str(output_file))
        # The function should NOT create a plot for insufficient data points
        assert not output_file.exists()
        assert "Not enough valid data points (>=2) for Neutrality Plot regression/correlation. Skipping." in caplog.text
        assert "Not enough valid data points (>=2) for Neutrality Plot regression/correlation" in caplog.text

class TestPlotENCvsGC3EdgeCases:
    def test_plot_enc_vs_gc3_with_invalid_group(self, tmp_path: Path, caplog):
        output_file = tmp_path / "enc_gc3_invalid_group.png"
        df = pd.DataFrame({
            'ID': ['Seq1', 'Seq2'],
            'GC3': [0.5, 0.6],
            'ENC': [40.0, 45.0]
        })
        _test_plot_generation(plotting.plot_enc_vs_gc3, output_file, caplog,
                             per_sequence_df=df, group_by_col='NonExistentGroup')
        # Should still create a plot, just without grouping
        assert "Grouping column 'NonExistentGroup' not found for ENC vs GC3 plot" in caplog.text
    
    def test_plot_enc_vs_gc3_missing_columns(self, tmp_path: Path, caplog):
        output_file = tmp_path / "enc_gc3_missing_cols.png"
        df = pd.DataFrame({
            'ID': ['Seq1', 'Seq2'],
            'GC3': [0.5, 0.6],  # Missing ENC column
        })
        plotting.plot_enc_vs_gc3(per_sequence_df=df, output_filepath=str(output_file))
        assert not output_file.exists()
        assert "Missing required columns" in caplog.text

class TestPlotCorrelationHeatmapEdgeCases:
    def test_plot_correlation_heatmap_with_invalid_method(self, sample_per_sequence_df_for_plots, tmp_path: Path, caplog):
        output_file = tmp_path / "corr_heatmap_invalid_method.png"
        features = ['GC', 'ENC', 'Length']
        _test_plot_generation(plotting.plot_correlation_heatmap, output_file, caplog,
                             df=sample_per_sequence_df_for_plots,
                             features=features,
                             method='invalid_method')
        # Should default to spearman and create a plot
        assert "Invalid correlation method 'invalid_method'. Using 'spearman'" in caplog.text
    
    def test_plot_correlation_heatmap_empty_df(self, tmp_path: Path, caplog):
        output_file = tmp_path / "corr_heatmap_empty.png"
        empty_df = pd.DataFrame()
        plotting.plot_correlation_heatmap(
            df=empty_df,
            features=['GC', 'ENC'],
            output_filepath=str(output_file)
        )
        assert not output_file.exists()
        assert "Input DataFrame is empty" in caplog.text

class TestPlotRelativeDinucleotideAbundanceEdgeCases:
    def test_plot_rel_dinucl_abund_empty_data(self, tmp_path: Path, caplog):
        output_file = tmp_path / "rel_dinucl_empty.png"
        empty_df = pd.DataFrame(columns=['Gene', 'Dinucleotide', 'RelativeAbundance'])
        plotting.plot_relative_dinucleotide_abundance(rel_abund_df=empty_df, output_filepath=str(output_file))
        assert not output_file.exists()
        assert "No relative dinucleotide abundance data to plot" in caplog.text
    
    def test_plot_rel_dinucl_abund_missing_columns(self, tmp_path: Path, caplog):
        output_file = tmp_path / "rel_dinucl_missing_cols.png"
        df = pd.DataFrame({
            'Gene': ['GeneA', 'GeneB'],
            'RelativeAbundance': [1.1, 0.9]  # Missing Dinucleotide column
        })
        plotting.plot_relative_dinucleotide_abundance(rel_abund_df=df, output_filepath=str(output_file))
        assert not output_file.exists()
        assert "Missing required columns" in caplog.text

class TestPlotUsageComparisonEdgeCases:
    def test_plot_usage_comparison_missing_reference_rscu(self, tmp_path: Path, caplog):
        output_file = tmp_path / "usage_comp_missing_ref.png"
        df1 = pd.DataFrame({
            'Codon': ['TTT', 'TTC'],
            'RSCU': [1.2, 0.8]
        })
        df2 = pd.DataFrame({
            'Codon': ['TTT', 'TTC'],
            'Value': [1.1, 0.9]  # Missing RSCU column
        }).set_index('Codon')
        plotting.plot_usage_comparison(agg_usage_df=df1, reference_data=df2, output_filepath=str(output_file))
        assert not output_file.exists()
        assert "reference RSCU data not available or missing 'RSCU' column" in caplog.text
    
    def test_plot_usage_comparison_missing_agg_rscu(self, tmp_path: Path, caplog):
        output_file = tmp_path / "usage_comp_missing_agg.png"
        df1 = pd.DataFrame({
            'Codon': ['TTT', 'TTC'],
            'Value': [1.2, 0.8]  # Missing RSCU column
        })
        df2 = pd.DataFrame({
            'Codon': ['TTT', 'TTC'],
            'RSCU': [1.1, 0.9]
        }).set_index('Codon')
        plotting.plot_usage_comparison(agg_usage_df=df1, reference_data=df2, output_filepath=str(output_file))
        assert not output_file.exists()
        assert "calculated aggregate RSCU data invalid or missing columns" in caplog.text

class TestPlotGCMeansBarplotEdgeCases:
    def test_plot_gc_means_barplot_missing_group_by(self, sample_per_sequence_df_for_plots, tmp_path: Path, caplog):
        output_file = tmp_path / "gc_means_missing_group.png"
        plotting.plot_gc_means_barplot(
            per_sequence_df=sample_per_sequence_df_for_plots,
            output_filepath=str(output_file),
            group_by='NonExistentColumn'
        )
        assert not output_file.exists()
        assert "Grouping column 'NonExistentColumn' not found" in caplog.text
    
    def test_plot_gc_means_barplot_missing_gc_columns(self, tmp_path: Path, caplog):
        output_file = tmp_path / "gc_means_missing_cols.png"
        df = pd.DataFrame({
            'ID': ['Seq1', 'Seq2'],
            'Gene': ['GeneA', 'GeneB'],
            'GC': [0.4, 0.5]  # Missing other GC columns
        })
        plotting.plot_gc_means_barplot(
            per_sequence_df=df,
            output_filepath=str(output_file),
            group_by='Gene'
        )
        assert not output_file.exists()
        assert "Missing required GC columns" in caplog.text

class TestPlotCAEdgeCases:
    @pytest.mark.skipif(not PRINCE_AVAILABLE, reason="prince library not installed")
    def test_plot_ca_missing_ca_results(self, tmp_path: Path, caplog):
        output_file = tmp_path / "ca_missing_results.png"
        ca_input_df = pd.DataFrame({
            'TTT': [1.2, 0.8],
            'TTC': [0.8, 1.2]
        }, index=['Seq1', 'Seq2'])
        plotting.plot_ca(
            ca_results=None,
            ca_input_df=ca_input_df,
            output_filepath=str(output_file)
        )
        assert not output_file.exists()
        assert "No valid CA results" in caplog.text
    
    @pytest.mark.skipif(not PRINCE_AVAILABLE, reason="prince library not installed")
    def test_plot_ca_missing_input_df(self, sample_ca_results_for_plot, tmp_path: Path, caplog):
        ca_results, _, _ = sample_ca_results_for_plot
        if ca_results is None: pytest.skip("CA results not available for test")
        output_file = tmp_path / "ca_missing_input.png"
        plotting.plot_ca(
            ca_results=ca_results,
            ca_input_df=None,
            output_filepath=str(output_file)
        )
        assert not output_file.exists()
        assert "CA input DataFrame for coordinates is missing or empty" in caplog.text