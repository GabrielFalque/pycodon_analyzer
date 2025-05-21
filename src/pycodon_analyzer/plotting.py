# src/pycodon_analyzer/plotting.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Functions for generating plots related to codon usage and sequence properties.
Uses Matplotlib and Seaborn for plotting.
"""
import re
import os
import sys
import logging # <-- Import logging
import traceback # Keep for detailed error logging if needed via logger.exception
from typing import List, Dict, Optional, Any, Tuple, Set, Union, TYPE_CHECKING


# Third-party library imports with error handling for optional ones
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes # For type hinting
    from matplotlib.figure import Figure # For type hinting
    from matplotlib.collections import PathCollection # For type hinting scatter plot object
except ImportError:
    # Critical dependency, log and exit might be too late if logger not set.
    print("CRITICAL ERROR: matplotlib is required but not installed. Please install it (`pip install matplotlib`).", file=sys.stderr)
    sys.exit(1)

try:
    import seaborn as sns
except ImportError:
    print("CRITICAL ERROR: seaborn is required but not installed. Please install it (`pip install seaborn`).", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("CRITICAL ERROR: pandas is required but not installed. Please install it (`pip install pandas`).", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("CRITICAL ERROR: numpy is required but not installed. Please install it (`pip install numpy`).", file=sys.stderr)
    sys.exit(1)

if TYPE_CHECKING:
    from scipy import stats as scipy_stats_module # For type checking if used
    SCIPY_AVAILABLE = True
    # ScipyStatsModule = type(scipy.stats) # More precise if needed
else:
    try:
        from scipy import stats as scipy_stats_module
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        scipy_stats_module = None # Runtime check

# Optional: Try importing prince for type hinting if needed
if TYPE_CHECKING:
    import prince # Import only for type checking
    PrinceCA = prince.CA # Actual type for type checker
else:
    PrinceCA = Any # Fallback type for runtime
    try:
        import prince
    except ImportError:
        prince = None # Set to None if not found

try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False

# --- Configure logging for this module ---
# Gets the logger instance configured in cli.py (or root logger if run standalone)
logger = logging.getLogger(__name__)

# Set default seaborn theme (optional, place where it's guaranteed to run once)
try:
    sns.set_theme(style="ticks", palette="deep")
except Exception as e:
    logger.warning(f"Could not set default seaborn theme: {e}")


# --- AA Code Mapping (Unchanged) ---
AA_1_TO_3: Dict[str, str] = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
}
AA_3_TO_1: Dict[str, str] = {v: k for k, v in AA_1_TO_3.items()}
AA_ORDER: List[str] = sorted(AA_1_TO_3.keys())

# --- Utility function to sanitize filenames (with type hints) ---
def sanitize_filename(name: Any) -> str:
    """Removes or replaces characters problematic for filenames."""
    if not isinstance(name, str): name = str(name) # Ensure string
    name = re.sub(r'[\[\]()]', '', name) # Remove brackets and parentheses
    name = re.sub(r'[\s/:]+', '_', name) # Replace whitespace, slashes, colons with underscore
    name = re.sub(r'[^\w.\-]+', '', name) # Remove remaining non-alphanumeric (excluding underscore, hyphen, period)
    name = name.strip('._-') # Remove leading/trailing problematic chars
    # Prevent names like "." or ""
    return name if name else "_invalid_name_"


# === Aggregate Plots ===

def plot_rscu(rscu_df: pd.DataFrame, output_dir: str, file_format: str = 'svg') -> None:
    """
    Generates a bar plot of aggregate RSCU values, grouped by amino acid.

    Args:
        rscu_df (pd.DataFrame): DataFrame containing RSCU values (must have
                                'AminoAcid', 'Codon', 'RSCU' columns).
        output_dir (str): Directory to save the plot.
        file_format (str): Format to save the plot (e.g., 'png', 'svg', 'pdf'). Default 'svg'.
    """
    required_cols = ['AminoAcid', 'Codon', 'RSCU']
    if rscu_df is None or rscu_df.empty or not all(col in rscu_df.columns for col in required_cols):
        logger.warning("Cannot plot RSCU. DataFrame is missing, empty, or lacks required columns (AminoAcid, Codon, RSCU).")
        return

    fig: Optional[Figure] = None # Initialize fig object for finally block
    try:
        # Filter out NaN RSCU values and sort for consistent plotting
        rscu_df_plot = rscu_df.dropna(subset=['RSCU', 'AminoAcid']).copy()
        # Ensure Codon and AA are suitable for plotting
        rscu_df_plot['Codon'] = rscu_df_plot['Codon'].astype(str)
        rscu_df_plot['AminoAcid'] = rscu_df_plot['AminoAcid'].astype(str)
        # Ensure RSCU is numeric
        rscu_df_plot['RSCU'] = pd.to_numeric(rscu_df_plot['RSCU'], errors='coerce')
        rscu_df_plot.dropna(subset=['RSCU'], inplace=True) # Drop if conversion failed

        rscu_df_plot.sort_values(by=['AminoAcid', 'Codon'], inplace=True)

        if rscu_df_plot.empty:
            logger.warning("No non-NaN RSCU data available to plot after filtering.")
            return

        fig, ax = plt.subplots(figsize=(18, 7)) # Create figure and axes

        # Use seaborn barplot
        sns.barplot(x='Codon', y='RSCU', hue='AminoAcid',
                    data=rscu_df_plot,
                    dodge=False, palette='tab20', ax=ax)

        ax.set_title('Relative Synonymous Codon Usage (RSCU)', fontsize=16)
        ax.set_xlabel('Codon', fontsize=12)
        ax.set_ylabel('RSCU Value', fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Improve legend
        try:
            ax.legend(title='Amino Acid', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout(rect=[0, 0, 0.9, 1])
        except Exception as legend_err:
            logger.warning(f"Could not optimally place RSCU plot legend: {legend_err}. Using default layout.")
            plt.tight_layout()

        # Save the plot
        output_filename = os.path.join(output_dir, f"rscu_barplot.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"RSCU plot saved to: {output_filename}")

    except (ValueError, TypeError) as data_err:
        logger.error(f"Data error during RSCU plot generation (check data types/values): {data_err}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during RSCU plot generation: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_codon_frequency(rscu_df: pd.DataFrame, output_dir: str, file_format: str = 'svg') -> None:
    """
    Generates a bar plot of aggregate codon frequencies.

    Args:
        rscu_df (pd.DataFrame): DataFrame containing frequency values (must have
                                'Codon', 'Frequency' columns).
        output_dir (str): Directory to save the plot.
        file_format (str): Format to save the plot. Default 'svg'.
    """
    required_cols = ['Codon', 'Frequency']
    if rscu_df is None or rscu_df.empty or not all(col in rscu_df.columns for col in required_cols):
        logger.warning("Cannot plot Codon Frequency. DataFrame is missing, empty, or lacks required columns (Codon, Frequency).")
        return

    fig: Optional[Figure] = None
    try:
        # Prepare data for plotting
        freq_df_plot = rscu_df[['Codon', 'Frequency']].dropna().copy()
        freq_df_plot['Codon'] = freq_df_plot['Codon'].astype(str)
        freq_df_plot['Frequency'] = pd.to_numeric(freq_df_plot['Frequency'], errors='coerce')
        freq_df_plot.dropna(subset=['Frequency'], inplace=True)
        freq_df_plot.sort_values(by='Codon', inplace=True)

        if freq_df_plot.empty:
            logger.warning("No valid Codon Frequency data available to plot after filtering.")
            return

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(x='Codon', y='Frequency', data=freq_df_plot, color='skyblue', ax=ax)

        ax.set_title('Codon Frequency', fontsize=16)
        ax.set_xlabel('Codon', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot
        output_filename = os.path.join(output_dir, f"codon_frequency_barplot.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Codon frequency plot saved to: {output_filename}")

    except (ValueError, TypeError) as data_err:
        logger.error(f"Error preparing data or plotting Codon Frequency: {data_err}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during Codon Frequency plot generation: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_dinucleotide_freq(dinucl_freqs: Dict[str, float], output_dir: str, file_format: str = 'svg') -> None:
    """
    Plots relative dinucleotide frequencies as a bar chart.

    Args:
        dinucl_freqs (Dict[str, float]): Dictionary mapping dinucleotides to frequencies.
        output_dir (str): Directory to save the plot.
        file_format (str): Format to save the plot. Default 'svg'.
    """
    if not dinucl_freqs:
        logger.warning("No dinucleotide frequency data provided to plot.")
        return

    fig: Optional[Figure] = None
    try:
        # Convert dict to DataFrame
        freq_df = pd.DataFrame.from_dict(dinucl_freqs, orient='index', columns=['Frequency'])
        freq_df['Frequency'] = pd.to_numeric(freq_df['Frequency'], errors='coerce')
        freq_df.dropna(inplace=True)
        freq_df.sort_index(inplace=True)

    except (ValueError, TypeError) as df_err:
         logger.error(f"Error creating DataFrame for dinucleotide plot: {df_err}")
         return

    if freq_df.empty:
        logger.warning("Dinucleotide frequency data is empty or all NaN after conversion.")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=freq_df.index, y=freq_df['Frequency'], palette='coolwarm', ax=ax)

        ax.set_title('Relative Dinucleotide Frequencies')
        ax.set_xlabel('Dinucleotide')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"dinucleotide_freq.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Dinucleotide frequency plot saved to: {output_filename}")

    except Exception as e:
         logger.exception(f"Error during seaborn barplot generation for Dinucleotides: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_gc_means_barplot(per_sequence_df: pd.DataFrame, output_dir: str, file_format: str = 'svg', group_by: str = 'Gene') -> None:
    """
    Plots a grouped bar chart of mean GC%, GC1-3%, GC12 values per group (e.g., 'Gene').

    Args:
        per_sequence_df (pd.DataFrame): DataFrame containing per-sequence metrics.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format. Default 'svg'.
        group_by (str): Column name in per_sequence_df to group by. Default 'Gene'.
    """
    gc_cols: List[str] = ['GC', 'GC1', 'GC2', 'GC3', 'GC12']
    if per_sequence_df is None or per_sequence_df.empty:
         logger.warning("Input DataFrame is empty for GC means barplot.")
         return
    if group_by not in per_sequence_df.columns:
         logger.error(f"Grouping column '{group_by}' not found for GC means plot.")
         return
    missing_gc_cols = [col for col in gc_cols if col not in per_sequence_df.columns]
    if missing_gc_cols:
        logger.error(f"Missing required GC columns ({', '.join(missing_gc_cols)}) for GC means plot.")
        return

    fig: Optional[Figure] = None
    try:
        # Ensure GC columns are numeric
        df_numeric = per_sequence_df.copy()
        for col in gc_cols:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        # Calculate mean GC values per group
        mean_gc_df = df_numeric.groupby(group_by)[gc_cols].mean().reset_index()

        if mean_gc_df.empty:
            logger.warning(f"No data after grouping by '{group_by}' for GC means plot.")
            return

        # Melt for easy plotting
        mean_gc_melted = mean_gc_df.melt(id_vars=[group_by], var_name='GC_Type', value_name='Mean_GC_Content')
        mean_gc_melted.dropna(subset=['Mean_GC_Content'], inplace=True)

        if mean_gc_melted.empty:
             logger.warning("No valid mean GC data after melting/filtering for GC means plot.")
             return

        # Sort groups for consistent order
        unique_groups = mean_gc_melted[group_by].unique()
        group_order = sorted([g for g in unique_groups if g != 'complete'])
        if 'complete' in unique_groups: group_order.append('complete')

        fig, ax = plt.subplots(figsize=(max(8, len(group_order) * 0.8), 6))

        sns.barplot(data=mean_gc_melted, x=group_by, y='Mean_GC_Content', hue='GC_Type',
                    order=group_order, palette='viridis', ax=ax)

        ax.set_title(f'Mean GC Content by {group_by}', fontsize=14)
        ax.set_xlabel(group_by, fontsize=12)
        ax.set_ylabel('Mean GC Content (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        try:
            ax.legend(title='GC Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
            plt.tight_layout(rect=[0, 0, 0.88, 1])
        except Exception as legend_err:
             logger.warning(f"Could not place GC means plot legend optimally: {legend_err}.")
             plt.tight_layout()

        output_filename = os.path.join(output_dir, f"gc_means_barplot_by_{sanitize_filename(group_by)}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"GC means barplot saved to: {output_filename}")

    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during GC means plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating GC means barplot: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_neutrality(per_sequence_df: pd.DataFrame, 
                    output_dir: str, 
                    file_format: str = 'svg', 
                    group_by: Optional[str] = 'Gene',
                    palette: Optional[Dict[str, Any]] = None
                    ) -> None:
    """
    Generates a Neutrality Plot (GC12 vs GC3). Optionally colors points by group.

    Args:
        per_sequence_df (pd.DataFrame): DataFrame with per-sequence metrics (must have GC12, GC3).
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format. Default 'svg'.
        group_by (Optional[str]): Column name to group/hue by. Default None.
    """
    required_cols = ['GC12', 'GC3']
    if per_sequence_df is None or per_sequence_df.empty:
         logger.warning("Input DataFrame empty for Neutrality plot.")
         return
    missing_cols = [col for col in required_cols if col not in per_sequence_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns ({', '.join(missing_cols)}) for Neutrality plot.")
        return
    if group_by and group_by not in per_sequence_df.columns:
        logger.warning(f"Grouping column '{group_by}' not found for Neutrality plot. Plotting without grouping.")
        group_by = None
        palette=None

    fig: Optional[Figure] = None
    scatter_plot_object: Optional[PathCollection] = None

    try:
        plot_df = per_sequence_df.copy()
        plot_df['GC3_num'] = pd.to_numeric(plot_df['GC3'], errors='coerce')
        plot_df['GC12_num'] = pd.to_numeric(plot_df['GC12'], errors='coerce')
        plot_df_valid = plot_df.dropna(subset=['GC3_num', 'GC12_num'])

        if len(plot_df_valid) < 2:
            logger.warning("Not enough valid data points (>=2) for Neutrality Plot regression/correlation.")
            # Proceed to plot scatter if points exist? Or return? Let's proceed with scatter.
            if plot_df_valid.empty: return # Return if absolutely no points

        # Determine grouping possibility
        perform_grouping = group_by and not plot_df_valid[group_by].isnull().all() and plot_df_valid[group_by].nunique() > 1
        hue_col = group_by if perform_grouping else None
        group_title = group_by if perform_grouping else "Group"

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        scatter_plot_object = sns.scatterplot(
            data=plot_df_valid, x='GC3_num', y='GC12_num', hue=hue_col,
            alpha=0.7, s=60, palette=palette, legend='full' if hue_col else False, ax=ax)

        # Overall regression line (only if enough points)
        slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
        r_squared = np.nan
        if len(plot_df_valid) >= 2 and scipy_stats_module is not None: # Check if scipy.stats was imported
            try:
                 slope, intercept, r_value, p_value, std_err = scipy_stats_module.linregress(plot_df_valid['GC3_num'], plot_df_valid['GC12_num'])
                 r_squared = r_value**2 if pd.notna(r_value) else np.nan
                 x_range = np.array([plot_df_valid['GC3_num'].min(), plot_df_valid['GC3_num'].max()])
                 y_vals = intercept + slope * x_range
                 ax.plot(x_range, y_vals, color="black", lw=1.5, ls='--', label=f"Overall (R²={r_squared:.3f})")
            except (ValueError, TypeError) as lin_reg_err:
                 logger.warning(f"Could not calculate overall regression for Neutrality plot: {lin_reg_err}")
                 slope, r_squared = np.nan, np.nan
        elif len(plot_df_valid) >= 2 and scipy_stats_module is None:
             logger.warning("Cannot calculate regression line for Neutrality plot: scipy.stats not available.")


        # Plotting customizations
        ax.set_title('Neutrality Plot (GC12 vs GC3)', fontsize=14)
        ax.set_xlabel('GC3 Content (%)', fontsize=12)
        ax.set_ylabel('GC12 Content (%)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        if not np.isnan(slope):
             ax.text(0.05, 0.95, f'Overall Slope={slope:.3f}', transform=ax.transAxes, va='top',
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        # Adjust Axis Limits
        min_gc3, max_gc3 = plot_df_valid['GC3_num'].min(), plot_df_valid['GC3_num'].max()
        min_gc12, max_gc12 = plot_df_valid['GC12_num'].min(), plot_df_valid['GC12_num'].max()
        x_pad = max((max_gc3 - min_gc3) * 0.05, 2)
        y_pad = max((max_gc12 - min_gc12) * 0.05, 2)
        x_lim = (max(0, min_gc3 - x_pad), min(100, max_gc3 + x_pad))
        y_lim = (max(0, min_gc12 - y_pad), min(100, max_gc12 + y_pad))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Add y=x line
        diag_lims = [max(x_lim[0], y_lim[0]), min(x_lim[1], y_lim[1])]
        ax.plot(diag_lims, diag_lims, 'gray', linestyle=':', alpha=0.7, lw=1, label='y=x')
        ax.tick_params(axis='both', which='major', labelsize=10)

        # --- Add Adjusted Gene Labels ---
        if perform_grouping and hue_col:
            texts: List[plt.Text] = []
            group_data = plot_df_valid.groupby(hue_col)
            for name, group in group_data:
                if not group.empty:
                    mean_x, mean_y = group['GC3_num'].mean(), group['GC12_num'].mean()
                    group_color = palette.get(name, 'darkgrey') if palette else 'darkgrey' # Use darker color for labels
                    # Create text object with higher alpha
                    txt = ax.text(mean_x, mean_y, str(name),
                                  fontsize=8,
                                  alpha=0.9,
                                  color=group_color, ha='center', va='center',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7)) # Optional faint background
                    texts.append(txt)

            if ADJUSTTEXT_AVAILABLE and texts:
                logger.debug("Adjusting text labels using adjustText for Neutrality Plot...")
                try:
                    adjust_text(
                        texts,
                        ax=ax,
                        add_objects=[scatter_plot_object] if scatter_plot_object else [], # <-- Avoid points
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6), # <-- Arrows enabled
                        force_points=(0.6, 0.8), # <-- INCREASED point repulsion force 
                        force_text=(0.4, 0.6),   # <-- INCREASED text repulsion force 
                        expand_points=(1.3, 1.3) # <-- INCREASED padding around points 
                        # lim=500 # Optional: Increase iterations if needed
                    )
                except Exception as adj_err:
                    logger.warning(f"adjustText failed for Neutrality Plot: {adj_err}. Labels might overlap.")
            elif not ADJUSTTEXT_AVAILABLE and texts:
                 logger.info("adjustText not installed. Install for automatic label adjustment (`pip install adjusttext`). Labels placed at centroids, may overlap.")
        
        # Legend handling
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            try:
                if perform_grouping:
                    ax.legend(title=group_title, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                else:
                    ax.legend()
                    plt.tight_layout()
            except Exception as legend_err:
                logger.warning(f"Could not place Neutrality plot legend optimally: {legend_err}.")
                plt.tight_layout() # Fallback
        else:
             plt.tight_layout()

        # Saving
        filename_suffix = f"_grouped_by_{sanitize_filename(group_by)}" if perform_grouping else ""
        output_filename = os.path.join(output_dir, f"neutrality_plot{filename_suffix}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Neutrality plot saved to: {output_filename}")

    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during Neutrality plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating Neutrality plot: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_enc_vs_gc3(per_sequence_df: pd.DataFrame, 
                    output_dir: str, 
                    file_format: str = 'svg', 
                    group_by: Optional[str] = None,
                    palette: Optional[Dict[str, Any]] = None
                    ) -> None:
    """
    Generates ENC vs GC3 plot, including Wright's expected curve. Optionally colors points by group.

    Args:
        per_sequence_df (pd.DataFrame): DataFrame with per-sequence metrics (must have ENC, GC3).
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format. Default 'svg'.
        group_by (Optional[str]): Column name to group/hue by. Default None.
        palette (Optional[Dict[str, Any]]): Palette for gene colors.
    """
    required_cols = ['ENC', 'GC3']
    if per_sequence_df is None or per_sequence_df.empty:
         logger.warning("Input DataFrame empty for ENC vs GC3 plot.")
         return
    missing_cols = [col for col in required_cols if col not in per_sequence_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns ({', '.join(missing_cols)}) for ENC vs GC3 plot.")
        return
    if group_by and group_by not in per_sequence_df.columns:
        logger.warning(f"Grouping column '{group_by}' not found for ENC vs GC3 plot. Plotting without grouping.")
        group_by = None

    fig: Optional[Figure] = None
    scatter_plot_object: Optional[PathCollection] = None

    try:
        plot_df = per_sequence_df.copy()
        # Ensure columns are numeric
        plot_df['ENC_num'] = pd.to_numeric(plot_df['ENC'], errors='coerce')
        plot_df['GC3_num'] = pd.to_numeric(plot_df['GC3'], errors='coerce')
        # Use GC3 fraction for Wright's curve comparison
        plot_df['GC3_frac'] = plot_df['GC3_num'] / 100.0
        plot_df_valid = plot_df.dropna(subset=['ENC_num', 'GC3_frac'])

        if plot_df_valid.empty:
            logger.warning("No valid ENC and GC3 data (after dropping NaNs) to plot.")
            return

        # Calculate Wright's expected curve
        s_values = np.linspace(0.01, 0.99, 200) # GC3 fraction
        denominator = (s_values**2 + (1 - s_values)**2)
        expected_enc = np.full_like(s_values, np.nan)
        valid_denom = denominator > 1e-9
        expected_enc[valid_denom] = 2 + s_values[valid_denom] + (29 / denominator[valid_denom])
        expected_enc = np.where(np.isfinite(expected_enc), expected_enc, np.nan)

        # Determine grouping
        perform_grouping = group_by and not plot_df_valid[group_by].isnull().all() and plot_df_valid[group_by].nunique() > 1
        hue_col = group_by if perform_grouping else None
        group_title = group_by if perform_grouping else "Group"

        fig, ax = plt.subplots(figsize=(9, 7))

        # Plot expected curve
        ax.plot(s_values, expected_enc, color='red', linestyle='--', lw=1.5, label="Expected ENC (No Selection)")

        # Plot scatter points
        scatter_plot_object =  sns.scatterplot(
            data=plot_df_valid, x='GC3_frac', y='ENC_num', hue=hue_col,
            alpha=0.7, s=60, palette=palette, legend='full' if hue_col else False, ax=ax)

        # Customizations
        ax.set_title('ENC vs GC3 Plot', fontsize=14)
        ax.set_xlabel('GC3 Content (Fraction)', fontsize=12)
        ax.set_ylabel('Effective Number of Codons (ENC)', fontsize=12)
        ax.set_xlim(0, 1)
        min_enc = max(15, plot_df_valid['ENC_num'].min() - 2) if not plot_df_valid.empty else 15
        max_enc = min(65, plot_df_valid['ENC_num'].max() + 2) if not plot_df_valid.empty else 65
        ax.set_ylim(min_enc, max_enc)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # --- Add Adjusted Gene Labels ---
        if perform_grouping and hue_col:
            texts: List[plt.Text] = []
            group_data = plot_df_valid.groupby(hue_col)
            for name, group in group_data:
                if not group.empty:
                    mean_x = group['GC3_frac'].mean()
                    mean_y = group['ENC_num'].mean()
                    group_color = palette.get(name, 'darkgrey') if palette else 'darkgrey'
                    txt = ax.text(mean_x, mean_y, str(name),
                                  fontsize=8,
                                  alpha=0.9, # <-- Less transparent
                                  color=group_color, ha='center', va='center',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
                    texts.append(txt)

            if ADJUSTTEXT_AVAILABLE and texts:
                logger.debug("Adjusting text labels using adjustText for ENC vs GC3 Plot...")
                try:
                    adjust_text(
                        texts,
                        ax=ax,
                        add_objects=[scatter_plot_object] if scatter_plot_object else [], # <-- Avoid points
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6), # <-- Arrows enabled
                        force_points=(0.6, 0.8), # <-- INCREASED point repulsion force
                        force_text=(0.4, 0.6),   # <-- INCREASED text repulsion force
                        expand_points=(1.3, 1.3) # <-- INCREASED padding
                    )
                except Exception as adj_err: logger.warning(f"adjustText failed for ENC vs GC3 Plot: {adj_err}.")
            elif not ADJUSTTEXT_AVAILABLE and texts: logger.info("adjustText not installed. Labels may overlap.")

        # Legend handling
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            try:
                if perform_grouping:
                    ax.legend(title=group_title, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                else:
                    ax.legend() # Show legend for Wright's curve only
                    plt.tight_layout()
            except Exception as legend_err:
                 logger.warning(f"Could not place ENC vs GC3 plot legend optimally: {legend_err}.")
                 plt.tight_layout()
        else:
            plt.tight_layout()

        # Saving
        filename_suffix = f"_grouped_by_{sanitize_filename(group_by)}" if perform_grouping else ""
        output_filename = os.path.join(output_dir, f"enc_vs_gc3_plot{filename_suffix}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"ENC vs GC3 plot saved to: {output_filename}")

    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during ENC vs GC3 plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating ENC vs GC3 plot: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_ca_contribution(ca_results: PrinceCA, dimension: int, n_top: int, output_dir: str, file_format: str) -> None: # type: ignore
    """
    Generates a bar plot of the top N variables (codons) contributing
    to a specific CA dimension.

    Args:
        ca_results (PrinceCA): Fitted CA object from the 'prince' library.
        dimension (int): The CA dimension index (0 for Dim 1, 1 for Dim 2, etc.).
        n_top (int): Number of top contributing variables to display.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
    """
    if prince is None or ca_results is None or not isinstance(ca_results, prince.CA): # Runtime check against imported module
        logger.warning("No valid CA results (or prince library missing) available for contribution plot.")
        return
    if not hasattr(ca_results, 'column_contributions_'):
        logger.error("CA results object missing 'column_contributions_'. Cannot plot contribution.")
        return

    fig: Optional[Figure] = None
    try:
        # Check if requested dimension exists
        if dimension >= ca_results.column_contributions_.shape[1]:
            logger.error(f"Requested dimension {dimension} exceeds available dimensions "
                         f"({ca_results.column_contributions_.shape[1]}) in CA results.")
            return

        # Get contributions for the specified dimension (%)
        contributions = pd.to_numeric(ca_results.column_contributions_.iloc[:, dimension] * 100, errors='coerce')
        contributions.dropna(inplace=True) # Remove codons if contribution couldn't be calculated

        if contributions.empty:
             logger.warning(f"No valid contribution data found for CA dimension {dimension+1}.")
             return

        # Sort by contribution descending and select top N
        top_contributors = contributions.sort_values(ascending=False).head(n_top)

        fig, ax = plt.subplots(figsize=(8, max(5, n_top * 0.4))) # Adjust height based on N

        # Barplot (horizontal for better codon label readability)
        sns.barplot(
            x=top_contributors.values, 
            y=top_contributors.index,
            palette='viridis',
            hue=top_contributors.index,
            legend=False,
            orient='h', 
            ax=ax)
        # Removed hue=top_contributors.index and legend=False as simple palette works better

        ax.set_title(f'Top {n_top} Contributing Codons to CA Dimension {dimension+1}', fontsize=14)
        ax.set_xlabel('Contribution (%)', fontsize=12)
        ax.set_ylabel('Codon', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Add text labels for percentage values
        try:
            for i, v in enumerate(top_contributors.values):
                if pd.notna(v):
                    # Position text slightly to the right of the bar end
                    ax.text(v + contributions.max()*0.01, i, f'{v:.2f}%', color='black', va='center', fontsize=8)
        except Exception as text_err:
             logger.warning(f"Could not add text labels to CA contribution plot: {text_err}")

        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"ca_contribution_dim{dimension+1}_top{n_top}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"CA contribution plot for Dim {dimension+1} saved to: {output_filename}")

    except AttributeError as ae:
         logger.error(f"AttributeError accessing CA results for contribution plot: {ae}")
    except (ValueError, TypeError) as data_err:
         logger.error(f"Data error during CA contribution plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating CA contribution plot for Dim {dimension+1}: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_ca_variance(ca_results: PrinceCA, n_dims: int, output_dir: str, file_format: str) -> None: # type: ignore
    """
    Generates a bar plot of the variance explained by the first N CA dimensions.

    Args:
        ca_results (PrinceCA): Fitted CA object.
        n_dims (int): Maximum number of dimensions to display.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
    """
    if prince is None or ca_results is None or not isinstance(ca_results, prince.CA): # Runtime check against imported module
        logger.warning("No valid CA results (or prince library missing) available for contribution plot.")
        return
    if not hasattr(ca_results, 'eigenvalues_summary'):
        logger.error("CA results object missing 'eigenvalues_summary'. Cannot plot variance.")
        return

    fig: Optional[Figure] = None
    try:
        variance_summary = ca_results.eigenvalues_summary
        variance_col_name = '% of variance' # Expected column name
        if variance_col_name not in variance_summary.columns:
            # Try alternative common names if primary name not found
            alt_names = ['% variance', 'explained_variance_ratio']
            found = False
            for name in alt_names:
                 if name in variance_summary.columns:
                     variance_col_name = name
                     found = True
                     break
            if not found:
                logger.error(f"Could not find variance column ('{variance_col_name}' or alternatives) in eigenvalues_summary.")
                return

        # Clean and convert variance percentage column
        try:
            variance_pct_raw = variance_summary[variance_col_name]
            # Convert to string, remove '%', strip whitespace, then convert to numeric
            variance_pct = pd.to_numeric(
                variance_pct_raw.astype(str).str.replace('%', '', regex=False).str.strip(),
                errors='coerce' # Turn errors into NaN
            )
        except (KeyError, TypeError, ValueError) as conv_err:
            logger.error(f"Error accessing or converting variance column '{variance_col_name}': {conv_err}")
            return

        variance_pct.dropna(inplace=True) # Remove NaNs from conversion errors

        if variance_pct.empty:
             logger.warning("No valid numeric variance data found after cleaning/conversion.")
             return

        n_dims_actual = min(n_dims, len(variance_pct)) # Number of dims to actually plot
        if n_dims_actual < 1:
            logger.warning("No dimensions available to plot for CA variance.")
            return

        variance_to_plot = variance_pct.head(n_dims_actual)
        dims = np.arange(1, n_dims_actual + 1) # Dimension numbers (1, 2, ...)

        fig, ax = plt.subplots(figsize=(max(6, n_dims_actual * 0.7), 5))

        # Barplot
        sns.barplot(x=dims, 
                    y=variance_to_plot.values, 
                    palette='mako',
                    hue=dims,
                    legend=False,
                    ax=ax)

        ax.set_title(f'Variance Explained by First {n_dims_actual} CA Dimensions', fontsize=14)
        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_ylabel('Variance Explained (%)', fontsize=12)
        ax.set_xticks(np.arange(n_dims_actual)) # Set ticks to 0, 1, ...
        ax.set_xticklabels(dims) # Set labels to 1, 2, ...
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_ylim(0, max(variance_to_plot.max() * 1.1, 10)) # Adjust y limit
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add text labels above bars
        try:
            for i, v in enumerate(variance_to_plot.values):
                 if pd.notna(v):
                      ax.text(i, v + ax.get_ylim()[1]*0.01, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
        except Exception as text_err:
            logger.warning(f"Could not add text labels to CA variance plot: {text_err}")

        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"ca_variance_explained_top{n_dims_actual}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"CA variance explained plot saved to: {output_filename}")

    except AttributeError as ae:
        logger.error(f"AttributeError accessing CA results for variance plot: {ae}")
    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during CA variance plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating CA variance explained plot: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_ca(
    ca_results: PrinceCA, # type: ignore
    ca_input_df: pd.DataFrame,
    output_dir: str,
    file_format: str = 'svg',
    comp_x: int = 0,
    comp_y: int = 1,
    groups: Optional[pd.Series] = None,
    palette: Optional[Dict[str, Any]] = None,
    filename_suffix: str = ""
) -> None:
    """
    Generates Correspondence Analysis biplot using the prince library results.
    Optionally colors row points based on the 'groups' Series.

    Args:
        ca_results (PrinceCA): Fitted CA object from prince.
        ca_input_df (pd.DataFrame): The input data used for CA fitting (must match index of groups).
        output_dir (str): Directory to save plot.
        file_format (str): Plot file format. Default 'svg'.
        comp_x (int): Index of the CA component for the x-axis. Default 0.
        comp_y (int): Index of the CA component for the y-axis. Default 1.
        groups (Optional[pd.Series]): Series mapping ca_input_df index to group labels
                                      (e.g., gene names) for coloring row points.
                                      Index must match ca_input_df.index. Default None.
        filename_suffix (str): Suffix to add to the output filename. Default "".
        palette (Optional[Dict[str, Any]]): Palette for gene colors.
    """
    if prince is None or ca_results is None or not isinstance(ca_results, prince.CA): # Runtime check against imported module
        logger.warning("No valid CA results (or prince library missing) available for contribution plot.")
        return
    if ca_input_df is None or ca_input_df.empty:
         logger.error("CA input DataFrame needed for plotting coordinates is missing or empty.")
         return

    fig: Optional[Figure] = None
    row_scatter_object: Optional[PathCollection] = None

    try:
        # Get coordinates using the input df used for fitting
        try:
             row_coords_raw = ca_results.row_coordinates(ca_input_df)
             col_coords_raw = ca_results.column_coordinates(ca_input_df)
        except Exception as coord_err:
             logger.error(f"Error getting coordinates from CA object: {coord_err}.")
             return

        # Filter out non-finite coordinates
        coords_to_plot: List[int] = [comp_x, comp_y]
        try:
            row_coords = row_coords_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=coords_to_plot)
            col_coords = col_coords_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=coords_to_plot)
        except KeyError:
             logger.error(f"Requested CA components ({comp_x}, {comp_y}) not found in coordinates DataFrame.")
             return

        if row_coords.empty and col_coords.empty:
             logger.warning("No finite coordinates found for CA plot after filtering.")
             return

        # Get variance explained for axis labels
        x_label, y_label = f'Component {comp_x+1}', f'Component {comp_y+1}' # Defaults
        try:
            if hasattr(ca_results, 'eigenvalues_summary'):
                variance_explained = ca_results.eigenvalues_summary
                variance_col = '% of variance'
                if variance_col not in variance_explained.columns:
                     alt_names = ['% variance', 'explained_variance_ratio'] # Try alternatives
                     for name in alt_names:
                          if name in variance_explained.columns: variance_col = name; break
                if variance_col in variance_explained.columns and comp_x < len(variance_explained) and comp_y < len(variance_explained):
                    x_var_raw = variance_explained.loc[comp_x, variance_col]
                    y_var_raw = variance_explained.loc[comp_y, variance_col]
                    # Clean potential '%' signs and convert
                    x_var = float(str(x_var_raw).replace('%','').strip())
                    y_var = float(str(y_var_raw).replace('%','').strip())
                    x_label = f'Component {comp_x+1} ({x_var:.2f}%)'
                    y_label = f'Component {comp_y+1} ({y_var:.2f}%)'
                else:
                     logger.warning("Could not retrieve or format variance explained for CA plot labels.")
            else:
                 logger.warning("'eigenvalues_summary' not found in CA results. Using default axis labels.")
        except (AttributeError, KeyError, ValueError, TypeError) as fmt_err:
             logger.warning(f"Could not format variance explained for CA plot labels: {fmt_err}. Using default labels.")


        fig, ax = plt.subplots(figsize=(12, 12))

        # Check groups validity AFTER filtering row_coords
        perform_grouping = False
        hue_group_name = 'Group'
        groups_filtered = None

        if groups is not None and isinstance(groups, pd.Series) and not row_coords.empty:
            try:
                # Align groups series with the filtered row_coords index
                groups_filtered = groups.reindex(row_coords.index).dropna()
                if not groups_filtered.empty and groups_filtered.nunique() > 1:
                    perform_grouping = True
                    hue_group_name = groups_filtered.name if groups_filtered.name else 'Group'
                elif not groups_filtered.empty:
                     logger.info("Only one unique group found for CA plot points after filtering. Coloring will be uniform.")
            except Exception as group_err:
                logger.warning(f"Error processing groups for CA plot: {group_err}. Plotting without grouping.")
                perform_grouping = False

        # Plot row points (sequences/genes)
        if not row_coords.empty:
            if perform_grouping and groups_filtered is not None:
                plot_data = row_coords.copy()
                # Ensure group column name doesn't clash
                group_col_temp = '_group_for_plotting_'
                plot_data[group_col_temp] = groups_filtered
                row_scatter_object = sns.scatterplot(data=plot_data, x=comp_x, y=comp_y, hue=group_col_temp,
                                ax=ax, s=50, alpha=0.7, palette=palette, legend='full')
                # Set legend title correctly
                try: ax.legend(title=hue_group_name, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
                except: pass # Ignore legend errors sometimes caused by tight_layout interaction
            else:
                # Fallback if grouping is not performed or failed
                row_scatter_object = ax.scatter(row_coords[comp_x], row_coords[comp_y], s=50, alpha=0.7, label='Rows (Sequences/Genes)', color='blue')

        # Plot column points (codons) - Optional: Can make plots very crowded
        show_col_points = False # Set to True to display codon points
        if show_col_points and not col_coords.empty:
            ax.scatter(col_coords[comp_x], col_coords[comp_y], marker='^', s=60, alpha=0.8, c='red', label='Cols (Codons)')
            # Add labels for codons (can be very crowded)
            show_col_labels = True
            if show_col_labels:
                texts_col = []
                for i, txt in enumerate(col_coords.index):
                     texts_col.append(ax.text(col_coords.iloc[i, comp_x], col_coords.iloc[i, comp_y], txt, fontsize=7, color='darkred'))
                # Optional: adjust text overlap
                # try: from adjustText import adjust_text; adjust_text(texts_col, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                # except ImportError: pass
        
        # --- Add Adjusted Gene Labels (for row points) ---
        if perform_grouping and groups_filtered is not None:
            texts: List[plt.Text] = []
            temp_label_df = row_coords.copy()
            temp_label_df['group_label'] = groups_filtered
            group_data = temp_label_df.groupby('group_label')

            for name, group in group_data:
                if not group.empty:
                    mean_x, mean_y = group[comp_x].mean(), group[comp_y].mean()
                    group_color = palette.get(name, 'darkgrey') if palette else 'darkgrey'
                    txt = ax.text(mean_x, mean_y, str(name),
                                  fontsize=8,
                                  alpha=0.9, # <-- Less transparent
                                  color=group_color, ha='center', va='center',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
                    texts.append(txt)

            if ADJUSTTEXT_AVAILABLE and texts:
                logger.debug("Adjusting text labels using adjustText for CA Plot...")
                try:
                    adjust_text(
                        texts,
                        ax=ax,
                        add_objects=[row_scatter_object] if row_scatter_object else [], # <-- Avoid row points
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6), # <-- Arrows enabled
                        force_points=(0.6, 0.8), # <-- INCREASED point repulsion force
                        force_text=(0.5, 0.7)    # <-- INCREASED text repulsion force
                    )
                except Exception as adj_err: logger.warning(f"adjustText failed for CA Plot: {adj_err}.")
            elif not ADJUSTTEXT_AVAILABLE and texts: logger.info("adjustText not installed. Labels may overlap.")

        # Customizations
        ax.set_title(f'Correspondence Analysis Biplot (Components {comp_x+1} & {comp_y+1})', fontsize=14)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.axhline(0, color='grey', lw=0.5, linestyle='--')
        ax.axvline(0, color='grey', lw=0.5, linestyle='--')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Adjust layout
        try:
            if perform_grouping: plt.tight_layout(rect=[0, 0, 0.85, 1])
            else: plt.tight_layout()
        except Exception as layout_err:
            logger.warning(f"Error during tight_layout for CA plot: {layout_err}")


        # Saving
        safe_suffix = sanitize_filename(filename_suffix)
        if safe_suffix and not safe_suffix.startswith('_'): safe_suffix = "_" + safe_suffix
        output_filename = os.path.join(output_dir, f"ca_biplot_comp{comp_x+1}v{comp_y+1}{safe_suffix}.{file_format}")

        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"CA biplot saved to: {output_filename}")

    except (ValueError, TypeError, KeyError, AttributeError) as data_err:
         logger.error(f"Data error during CA plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating CA plot: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_usage_comparison(agg_usage_df: pd.DataFrame, reference_data: pd.DataFrame, output_dir: str, file_format: str = 'svg') -> None:
    """
    Plots observed vs reference RSCU values (Scatter plot).

    Args:
        agg_usage_df (pd.DataFrame): DataFrame with calculated aggregate usage (needs Codon, RSCU).
        reference_data (pd.DataFrame): DataFrame with reference usage (needs Codon index, RSCU column).
        output_dir (str): Directory to save plot.
        file_format (str): Plot file format. Default 'svg'.
    """
    if reference_data is None or 'RSCU' not in reference_data.columns:
        logger.warning("Cannot plot RSCU comparison: reference RSCU data not available or missing 'RSCU' column.")
        return
    if agg_usage_df is None or agg_usage_df.empty or 'RSCU' not in agg_usage_df.columns or 'Codon' not in agg_usage_df.columns:
        logger.warning("Cannot plot RSCU comparison: calculated aggregate RSCU data invalid or missing columns.")
        return

    fig: Optional[Figure] = None
    try:
        # Prepare dataframes for merging
        obs_rscu = agg_usage_df[['Codon', 'RSCU']].rename(columns={'RSCU': 'Observed_RSCU'})
        ref_rscu = reference_data[['RSCU']].rename(columns={'RSCU': 'Reference_RSCU'}) # Assumes 'Codon' is index

        # Merge observed and reference RSCU on Codon
        comp_df = pd.merge(obs_rscu, ref_rscu, left_on='Codon', right_index=True, how='inner')
        comp_df.dropna(inplace=True) # Drop codons where either RSCU is NaN

        if comp_df.empty:
            logger.warning("No common codons with valid RSCU values found for comparison plot.")
            return

        # Ensure data is numeric
        comp_df['Observed_RSCU'] = pd.to_numeric(comp_df['Observed_RSCU'], errors='coerce')
        comp_df['Reference_RSCU'] = pd.to_numeric(comp_df['Reference_RSCU'], errors='coerce')
        comp_df.dropna(inplace=True)

        if len(comp_df) < 2:
            logger.warning("Not enough comparable RSCU points (>=2) for scatter plot.")
            return

        # Calculate correlation
        correlation = comp_df['Reference_RSCU'].corr(comp_df['Observed_RSCU'])
        r_squared = correlation**2 if pd.notna(correlation) else np.nan

        # Plot scatter with regression line
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.regplot(x='Reference_RSCU', y='Observed_RSCU', data=comp_df,
                     line_kws={"color": "blue", "lw": 1},
                     scatter_kws={"alpha": 0.6, "s": 50}, ax=ax)

        ax.set_title('Observed vs Reference RSCU Comparison')
        ax.set_xlabel('Reference RSCU')
        ax.set_ylabel('Observed RSCU')
        ax.grid(True, linestyle='--', alpha=0.6)
        if pd.notna(r_squared):
             ax.text(0.05, 0.95, f'R²={r_squared:.3f}', transform=ax.transAxes, va='top',
                      bbox=dict(boxstyle='round,pad=0.3', fc='lightcyan', alpha=0.7))

        # Add diagonal line y=x
        all_vals = pd.concat([comp_df['Reference_RSCU'], comp_df['Observed_RSCU']])
        lim_min = max(0, all_vals.min() - 0.1) if not all_vals.empty else 0
        lim_max = all_vals.max() + 0.1 if not all_vals.empty else 1.0
        lims = [lim_min, lim_max]
        ax.plot(lims, lims, 'k--', alpha=0.7, lw=1, label='y=x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend()
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"rscu_comparison_scatter.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"RSCU comparison plot saved to: {output_filename}")

    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during RSCU comparison plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating RSCU comparison plot: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_relative_dinucleotide_abundance(rel_abund_df: pd.DataFrame, 
                                         output_dir: str, 
                                         file_format: str = 'svg',
                                         palette: Optional[Dict[str, Any]] = None
                                         ) -> None:
    """
    Plots the relative dinucleotide abundance (O/E ratio) per gene using a line plot.

    Args:
        rel_abund_df (pd.DataFrame): DataFrame in long format with columns
                                     'Gene', 'Dinucleotide', 'RelativeAbundance'.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format. Default 'svg'.
        palette (Optional[Dict[str, Any]]): Palette for gene colors.
    """
    required_cols = ['Gene', 'Dinucleotide', 'RelativeAbundance']
    if rel_abund_df is None or rel_abund_df.empty:
        logger.warning("No relative dinucleotide abundance data to plot.")
        return
    if not all(col in rel_abund_df.columns for col in required_cols):
        logger.error(f"Missing required columns ({', '.join(required_cols)}) for relative dinucleotide plot.")
        return

    fig: Optional[Figure] = None
    try:
        # Ensure numeric and drop NaNs (where O/E might be undefined)
        plot_data = rel_abund_df.copy()
        plot_data['RelativeAbundance'] = pd.to_numeric(plot_data['RelativeAbundance'], errors='coerce')
        plot_data.dropna(subset=['RelativeAbundance'], inplace=True)

        if plot_data.empty:
            logger.warning("No valid relative dinucleotide abundance data remaining after filtering NaN.")
            return

        # Ensure consistent order
        dinucl_order = sorted(plot_data['Dinucleotide'].unique())
        unique_genes = plot_data['Gene'].unique()
        gene_order = sorted([g for g in unique_genes if g != 'complete'])
        if 'complete' in unique_genes: gene_order.append('complete')

        # Determine figure size
        fig_width = max(10, len(dinucl_order) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # Use lineplot to connect points for each gene
        sns.lineplot(
            data=plot_data, x='Dinucleotide', y='RelativeAbundance',
            hue='Gene', style='Gene', hue_order=gene_order, style_order=gene_order,
            markers=True, markersize=7, palette=palette, legend='full', ax=ax)

        # Add horizontal line at y=1.0 (Expected ratio)
        ax.axhline(1.0, color='grey', linestyle='--', linewidth=1, label='Expected (O/E = 1)')

        ax.set_title('Relative Dinucleotide Abundance (Observed/Expected)', fontsize=14)
        ax.set_xlabel('Dinucleotide', fontsize=12)
        ax.set_ylabel('Relative Abundance (O/E Ratio)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        # Adjust legend position
        try:
            ax.legend(title='Gene', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
            plt.tight_layout(rect=[0, 0, 0.88, 1])
        except Exception as legend_err:
             logger.warning(f"Could not place relative dinuc abundance legend optimally: {legend_err}.")
             plt.tight_layout()

        output_filename = os.path.join(output_dir, f"relative_dinucleotide_abundance.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Relative dinucleotide abundance plot saved to: {output_filename}")

    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during relative dinucleotide abundance plot generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating relative dinucleotide abundance plot: {e}")
    finally:
        if fig is not None: plt.close(fig)


def plot_rscu_boxplot_per_gene(
    long_rscu_df: pd.DataFrame,
    agg_rscu_df: pd.DataFrame,
    gene_name: str,
    output_dir: str,
    file_format: str = 'svg'
) -> None:
    """
    Generates a box plot of RSCU value distributions for a specific gene,
    with codons grouped by amino acid on the x-axis. Highlights x-axis labels
    for codons with highest (red) / lowest (blue) MEAN RSCU from agg_rscu_df.

    Args:
        long_rscu_df (pd.DataFrame): Long format RSCU data (SeqID, Codon, RSCU, AminoAcid, Gene).
        agg_rscu_df (pd.DataFrame): Aggregate RSCU data for label coloring (Codon, AminoAcid, RSCU mean).
        gene_name (str): Name of the gene (for title and filename).
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format. Default 'svg'.
    """
    long_req_cols = ['Codon', 'AminoAcid', 'RSCU']
    agg_req_cols = ['Codon', 'AminoAcid', 'RSCU']

    if long_rscu_df is None or long_rscu_df.empty or not all(c in long_rscu_df.columns for c in long_req_cols):
        logger.warning(f"Skipping RSCU boxplot for '{gene_name}'. Input distribution data invalid.")
        return
    if agg_rscu_df is None or agg_rscu_df.empty or not all(c in agg_rscu_df.columns for c in agg_req_cols):
        logger.warning(f"Skipping RSCU boxplot label coloring for '{gene_name}'. Input aggregate data invalid.")
        # Proceed with boxplot but without coloring if agg data is bad
        color_ref_data = pd.DataFrame() # Use empty df for coloring logic
    else:
        color_ref_data = agg_rscu_df.copy()

    fig: Optional[Figure] = None
    try:
        # Prepare data for boxplot (long format)
        plot_data = long_rscu_df.dropna(subset=['RSCU', 'AminoAcid']).copy()
        plot_data = plot_data[plot_data['AminoAcid'] != '*'] # Exclude stops
        plot_data['RSCU'] = pd.to_numeric(plot_data['RSCU'], errors='coerce')
        plot_data.dropna(subset=['RSCU'], inplace=True)

        if plot_data.empty:
             logger.warning(f"Skipping RSCU boxplot for '{gene_name}'. No valid RSCU data for coding codons in long format.")
             return

        # Prepare aggregate data for label coloring (use color_ref_data)
        codon_colors: Dict[str, str] = {}
        if not color_ref_data.empty:
             color_ref_data = color_ref_data.dropna(subset=['RSCU', 'AminoAcid']).copy()
             color_ref_data = color_ref_data[color_ref_data['AminoAcid'] != '*']
             color_ref_data['RSCU'] = pd.to_numeric(color_ref_data['RSCU'], errors='coerce')
             color_ref_data.dropna(subset=['RSCU'], inplace=True)

             if not color_ref_data.empty:
                # Identify preferred/least preferred based on MEAN RSCU
                aa_groups_for_color = color_ref_data.groupby('AminoAcid')
                for aa, group in aa_groups_for_color:
                    if len(group) > 1 and group['RSCU'].notna().any():
                        max_rscu = group['RSCU'].max()
                        min_rscu = group['RSCU'].min()
                        if not np.isclose(max_rscu, min_rscu):
                            for _, row in group.iterrows():
                                codon, rscu_val = row['Codon'], row['RSCU']
                                if pd.notna(rscu_val):
                                     if np.isclose(rscu_val, max_rscu): codon_colors[codon] = 'red'
                                     elif np.isclose(rscu_val, min_rscu): codon_colors[codon] = 'blue'
                                     else: codon_colors[codon] = 'black'
                                else: codon_colors[codon] = 'black'
                        else: # All values same or only one value
                            for codon in group['Codon']: codon_colors[codon] = 'black'
                    else: # Single codon or no valid RSCU
                        for codon in group['Codon']: codon_colors[codon] = 'black'
             else: logger.debug(f"No valid aggregate data for label coloring for '{gene_name}'.")
        # else: logger.debug("Aggregate data was empty/invalid, skipping label coloring.")


        # Get AA codes and sort plot_data for axis order
        plot_data['AA3'] = plot_data['AminoAcid'].map(AA_1_TO_3)
        # Ensure consistent AA ordering using the predefined AA_ORDER
        plot_data['AminoAcid'] = pd.Categorical(plot_data['AminoAcid'], categories=AA_ORDER, ordered=True)
        plot_data.sort_values(by=['AminoAcid', 'Codon'], inplace=True)
        # Determine the order of codons for the x-axis based on the sorted data
        codon_order: List[str] = plot_data['Codon'].unique().tolist()

        # Calculate bounds for separator lines and AA labels based on codon_order
        aa_group_bounds: Dict[str, Dict[str, Union[float, int]]] = {}
        current_aa: Optional[str] = None
        # Create a mapping from sorted unique codons back to their amino acid
        temp_aa_map = plot_data.drop_duplicates(subset=['Codon'])[['Codon', 'AminoAcid']].set_index('Codon')['AminoAcid']
        for i, codon in enumerate(codon_order):
            aa = temp_aa_map.get(codon)
            if aa is None: continue # Should not happen if codon_order comes from plot_data
            aa = str(aa) # Ensure string key
            if aa != current_aa:
                 if current_aa is not None and current_aa in aa_group_bounds:
                     aa_group_bounds[current_aa]['end'] = i - 0.5
                     aa_group_bounds[current_aa]['mid'] = (aa_group_bounds[current_aa]['start_idx'] + i - 1) / 2
                 current_aa = aa
                 if current_aa not in aa_group_bounds: aa_group_bounds[current_aa] = {}
                 aa_group_bounds[current_aa]['start'] = i - 0.5
                 aa_group_bounds[current_aa]['start_idx'] = i
        # Final group bounds
        if current_aa is not None and current_aa in aa_group_bounds and 'start_idx' in aa_group_bounds[current_aa]:
            aa_group_bounds[current_aa]['end'] = len(codon_order) - 0.5
            aa_group_bounds[current_aa]['mid'] = (aa_group_bounds[current_aa]['start_idx'] + len(codon_order) - 1) / 2


        # --- Plotting ---
        fig, ax1 = plt.subplots(figsize=(18, 7))

        # Box plot for RSCU distributions
        sns.boxplot(
            data=plot_data, x='Codon', y='RSCU', order=codon_order, ax=ax1,
            palette="vlag", fliersize=2, linewidth=0.8, showmeans=False,
            hue='Codon', legend=False # Use hue to map colors but disable its legend
            )

        # Set primary X-axis ticks and labels (Codons)
        ax1.set_xticks(np.arange(len(codon_order)))
        ax1.set_xticklabels(codon_order, rotation=90, fontsize=8)

        # Color the tick labels using codon_colors from aggregate data
        for ticklabel, codon in zip(ax1.get_xticklabels(), codon_order):
            ticklabel.set_color(codon_colors.get(codon, 'black')) # Default to black

        ax1.set_ylabel('RSCU Value Distribution', fontsize=12)
        ax1.set_xlabel('Codon', fontsize=12)
        ax1.set_title(f'RSCU Distribution for Gene: {gene_name}', fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_xlim(-0.7, len(codon_order) - 0.3)
        ax1.set_ylim(bottom=max(0, plot_data['RSCU'].min() - 0.1)) # Adjust bottom limit slightly below min if needed
        ax1.tick_params(axis='x', which='major', pad=1)

        # Add vertical separator lines between AA groups
        valid_bounds = [b for b in aa_group_bounds.values() if 'start' in b]
        for bounds in valid_bounds:
             if bounds['start'] > -0.5: ax1.axvline(x=bounds['start'], color='grey', linestyle=':', linewidth=1.2)
        ax1.axvline(x=len(codon_order) - 0.5, color='grey', linestyle=':', linewidth=1.2) # Final line


        # Add centered AA labels using a secondary axis
        ax2: Axes = ax1.twiny() # Create secondary axis sharing y
        ax2.set_xlim(ax1.get_xlim())
        aa_ticks: List[float] = []
        aa_labels_3_letter: List[str] = []
        # Ensure AAs are added in the canonical order (AA_ORDER)
        sorted_aa_keys = sorted(aa_group_bounds.keys(), key=lambda aa: AA_ORDER.index(aa) if aa in AA_ORDER else float('inf'))
        for aa_1_letter in sorted_aa_keys:
            bounds = aa_group_bounds[aa_1_letter]
            if aa_1_letter in AA_1_TO_3 and 'mid' in bounds and pd.notna(bounds['mid']):
                aa_ticks.append(float(bounds['mid'])) # Ensure float
                aa_labels_3_letter.append(AA_1_TO_3[aa_1_letter])

        if aa_ticks and aa_labels_3_letter:
             ax2.set_xticks(aa_ticks)
             ax2.set_xticklabels(aa_labels_3_letter, fontsize=10, fontweight='bold')
        else:
             ax2.set_xticks([])
             ax2.set_xticklabels([])

        # Style the secondary axis
        ax2.tick_params(axis='x', which='both', length=0, top=False, labeltop=True)
        for spine in ax2.spines.values(): spine.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly

        # Save the plot
        safe_gene_name = sanitize_filename(gene_name)
        output_filename = os.path.join(output_dir, f"RSCU_boxplot_{safe_gene_name}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"RSCU boxplot saved to: {output_filename}")

    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during RSCU boxplot generation for '{gene_name}': {data_err}")
    except Exception as e:
        logger.exception(f"Error generating RSCU boxplot for '{gene_name}': {e}")
    finally:
        if fig is not None: plt.close(fig)


# --- plot_correlation_heatmap (already refactored in previous example, included again for completeness) ---
def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    output_dir: str,
    file_format: str,
    method: str = 'spearman'
) -> None:
    """
    Generates a heatmap of the correlation matrix for selected features.

    Args:
        df (pd.DataFrame): DataFrame containing the features.
        features (List[str]): List of column names to include in the correlation.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
        method (str): Correlation method ('spearman' or 'pearson'). Default 'spearman'.
    """
    if df is None or df.empty:
         logger.warning("Input DataFrame is empty. Cannot plot correlation heatmap.")
         return

    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
         logger.warning(f"Need at least two available features for correlation. Found: {available_features}. Skipping heatmap.")
         return

    if method not in ['spearman', 'pearson']:
        logger.warning(f"Invalid correlation method '{method}'. Using 'spearman'.")
        method = 'spearman'

    fig: Optional[Figure] = None
    try:
        # Select data and ensure numeric, drop rows with NaNs in selected columns
        corr_data = df[available_features].copy()
        for col in available_features:
             corr_data[col] = pd.to_numeric(corr_data[col], errors='coerce')
        corr_data.dropna(inplace=True)

        if len(corr_data) < 2:
             logger.warning("Not enough data rows remaining after handling NaNs for correlation heatmap.")
             return
        if corr_data.shape[1] < 2: # Check if enough columns remain
             logger.warning("Not enough valid feature columns remaining after handling NaNs for correlation heatmap.")
             return

        # Calculate correlation matrix
        corr_matrix = corr_data.corr(method=method)

        # Plot heatmap
        fig_width = max(8, len(available_features) * 0.9)
        fig_height = max(6, len(available_features) * 0.7)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            corr_matrix,
            annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, linecolor='lightgray', cbar=True,
            square=False, annot_kws={"size": 8}, ax=ax
        )
        ax.set_title(f'{method.capitalize()} Correlation Between Features', fontsize=14)
        x_fontsize = 9 if len(available_features) < 15 else 7
        y_fontsize = 9 if len(available_features) < 15 else 7
        ax.tick_params(axis='x', rotation=45, labelsize=x_fontsize)
        ax.tick_params(axis='y', rotation=0, labelsize=y_fontsize)
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"feature_correlation_heatmap_{method}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Feature correlation heatmap saved to: {output_filename}")

    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Data error during correlation heatmap generation: {data_err}")
    except Exception as e:
        logger.exception(f"Error generating correlation heatmap: {e}")
    finally:
        if fig is not None: plt.close(fig)

def plot_ca_axes_feature_correlation(
    ca_dims_df: pd.DataFrame, # DataFrame with CA_Dim1, CA_Dim2, index should be gene__sequenceID
    metrics_df: pd.DataFrame, # DataFrame with metrics, index should be gene__sequenceID
    rscu_df: pd.DataFrame,    # DataFrame with RSCU values, index should be gene__sequenceID
    output_dir: str,
    file_format: str,
    significance_threshold: float = 0.05,
    method_name: str = "Spearman",
    features_to_correlate: Optional[List[str]] = None # Allow passing specific features
    ) -> None:
    """
    Generates a heatmap showing correlations between CA axes (Dim1, Dim2)
    and other calculated features, highlighting significant correlations.

    Args:
        ca_dims_df (pd.DataFrame): DataFrame with CA dimensions (e.g., 'CA_Dim1', 'CA_Dim2')
                                   as columns and an index matching metrics_df and rscu_df.
        metrics_df (pd.DataFrame): DataFrame of per-sequence metrics. Must have an index
                                   that can be aligned with ca_dims_df and rscu_df.
                                   Expected to contain various feature columns.
        rscu_df (pd.DataFrame): DataFrame of per-sequence RSCU values (codons as columns).
                                Must have an index that can be aligned.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format (e.g., 'png').
        significance_threshold (float): P-value threshold for highlighting. Default 0.05.
        method_name (str): Name of the correlation method used (for title). Default 'Spearman'.
        features_to_correlate (Optional[List[str]]): Specific list of features from metrics_df
                                                     and rscu_df to correlate against CA axes.
                                                     If None, sensible defaults are used.
    """
    if ca_dims_df is None or ca_dims_df.empty:
        logger.warning("CA dimensions DataFrame is missing or empty. Cannot plot CA-Feature correlation.")
        return
    if metrics_df is None or metrics_df.empty:
        logger.warning("Metrics DataFrame is missing or empty. Cannot plot CA-Feature correlation.")
        return
    if rscu_df is None or rscu_df.empty:
        logger.warning("RSCU DataFrame is missing or empty. Cannot plot CA-Feature correlation.")
        return

    # --- Validate input DataFrames and their indices ---
    # Ensure indices are unique for reliable merging
    if not ca_dims_df.index.is_unique:
        logger.error("Index of CA dimensions DataFrame is not unique. Aborting correlation plot.")
        return
    if not metrics_df.index.is_unique:
        logger.error("Index of metrics DataFrame is not unique. Aborting correlation plot.")
        return
    if not rscu_df.index.is_unique:
        logger.error("Index of RSCU DataFrame is not unique. Aborting correlation plot.")
        return

    fig: Optional[Figure] = None
    try:
        # --- Merge DataFrames robustly ---
        logger.debug(f"Initial shapes: CA_dims({ca_dims_df.shape}), Metrics({metrics_df.shape}), RSCU({rscu_df.shape})")

        # Find common indices across all three DataFrames
        common_index = ca_dims_df.index.intersection(metrics_df.index).intersection(rscu_df.index)

        if common_index.empty:
            logger.error("No common indices found between CA dimensions, metrics, and RSCU DataFrames. Cannot merge for correlation plot.")
            return
        
        logger.info(f"Found {len(common_index)} common entries for merging CA dimensions, metrics, and RSCU data.")

        # Align all DataFrames to the common index before merging
        ca_dims_aligned = ca_dims_df.loc[common_index]
        metrics_aligned = metrics_df.loc[common_index]
        rscu_aligned = rscu_df.loc[common_index]

        # Concatenate horizontally (axis=1) as they now share the same index
        # This assumes no overlapping column names between metrics_aligned and rscu_aligned,
        # except for the CA_dims already being separate.
        # And CA_dims should not have common names with metrics or RSCU.
        
        # Check for column name conflicts before merging metrics_aligned and rscu_aligned
        metric_cols = set(metrics_aligned.columns)
        rscu_cols = set(rscu_aligned.columns)
        overlapping_cols_mr = metric_cols.intersection(rscu_cols)
        if overlapping_cols_mr:
            logger.warning(f"Overlapping columns found between metrics and RSCU data: {overlapping_cols_mr}. These might cause issues or be overwritten during merge.")
            # Potentially add suffixes or raise error
            # For now, we'll let pandas handle it (last one wins for overlapping non-index cols)

        merged_df_features = pd.concat([metrics_aligned, rscu_aligned], axis=1)
        
        # Now merge with CA dimensions
        # Check for column name conflicts between merged_df_features and ca_dims_aligned
        features_cols = set(merged_df_features.columns)
        ca_dim_cols = set(ca_dims_aligned.columns)
        overlapping_cols_fc = features_cols.intersection(ca_dim_cols)
        if overlapping_cols_fc:
             logger.error(f"Critical: Overlapping columns found between features and CA dimensions: {overlapping_cols_fc}. Aborting plot.")
             return


        merged_df = pd.concat([ca_dims_aligned, merged_df_features], axis=1)

        if merged_df.empty:
            logger.error("Merged DataFrame for CA-Feature correlation is empty. This should not happen if common_index was found. Aborting.")
            return
        
        logger.info(f"Successfully merged data for correlation, final shape: {merged_df.shape}")

        # --- Define features to correlate ---
        if features_to_correlate is None:
            # Default features if not provided
            default_metric_features = ['Length', 'TotalCodons', 'GC', 'GC1', 'GC2', 'GC3', 'GC12',
                                       'ENC', 'CAI', 'Fop', 'RCDI', 'ProteinLength', 'GRAVY', 'Aromaticity']
            # Filter for those actually present in the merged_df
            available_metric_features = [f for f in default_metric_features if f in merged_df.columns]
            # Get all RSCU columns (typically 3-letter uppercase codons)
            available_rscu_columns = sorted([col for col in rscu_aligned.columns if len(col) == 3 and col.isupper()])
            features_to_correlate = available_metric_features + available_rscu_columns
        else:
            # If a list is provided, filter it to ensure all features exist in the merged_df
            original_feature_count = len(features_to_correlate)
            features_to_correlate = [f for f in features_to_correlate if f in merged_df.columns]
            if len(features_to_correlate) < original_feature_count:
                logger.warning("Some requested features for correlation were not found in the merged data and were skipped.")
        
        if not features_to_correlate:
            logger.error("No valid features selected or available for CA-Feature correlation. Aborting plot.")
            return

        # CA dimension columns (assuming they are the first columns from ca_dims_aligned)
        ca_dim_column_names = list(ca_dims_aligned.columns)
        if not ca_dim_column_names:
            logger.error("No CA dimension columns found in ca_dims_df. Aborting plot.")
            return

        # --- Calculate Correlations ---
        logger.info(f"Calculating {method_name} correlations for {len(ca_dim_column_names)} CA Axes vs {len(features_to_correlate)} features...")
        all_corr_coeffs: Dict[str, Dict[str, float]] = {}
        all_p_values: Dict[str, Dict[str, float]] = {}

        if not SCIPY_AVAILABLE: # Assuming SCIPY_AVAILABLE is defined globally or passed
            logger.warning(f"Scipy not installed. Cannot calculate p-values for {method_name} correlations. Heatmap will show coefficients only.")
            # Fallback to pandas correlation for all CA dims vs all features
            corr_matrix_full = merged_df[ca_dim_column_names + features_to_correlate].corr(method=method_name.lower())
            corr_matrix_subset = corr_matrix_full.loc[ca_dim_column_names, features_to_correlate]
            pval_matrix_subset = pd.DataFrame(np.nan, index=corr_matrix_subset.index, columns=corr_matrix_subset.columns)
        else:
            for ca_dim_col in ca_dim_column_names:
                all_corr_coeffs[ca_dim_col] = {}
                all_p_values[ca_dim_col] = {}
                ca_dim_data = merged_df[ca_dim_col]
                for feature in features_to_correlate:
                    feature_data = merged_df[feature]
                    common_mask = ca_dim_data.notna() & feature_data.notna()
                    n_common = common_mask.sum()

                    if n_common < 3 or feature_data[common_mask].nunique() <= 1 or ca_dim_data[common_mask].nunique() <= 1 :
                        all_corr_coeffs[ca_dim_col][feature] = np.nan
                        all_p_values[ca_dim_col][feature] = np.nan
                        continue
                    try:
                        if method_name.lower() == 'spearman':
                            corr, pval = scipy_stats_module.spearmanr(ca_dim_data[common_mask], feature_data[common_mask])
                        elif method_name.lower() == 'pearson':
                            corr, pval = scipy_stats_module.pearsonr(ca_dim_data[common_mask], feature_data[common_mask])
                        else:
                            logger.warning(f"Unsupported correlation method '{method_name}'. Defaulting to Spearman.")
                            corr, pval = scipy_stats_module.spearmanr(ca_dim_data[common_mask], feature_data[common_mask])
                        all_corr_coeffs[ca_dim_col][feature] = corr
                        all_p_values[ca_dim_col][feature] = pval
                    except ValueError as spe_err:
                        logger.warning(f"Could not calculate {method_name} correlation for {ca_dim_col} vs '{feature}': {spe_err}")
                        all_corr_coeffs[ca_dim_col][feature] = np.nan
                        all_p_values[ca_dim_col][feature] = np.nan
            
            corr_matrix_subset = pd.DataFrame.from_dict(all_corr_coeffs, orient='index')
            pval_matrix_subset = pd.DataFrame.from_dict(all_p_values, orient='index')
            # Reorder columns to match features_to_correlate if necessary
            if not corr_matrix_subset.empty:
                corr_matrix_subset = corr_matrix_subset.reindex(columns=features_to_correlate)
            if not pval_matrix_subset.empty:
                pval_matrix_subset = pval_matrix_subset.reindex(columns=features_to_correlate)


        if corr_matrix_subset.empty:
            logger.error("Correlation matrix is empty. Cannot generate heatmap.")
            return

        # --- Plotting ---
        # Ensure figure size calculation uses corr_matrix_subset
        n_features_plot = len(corr_matrix_subset.columns)
        fig_height = max(4, len(corr_matrix_subset.index) * 0.8)
        fig_width = max(10, n_features_plot * 0.4)
        fig_width = min(fig_width, 45)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        annot_mask = pval_matrix_subset < significance_threshold
        annot_data = np.where(
            annot_mask,
            corr_matrix_subset.round(2).astype(str) + "*",
            # For non-significant, show value if you want, or ""
            # corr_matrix_subset.round(2).astype(str) # To show all values
            "" # To show only significant values
        )

        cmap = sns.diverging_palette(240, 10, s=99, l=50, as_cmap=True)

        sns.heatmap(
            corr_matrix_subset, # Use the correctly ordered subset
            annot=annot_data,
            fmt="",
            cmap=cmap,
            linewidths=.5,
            linecolor='lightgray',
            cbar=True,
            center=0,
            vmin=-1, vmax=1,
            annot_kws={"size": 7},
            cbar_kws={'label': f'{method_name} Correlation Coefficient', 'shrink': 0.7},
            ax=ax
        )

        ax.set_title(f'{method_name} Correlation: CA Axes vs Features (p < {significance_threshold} marked with *)', fontsize=14)
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("CA Axes", fontsize=12)
        
        xtick_fontsize = 8 if n_features_plot < 50 else (6 if n_features_plot < 80 else 5)
        plt.xticks(rotation=90, ha='right', fontsize=xtick_fontsize) # ha='right' for better alignment
        plt.yticks(rotation=0, fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        safe_method_name = sanitize_filename(method_name) # Assuming sanitize_filename is available
        output_filename = os.path.join(output_dir, f"ca_axes_feature_corr_{safe_method_name}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"CA Axes vs Features correlation heatmap saved to: {output_filename}")

    except Exception as e:
        logger.exception(f"Error generating CA Axes vs Features correlation heatmap: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


# --- [Optional] plot_rscu_distribution_per_gene ---
# This function might be redundant if plot_rscu_boxplot_per_gene is preferred.
# If kept, it needs the same refactoring treatment: logging, try/except, type hints, plt.close().
# def plot_rscu_distribution_per_gene(...) -> None: ...