# src/pycodonanalyzer/plotting.py

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Functions for generating plots related to codon usage and sequence properties.
Uses Matplotlib and Seaborn for plotting.
"""
import re
import traceback
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
import sys

# Optional: Try importing prince for type hinting if needed
try:
    import prince # type: ignore
except ImportError:
    prince = None

sns.set_theme(style="ticks", palette="deep")

# --- AA Code Mapping ---
AA_1_TO_3 = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
    # '*' excluded if not plotted
}
# --- NEW: Inverse mapping ---
AA_3_TO_1 = {v: k for k, v in AA_1_TO_3.items()}
# Canonical order for AAs (1-letter codes)
AA_ORDER = sorted(AA_1_TO_3.keys())

# --- Utility function to sanitize filenames (if not already in utils.py) ---
def sanitize_filename(name):
    """Removes or replaces characters problematic for filenames."""
    if not isinstance(name, str): name = str(name) # Ensure string
    name = re.sub(r'[\[\]\(\)]', '', name)
    name = re.sub(r'[\s/:]', '_', name)
    name = re.sub(r'[^\w\-.]', '', name)
    if name.startswith('.') or name.startswith('-'): name = '_' + name
    return name if name else "_invalid_name_"

# === Aggregate Plots ===

def plot_rscu(rscu_df, output_dir, file_format='png', verbose=False):
    """
    Generates a bar plot of aggregate RSCU values, grouped by amino acid.

    Args:
        rscu_df (pd.DataFrame): DataFrame containing RSCU values (must have
                                'AminoAcid', 'Codon', 'RSCU' columns).
        output_dir (str): Directory to save the plot.
        file_format (str): Format to save the plot (e.g., 'png', 'svg', 'pdf').
    """
    if rscu_df is None or rscu_df.empty or 'RSCU' not in rscu_df.columns:
        print("Warning: Cannot plot RSCU. DataFrame is empty or missing 'RSCU' column.")
        return

    # Filter out NaN RSCU values and sort for consistent plotting
    rscu_df_sorted = rscu_df.dropna(subset=['RSCU', 'AminoAcid']).sort_values(by=['AminoAcid', 'Codon'])

    if rscu_df_sorted.empty:
        print("Warning: No non-NaN RSCU data available to plot.")
        return

    # Make AminoAcid categorical for potential ordering (though seaborn usually handles it)
    rscu_df_sorted['AminoAcid'] = pd.Categorical(rscu_df_sorted['AminoAcid'])


    plt.figure(figsize=(18, 7)) # Adjust figure size as needed

    # Using seaborn barplot
    try:
        barplot = sns.barplot(x='Codon', y='RSCU', hue='AminoAcid',
                              data=rscu_df_sorted, dodge=False, palette='tab20') # Use a palette with more colors
    except Exception as e:
         print(f"Error during seaborn barplot creation for RSCU: {e}", file=sys.stderr)
         plt.close() # Close figure on error
         return


    plt.title('Relative Synonymous Codon Usage (RSCU)', fontsize=16)
    plt.xlabel('Codon', fontsize=12)
    plt.ylabel('RSCU Value', fontsize=12)
    plt.xticks(rotation=90, fontsize=8) # Rotate labels for readability
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

    # Improve legend - place outside plot area
    try:
        plt.legend(title='Amino Acid', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    except Exception as e:
        print(f"Warning: Could not optimally place RSCU plot legend: {e}", file=sys.stderr)
        plt.tight_layout() # Use default tight layout as fallback


    # Save the plot
    output_filename = os.path.join(output_dir, f"rscu_barplot.{file_format}")
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose :
            print(f"RSCU plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving RSCU plot '{output_filename}': {e}", file=sys.stderr)
    finally:
        plt.close() # Close the plot figure to free memory


def plot_codon_frequency(rscu_df, output_dir, file_format='png', verbose=False):
    """
    Generates a bar plot of aggregate codon frequencies.

    Args:
        rscu_df (pd.DataFrame): DataFrame containing frequency values (must have
                                'Codon', 'Frequency' columns). Usually the same
                                DataFrame returned by calculate_rscu.
        output_dir (str): Directory to save the plot.
        file_format (str): Format to save the plot (e.g., 'png', 'svg', 'pdf').
    """
    if rscu_df is None or rscu_df.empty or 'Frequency' not in rscu_df.columns:
        print("Warning: Cannot plot Frequency. DataFrame is empty or missing 'Frequency' column.")
        return

    rscu_df_sorted = rscu_df.dropna(subset=['Frequency']).sort_values(by='Codon')

    if rscu_df_sorted.empty:
        print("Warning: No non-NaN Frequency data available to plot.")
        return

    plt.figure(figsize=(16, 6))
    try:
        sns.barplot(x='Codon', y='Frequency', data=rscu_df_sorted, color='skyblue') # Use a single color
    except Exception as e:
        print(f"Error during seaborn barplot creation for Frequency: {e}", file=sys.stderr)
        plt.close()
        return

    plt.title('Codon Frequency', fontsize=16)
    plt.xlabel('Codon', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    plt.tight_layout()

    # Save the plot
    output_filename = os.path.join(output_dir, f"codon_frequency_barplot.{file_format}")
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Codon frequency plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving frequency plot '{output_filename}': {e}", file=sys.stderr)
    finally:
        plt.close()

def plot_dinucleotide_freq(dinucl_freqs, output_dir, file_format='png', verbose=False):
    """
    Plots relative dinucleotide frequencies.

    Args:
        dinucl_freqs (dict): Dictionary mapping dinucleotides to frequencies.
        output_dir (str): Directory to save the plot.
        file_format (str): Format to save the plot.
    """
    if not dinucl_freqs:
        print("Warning: No dinucleotide frequency data to plot.")
        return

    try:
        freq_df = pd.DataFrame.from_dict(dinucl_freqs, orient='index', columns=['Frequency'])
        freq_df = freq_df.sort_index()
    except Exception as e:
         print(f"Error creating DataFrame for dinucleotide plot: {e}", file=sys.stderr)
         return

    if freq_df.empty or freq_df['Frequency'].isnull().all():
        print("Warning: Dinucleotide frequency data is empty or all NaN.")
        return

    plt.figure(figsize=(10, 5))
    try:
        sns.barplot(x=freq_df.index, y=freq_df['Frequency'], palette='coolwarm')
    except Exception as e:
         print(f"Error during seaborn barplot creation for Dinucleotides: {e}", file=sys.stderr)
         plt.close()
         return

    plt.title('Relative Dinucleotide Frequencies')
    plt.xlabel('Dinucleotide')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_filename = os.path.join(output_dir, f"dinucleotide_freq.{file_format}")
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Dinucleotide frequency plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving dinucleotide frequency plot '{output_filename}': {e}", file=sys.stderr)
    finally:
        plt.close()


# --- Function for GC means barplot ---
def plot_gc_means_barplot(per_sequence_df, output_dir, file_format='png', group_by='Gene', verbose=False):
    """
    Plots a grouped bar chart of mean GC%, GC1-3%, GC12 values per group (e.g., 'Gene').

    Args:
        per_sequence_df (pd.DataFrame): DataFrame containing per-sequence metrics, including a 'Gene' column.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
        group_by (str): Column name in per_sequence_df to group by (must be 'Gene' for this specific plot).
    """
    gc_cols = ['GC', 'GC1', 'GC2', 'GC3', 'GC12']
    if per_sequence_df is None or per_sequence_df.empty:
         print("Warning: Input DataFrame is empty. Cannot plot GC means barplot.")
         return
    if group_by not in per_sequence_df.columns:
         print(f"Warning: Grouping column '{group_by}' not found. Cannot create grouped GC means plot.", file=sys.stderr)
         return
    # Check if essential GC columns exist
    if not all(col in per_sequence_df.columns for col in gc_cols):
        print(f"Warning: Missing one or more GC columns ({gc_cols}). Cannot plot GC means.", file=sys.stderr)
        return

    try:
        # Calculate mean GC values per gene
        mean_gc_df = per_sequence_df.groupby(group_by)[gc_cols].mean().reset_index()

        if mean_gc_df.empty:
            print(f"Warning: No data available after grouping by '{group_by}' for GC means plot.")
            return

        # Melt the DataFrame for easy plotting with seaborn barplot
        mean_gc_melted = mean_gc_df.melt(id_vars=[group_by], var_name='GC_Type', value_name='Mean_GC_Content')
        mean_gc_melted.dropna(subset=['Mean_GC_Content'], inplace=True) # Drop if mean calculation resulted in NaN

        if mean_gc_melted.empty:
             print(f"Warning: No valid mean GC data remaining after melting/filtering for barplot.")
             return

        # Sort genes for consistent plot order (optional, e.g., alphabetical excluding 'complete')
        gene_order = sorted([g for g in mean_gc_melted[group_by].unique() if g != 'complete'])
        if 'complete' in mean_gc_melted[group_by].unique():
            gene_order.append('complete') # Add 'complete' at the end

        plt.figure(figsize=(max(8, len(gene_order) * 0.8), 6)) # Adjust width based on number of genes

        # Create the grouped barplot
        barplot = sns.barplot(data=mean_gc_melted, x=group_by, y='Mean_GC_Content', hue='GC_Type',
                              order=gene_order, palette='viridis') # Use order and specify palette

        plt.title(f'Mean GC Content by {group_by}', fontsize=14)
        plt.xlabel(group_by, fontsize=12)
        plt.ylabel('Mean GC Content (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9) # Rotate labels
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='GC Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout for legend

        # Construct filename
        output_filename = os.path.join(output_dir, f"gc_means_barplot_by_{group_by}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose :
            print(f"GC means barplot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating GC means barplot: {e}", file=sys.stderr)
        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
    finally:
        plt.close()


def plot_neutrality(per_sequence_df, output_dir, file_format='png', group_by=None, verbose=False):
    """
    Generates a Neutrality Plot (GC12 vs GC3).
    Optionally colors points by group (e.g., 'Gene'). Calculates overall trend.

    Args:
        per_sequence_df (pd.DataFrame): DataFrame containing per-sequence metrics.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
        group_by (str, optional): Column name to group/hue by.
    """
    required_cols = ['GC12', 'GC3']
    if per_sequence_df is None or per_sequence_df.empty:
         print("Warning: Input DataFrame is empty. Cannot plot Neutrality plot.")
         return
    if not all(col in per_sequence_df.columns for col in required_cols):
        print(f"Warning: Missing required columns ({required_cols}) for Neutrality plot.", file=sys.stderr)
        return

    # Prepare data, adding group column if needed
    plot_cols = required_cols + ([group_by] if group_by and group_by in per_sequence_df.columns else [])
    try:
        plot_df = per_sequence_df[plot_cols].dropna(subset=required_cols).copy()
    except KeyError as e:
         print(f"Error selecting columns for Neutrality plot: {e}. Check DataFrame columns.", file=sys.stderr)
         return

    if len(plot_df) < 2: # Need at least 2 points for regression/correlation
        print("Warning: Not enough valid data points (after dropping NaNs) for Neutrality Plot.")
        return

    # Check if grouping is possible
    perform_grouping = group_by and group_by in plot_df.columns and not plot_df[group_by].isnull().all()
    hue_col = group_by if perform_grouping else None

    try:
        fig, ax = plt.subplots(figsize=(8, 8)) # Use ax for axes object

        # --- Plot scatter points first to establish data range ---
        # Ensure columns are numeric for plotting and range calculation
        plot_df['GC3_num'] = pd.to_numeric(plot_df['GC3'], errors='coerce')
        plot_df['GC12_num'] = pd.to_numeric(plot_df['GC12'], errors='coerce')
        plot_df_valid = plot_df.dropna(subset=['GC3_num', 'GC12_num'])

        if plot_df_valid.empty:
             print("Warning: No valid numeric points for Neutrality plot after coercion.", file=sys.stderr)
             plt.close()
             return

        scatter = sns.scatterplot(
            data=plot_df_valid, # Use valid numeric data
            x='GC3_num',
            y='GC12_num',
            hue=hue_col,
            alpha=0.7, s=60, palette='tab10', legend='full' if hue_col else False,
            ax=ax # Plot on the created axes
            )
        # --- End Scatter Plot ---

        # --- Calculate and plot overall regression line using valid numeric data ---
        slope, intercept = np.nan, np.nan
        r_squared = np.nan
        if len(plot_df_valid) >= 2:
            try:
                 slope, intercept = np.polyfit(plot_df_valid['GC3_num'], plot_df_valid['GC12_num'], 1)
                 correlation = plot_df_valid['GC3_num'].corr(plot_df_valid['GC12_num'])
                 r_squared = correlation**2 if pd.notna(correlation) else np.nan

                 # Plot line across the data range
                 x_data_range = np.array([plot_df_valid['GC3_num'].min(), plot_df_valid['GC3_num'].max()])
                 y_vals = intercept + slope * x_data_range
                 ax.plot(x_data_range, y_vals, color="black", lw=1.5, ls='--', label=f"Overall Trend (R²={r_squared:.3f})")

            except (np.linalg.LinAlgError, ValueError, TypeError) as lin_err:
                 print(f"Warning: Could not calculate or plot overall regression line for Neutrality plot: {lin_err}", file=sys.stderr)
                 slope, r_squared = np.nan, np.nan
        # --- End Regression ---


        ax.set_title('Neutrality Plot (GC12 vs GC3)', fontsize=14)
        ax.set_xlabel('GC3 Content (%)', fontsize=12)
        ax.set_ylabel('GC12 Content (%)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        if not np.isnan(slope): # Add slope text if valid
             ax.text(0.05, 0.95, f'Overall Slope={slope:.3f}', transform=ax.transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)) # Use ax.transAxes

        # --- Adjust Axis Limits ---
        # Calculate data range
        min_gc3 = plot_df_valid['GC3_num'].min()
        max_gc3 = plot_df_valid['GC3_num'].max()
        min_gc12 = plot_df_valid['GC12_num'].min()
        max_gc12 = plot_df_valid['GC12_num'].max()

        # Calculate padding (e.g., 5% of the range, or a minimum)
        x_padding = max((max_gc3 - min_gc3) * 0.05, 2) # Min padding of 2 units
        y_padding = max((max_gc12 - min_gc12) * 0.05, 2)

        # Set limits, ensuring they don't go beyond 0-100
        x_lim_low = max(0, min_gc3 - x_padding)
        x_lim_high = min(100, max_gc3 + x_padding)
        y_lim_low = max(0, min_gc12 - y_padding)
        y_lim_high = min(100, max_gc12 + y_padding)

        ax.set_xlim(x_lim_low, x_lim_high)
        ax.set_ylim(y_lim_low, y_lim_high)
        # --- End Adjust Axis Limits ---

        # Add y=x reference line across the visible adjusted range
        diag_lims = [max(x_lim_low, y_lim_low), min(x_lim_high, y_lim_high)]
        ax.plot(diag_lims, diag_lims, 'gray', linestyle=':', alpha=0.7, lw=1, label='y=x (Reference)')


        ax.tick_params(axis='both', which='major', labelsize=10) # Use ax method

        # Handle legend placement
        handles, labels = ax.get_legend_handles_labels()
        if handles: # Check if there are any legend items
            # Place legend outside if grouping, otherwise default
            if perform_grouping:
                ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
                plt.tight_layout(rect=[0, 0, 0.85, 1])
            else:
                ax.legend()
                plt.tight_layout()
        else:
             plt.tight_layout()


        filename_suffix = f"_grouped_by_{group_by}" if perform_grouping else ""
        output_filename = os.path.join(output_dir, f"neutrality_plot{filename_suffix}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Neutrality plot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating Neutrality plot: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        plt.close()


def plot_enc_vs_gc3(per_sequence_df, output_dir, file_format='png', group_by=None, verbose=False):
    """
    Generates ENC vs GC3 plot, including Wright's expected curve.
    Optionally colors points by group.

    Args:
        per_sequence_df (pd.DataFrame): DataFrame containing per-sequence metrics.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
        group_by (str, optional): Column name to group/hue by.
    """
    required_cols = ['ENC', 'GC3']
    if per_sequence_df is None or per_sequence_df.empty:
         print("Warning: Input DataFrame is empty. Cannot plot ENC vs GC3.")
         return
    if not all(col in per_sequence_df.columns for col in required_cols):
        print(f"Warning: Missing required columns ({required_cols}) for ENC vs GC3 plot.", file=sys.stderr)
        return

    # Prepare data
    plot_cols = required_cols + ([group_by] if group_by and group_by in per_sequence_df.columns else [])
    try:
        plot_df = per_sequence_df[plot_cols].dropna(subset=required_cols).copy()
    except KeyError as e:
         print(f"Error selecting columns for ENC vs GC3 plot: {e}. Check DataFrame columns.", file=sys.stderr)
         return

    if plot_df.empty:
        print("Warning: No valid ENC and GC3 data (after dropping NaNs) to plot.")
        return
    
    # --- Ensure temporary column is created correctly ---
    if 'GC3' in plot_df.columns:
         plot_df['GC3_frac'] = plot_df['GC3'] / 100.0
    else:
         print("Warning: GC3 column missing for ENC vs GC3 plot.", file=sys.stderr)
         return # Cannot proceed without GC3

    # Calculate Wright's expected curve
    s_values = np.linspace(0.01, 0.99, 200) # GC3 fraction
    # Handle potential division by zero in formula if s is exactly 0 or 1
    denominator = (s_values**2 + (1 - s_values)**2)
    expected_enc = np.full_like(s_values, np.nan) # Initialize with NaN
    valid_denominator = denominator > 1e-9 # Avoid division by zero or tiny numbers
    expected_enc[valid_denominator] = 2 + s_values[valid_denominator] + (29 / denominator[valid_denominator])
    # Clean up potential remaining non-finite values (though unlikely now)
    expected_enc = np.where(np.isfinite(expected_enc), expected_enc, np.nan)

    # Check if grouping is possible
    perform_grouping = group_by and group_by in plot_df.columns and not plot_df[group_by].isnull().all()
    hue_col = group_by if perform_grouping else None

    try:
        plt.figure(figsize=(9, 7))
        # Plot expected curve
        plt.plot(s_values, expected_enc, color='red', linestyle='--', lw=1.5, label="Expected ENC (No Selection)")

        # --- Ensure 'data' argument is present and x, y, hue are column names ---
        scatter = sns.scatterplot(
            data=plot_df,      # Pass the DataFrame
            x='GC3_frac',      # Use the created column name
            y='ENC',           # Use column name
            hue=hue_col,       # Use column name (or None)
            alpha=0.7, s=60, palette='tab10', legend='full' if hue_col else False
        )

        plt.title('ENC vs GC3 Plot', fontsize=14)
        plt.xlabel('GC3 Content (Fraction)', fontsize=12)
        plt.ylabel('Effective Number of Codons (ENC)', fontsize=12)
        plt.xlim(0, 1)
        # Adjust Y limits based on data, but keep reasonable bounds
        min_enc = max(15, plot_df['ENC'].min() - 2 if not plot_df.empty else 15)
        max_enc = min(65, plot_df['ENC'].max() + 2 if not plot_df.empty else 65)
        plt.ylim(min_enc, max_enc)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        if hue_col:
            plt.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
        else:
            plt.legend() # Show legend for Wright's curve only
            plt.tight_layout()

        filename_suffix = f"_grouped_by_{group_by}" if perform_grouping else ""
        output_filename = os.path.join(output_dir, f"enc_vs_gc3_plot{filename_suffix}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"ENC vs GC3 plot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating ENC vs GC3 plot: {e}", file=sys.stderr)
    finally:
        plt.close()


def plot_ca_contribution(ca_results, dimension: int, n_top: int, output_dir: str, file_format: str, verbose=False):
    """
    Generates a bar plot of the top N variables (codons) contributing
    to a specific CA dimension.

    Args:
        ca_results (prince.CA): Fitted CA object.
        dimension (int): The CA dimension index (0 for Dim 1, 1 for Dim 2, etc.).
        n_top (int): Number of top contributing variables to display.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
    """
    if ca_results is None or not isinstance(ca_results, prince.CA):
        print("Warning: No valid CA results available for contribution plot.")
        return
    if not hasattr(ca_results, 'column_contributions_'):
        print("Warning: CA results object does not have 'column_contributions_'. Cannot plot contribution.")
        return
    if dimension >= ca_results.column_contributions_.shape[1]:
        print(f"Warning: Requested dimension {dimension} exceeds available dimensions "
              f"({ca_results.column_contributions_.shape[1]}) in CA results.", file=sys.stderr)
        return

    try:
        # Get contributions for the specified dimension (%)
        # Ensure contributions are numeric
        contributions = pd.to_numeric(ca_results.column_contributions_.iloc[:, dimension] * 100, errors='coerce')
        contributions = contributions.dropna() # Remove codons if contribution couldn't be calculated

        if contributions.empty:
             print(f"Warning: No valid contribution data found for CA dimension {dimension+1}.", file=sys.stderr)
             return

        # Sort by contribution descending and select top N
        top_contributors = contributions.sort_values(ascending=False).head(n_top)

        plt.figure(figsize=(8, max(5, n_top * 0.4))) # Adjust height based on N

        # Barplot (horizontal for better codon label readability)
        barplot = sns.barplot(
            x=top_contributors.values,
            y=top_contributors.index,
            hue=top_contributors.index, # Assign y-variable (codons) to hue
            palette='viridis',
            orient='h',
            legend=False # Disable the hue legend
        )
        
        plt.title(f'Top {n_top} Contributing Codons to CA Dimension {dimension+1}', fontsize=14)
        plt.xlabel('Contribution (%)', fontsize=12)
        plt.ylabel('Codon', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=9)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Add text labels for percentage values
        for i, v in enumerate(top_contributors.values):
             # Position text slightly to the right of the bar
             # Check barplot orientation details if text position is off
             if pd.notna(v):
                  ax = plt.gca()
                  ax.text(v + contributions.max()*0.01, i, f'{v:.2f}%', color='black', va='center', fontsize=8)


        output_filename = os.path.join(output_dir, f"ca_contribution_dim{dimension+1}_top{n_top}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"CA contribution plot for Dim {dimension+1} saved to: {output_filename}")

    except AttributeError:
         print("Warning: Could not access 'column_contributions_' from CA results. Cannot plot contribution.", file=sys.stderr)
    except Exception as e:
        print(f"Error generating CA contribution plot for Dim {dimension+1}: {e}", file=sys.stderr)
        # import traceback; traceback.print_exc()
    finally:
        plt.close()


def plot_ca_variance(ca_results, n_dims: int, output_dir: str, file_format: str, verbose=False):
    """
    Generates a bar plot of the variance explained by the first N CA dimensions.

    Args:
        ca_results (prince.CA): Fitted CA object.
        n_dims (int): Number of dimensions to display (e.g., 10).
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
    """
    if ca_results is None or not isinstance(ca_results, prince.CA):
        print("Warning: No valid CA results available for variance plot.")
        return
    if not hasattr(ca_results, 'eigenvalues_summary'):
        print("Warning: CA results object does not have 'eigenvalues_summary'. Cannot plot variance.")
        return

    try:
        variance_summary = ca_results.eigenvalues_summary
        if '% of variance' not in variance_summary.columns:
            print("Warning: '% of variance' column not found in eigenvalues_summary.", file=sys.stderr)
            return
        
        # Get the raw column data
        variance_pct_raw = variance_summary['% of variance']

        # Attempt to clean and convert
        try:
            # Convert to string, remove '%', strip whitespace, then convert to numeric
            variance_pct = pd.to_numeric(
                variance_pct_raw.astype(str).str.replace('%', '', regex=False).str.strip(),
                errors='coerce' # Turn errors into NaN
            )
        except Exception as e:
            print(f"Warning: Error during initial cleaning/conversion of '% of variance': {e}", file=sys.stderr)
            variance_pct = pd.Series([np.nan]) # Create empty/NaN Series on error

        variance_pct = variance_pct.dropna() # Remove NaNs resulting from conversion errors

        if variance_pct.empty:
             print(f"Warning: No valid numeric variance data found after cleaning/conversion.")
             return
        
        n_dims_actual = min(n_dims, len(variance_pct)) # Number of dims to actually plot
        variance_to_plot = variance_pct.head(n_dims_actual)

        if variance_to_plot.empty:
             print(f"Warning: No valid variance data found to plot.", file=sys.stderr)
             return

        dims = np.arange(1, n_dims_actual + 1) # Dimension numbers (1, 2, ...)

        plt.figure(figsize=(max(6, n_dims_actual * 0.7), 5))

        # --- Correction: Add hue=dims and legend=False ---
        barplot = sns.barplot(
            x=dims,
            y=variance_to_plot.values,
            palette='mako',
            hue=dims,     # Assign x variable to hue
            legend=False  # Disable hue legend
            )

        plt.title(f'Variance Explained by First {n_dims_actual} CA Dimensions', fontsize=14)
        plt.xlabel('Dimension', fontsize=12)
        plt.ylabel('Variance Explained (%)', fontsize=12)
        plt.xticks(ticks=np.arange(n_dims_actual), labels=dims, fontsize=10) # Ensure ticks match labels
        plt.yticks(fontsize=10)
        plt.ylim(0, max(variance_to_plot.max() * 1.1, 10)) # Adjust y limit slightly above max bar
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add text labels above bars
        ax = plt.gca()
        for i, v in enumerate(variance_to_plot.values):
             if pd.notna(v):
                  ax.text(i, v + ax.get_ylim()[1]*0.01, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"ca_variance_explained_top{n_dims_actual}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"CA variance explained plot saved to: {output_filename}")

    except AttributeError:
        print("Warning: Could not access 'eigenvalues_summary' or '% of variance' from CA results.", file=sys.stderr)
    except Exception as e:
        print(f"Error generating CA variance explained plot: {e}", file=sys.stderr)
        # import traceback; traceback.print_exc()
    finally:
        plt.close()


def plot_ca(ca_results, ca_input_df, output_dir, file_format='png',
            comp_x=0, comp_y=1, groups=None, filename_suffix="", verbose=False):
    """
    Generates Correspondence Analysis biplot using the prince library results.
    Optionally colors row points based on the 'groups' Series.

    Args:
        ca_results (prince.CA): Fitted CA object.
        ca_input_df (pd.DataFrame): The input data used for CA (must match index of groups).
        output_dir (str): Directory to save plot.
        file_format (str): Plot file format.
        comp_x (int): Index of the CA component for the x-axis (default 0).
        comp_y (int): Index of the CA component for the y-axis (default 1).
        groups (pd.Series, optional): Series mapping ca_input_df index to group labels
                                     (e.g., gene names) for coloring row points.
                                     Index must match ca_input_df.index.
        filename_suffix (str): Suffix to add to the output filename (e.g., "_combined_by_gene").
    """
    if ca_results is None or not isinstance(ca_results, prince.CA):
        print("Warning: No valid CA results available to plot.")
        return
    if ca_input_df is None or ca_input_df.empty:
         print("Warning: CA input DataFrame needed for plotting coordinates is missing or empty.")
         return

    try:
        # Get coordinates from prince object using the input df
        # Prince needs the same df used for fitting to get coordinates correctly
        try:
             row_coords_raw = ca_results.row_coordinates(ca_input_df)
             col_coords_raw = ca_results.column_coordinates(ca_input_df)
        except Exception as coord_err:
             print(f"Error getting coordinates from CA object: {coord_err}.", file=sys.stderr)
             return
        
        # --- Filter out non-finite coordinates ---
        coords_to_plot = [comp_x, comp_y]
        row_coords = row_coords_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=coords_to_plot)
        col_coords = col_coords_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=coords_to_plot)

        if row_coords.empty and col_coords.empty:
             print("Warning: No finite coordinates found for CA plot after filtering.", file=sys.stderr)
             return


        variance_explained = ca_results.eigenvalues_summary
        # Check if requested components exist
        if comp_x >= len(variance_explained) or comp_y >= len(variance_explained):
             print(f"Error: Requested CA components ({comp_x}, {comp_y}) are out of bounds "
                   f"(found {len(variance_explained)} components). Cannot plot.", file=sys.stderr)
             return

        try:
            # Attempt to access and convert, handle potential errors
            x_var_raw = variance_explained.loc[comp_x, '% of variance']
            y_var_raw = variance_explained.loc[comp_y, '% of variance']

            # Clean potential '%' signs and convert to float
            x_var = float(str(x_var_raw).replace('%','').strip())
            y_var = float(str(y_var_raw).replace('%','').strip())

            # Format labels only if conversion was successful
            x_label = f'Component {comp_x+1} ({x_var:.2f}%)'
            y_label = f'Component {comp_y+1} ({y_var:.2f}%)'
        except (KeyError, ValueError, TypeError) as fmt_err:
             print(f"Warning: Could not format variance explained for CA plot labels: {fmt_err}. Using default labels.", file=sys.stderr)
             # Fallback labels if formatting fails
             x_label = f'Component {comp_x+1}'
             y_label = f'Component {comp_y+1}'

        fig, ax = plt.subplots(figsize=(12, 12))

        # Check groups validity AFTER filtering row_coords
        perform_grouping = False
        hue_group_name = 'Group'
        groups_filtered = None

        if groups is not None and isinstance(groups, pd.Series):
            # Filter groups based on the index of filtered row_coords
            groups_filtered = groups.reindex(row_coords.index).dropna()
            if not groups_filtered.empty: # Check if any groups remain
                if groups_filtered.nunique() > 1: # Check if there's more than one unique group left
                    perform_grouping = True
                    hue_group_name = groups_filtered.name if groups_filtered.name else 'Group'
                else:
                     # Only one group left, treat as non-grouped for coloring/legend
                     print("Warning: Only one unique group found for CA plot points after filtering. Coloring will be uniform.", file=sys.stderr)
                     # perform_grouping remains False


        # --- Plot row points (sequences/genes) ---
        if perform_grouping:
            plot_data = row_coords.copy()
            group_col_name = '_group_' # Define temporary column name
            plot_data[group_col_name] = groups_filtered # Add filtered groups as a new column

            # Ensure hue points to the correct column name, specify palette
            sns.scatterplot(
                data=plot_data,
                x=comp_x,
                y=comp_y,
                hue=group_col_name, # Use the temporary column name added to plot_data
                ax=ax,
                s=50,
                alpha=0.7,
                palette='tab10',    # Specify consistent palette
                legend='full'
            )
            # Set legend title correctly
            ax.legend(title=hue_group_name, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        elif not row_coords.empty:
            # Fallback if grouping is not performed or not possible
            ax.scatter(row_coords[comp_x], row_coords[comp_y], s=50, alpha=0.7, label='Rows (Sequences/Genes)', color='blue') # Default to blue if no grouping


        # Plot column points (codons)
        #if not col_coords.empty:
        #    ax.scatter(col_coords[comp_x], col_coords[comp_y], marker='^', s=60, alpha=0.8, c='red', label='Cols (Codons)')

        # Add labels (optional, can be crowded)
        # ... (Add text labels carefully using filtered row_coords/col_coords indices) ...
        show_row_labels = False # Keep False generally
        show_col_labels = True
        texts = []
        if show_row_labels and not row_coords.empty:
            for i, txt in enumerate(row_coords.index): texts.append(ax.text(row_coords.iloc[i, comp_x], row_coords.iloc[i, comp_y], txt, fontsize=6))
        if show_col_labels and not col_coords.empty:
             for i, txt in enumerate(col_coords.index): texts.append(ax.text(col_coords.iloc[i, comp_x], col_coords.iloc[i, comp_y], txt, fontsize=8, color='darkred'))

        # Optional: Adjust text overlap if using adjustText library
        # try:
        #    from adjustText import adjust_text
        #    if texts: adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        # except ImportError:
        #    pass # adjustText not installed

        ax.set_title(f'Correspondence Analysis Biplot (Components {comp_x+1} & {comp_y+1})', fontsize=14)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.axhline(0, color='grey', lw=0.5, linestyle='--')
        ax.axvline(0, color='grey', lw=0.5, linestyle='--')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Adjust layout if legend is present
        if perform_grouping:
             plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make space for external legend
        else:
             plt.tight_layout()

        # Construct filename
        # Ensure filename_suffix starts with '_' if not empty
        if filename_suffix and not filename_suffix.startswith('_'):
             filename_suffix = "_" + filename_suffix
        output_filename = os.path.join(output_dir, f"ca_biplot_comp{comp_x+1}v{comp_y+1}{filename_suffix}.{file_format}")

        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"CA biplot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating CA plot: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc() # Uncomment for detailed debug info
    finally:
        plt.close() # Ensure plot is closed


# --- Comparison Plot (May need update depending on context) ---
def plot_usage_comparison(agg_usage_df, reference_data, output_dir, file_format='png', verbose=False):
    """
    Plots observed vs reference RSCU values (Scatter plot).
    Note: 'agg_usage_df' might represent combined data in the new workflow.
    Consider if this comparison is meaningful for combined data.
    """
    if reference_data is None or 'RSCU' not in reference_data.columns:
        print("Warning: Cannot plot comparison, reference RSCU data not available.")
        return
    if agg_usage_df is None or agg_usage_df.empty or 'RSCU' not in agg_usage_df.columns:
        print("Warning: Cannot plot comparison, calculated aggregate RSCU data not available.")
        return

    # Merge observed and reference RSCU
    # Ensure reference_data index is 'Codon' (as set by load_reference_usage)
    try:
         comp_df = pd.merge(
             agg_usage_df[['Codon', 'RSCU']].rename(columns={'RSCU': 'Observed_RSCU'}),
             reference_data[['RSCU']].rename(columns={'RSCU': 'Reference_RSCU'}),
             left_on='Codon',
             right_index=True, # Use index from reference_data
             how='inner' # Only compare codons present in both
         ).dropna() # Drop codons where either RSCU is NaN
    except Exception as merge_err:
        print(f"Error merging observed and reference RSCU data for plot: {merge_err}", file=sys.stderr)
        return

    if comp_df.empty:
        print("Warning: No common codons with valid RSCU values found for comparison plot.")
        return

    try:
        plt.figure(figsize=(7, 7))
        sns.regplot(x='Reference_RSCU', y='Observed_RSCU', data=comp_df,
                     line_kws={"color": "blue", "lw": 1},
                     scatter_kws={"alpha": 0.6, "s": 50})

        correlation = comp_df['Reference_RSCU'].corr(comp_df['Observed_RSCU'])
        r_squared = correlation**2

        plt.title('Observed vs Reference RSCU Comparison')
        plt.xlabel('Reference RSCU')
        plt.ylabel('Observed RSCU')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.text(0.05, 0.95, f'R²={r_squared:.3f}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightcyan', alpha=0.7))

        # Add diagonal line y=x
        all_vals = pd.concat([comp_df['Reference_RSCU'], comp_df['Observed_RSCU']])
        lim_min = max(0, all_vals.min() - 0.1)
        lim_max = all_vals.max() + 0.1
        lims = [lim_min, lim_max]
        plt.plot(lims, lims, 'k--', alpha=0.7, lw=1, label='y=x')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.legend()
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"rscu_comparison_scatter.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"RSCU comparison plot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating RSCU comparison plot: {e}", file=sys.stderr)
    finally:
        plt.close()


def plot_relative_dinucleotide_abundance(rel_abund_df, output_dir, file_format='png', verbose=False):
    """
    Plots the relative dinucleotide abundance (O/E ratio) per gene.

    Connects points for the same gene with lines.

    Args:
        rel_abund_df (pd.DataFrame): DataFrame in long format with columns
                                     'Gene', 'Dinucleotide', 'RelativeAbundance'.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
    """
    if rel_abund_df is None or rel_abund_df.empty:
        print("Warning: No relative dinucleotide abundance data to plot.")
        return
    required_cols = ['Gene', 'Dinucleotide', 'RelativeAbundance']
    if not all(col in rel_abund_df.columns for col in required_cols):
        print(f"Warning: Missing required columns ({required_cols}) for relative dinucleotide plot.", file=sys.stderr)
        return

    # Drop rows where RelativeAbundance is NaN (e.g., where Exp=0, Obs>0)
    plot_data = rel_abund_df.dropna(subset=['RelativeAbundance']).copy()

    if plot_data.empty:
        print("Warning: No valid relative dinucleotide abundance data remaining after dropping NaN.", file=sys.stderr)
        return

    # Ensure Dinucleotide order is consistent (alphabetical)
    dinucl_order = sorted(plot_data['Dinucleotide'].unique())
    # Ensure Gene order is consistent (alphabetical, 'complete' last)
    gene_order = sorted([g for g in plot_data['Gene'].unique() if g != 'complete'])
    if 'complete' in plot_data['Gene'].unique():
        gene_order.append('complete')

    # Determine figure size based on number of dinucleotides
    fig_width = max(10, len(dinucl_order) * 0.6)
    plt.figure(figsize=(fig_width, 6))

    try:
        # Use lineplot to connect points for each gene
        lineplot = sns.lineplot(
            data=plot_data,
            x='Dinucleotide',
            y='RelativeAbundance',
            hue='Gene',
            style='Gene', # Use style for different markers per gene (optional)
            hue_order=gene_order,
            style_order=gene_order,
            markers=True, # Show points
            markersize=7,
            palette='tab10', # Consistent palette
            legend='full'
        )

        # Add horizontal line at y=1.0 for reference (Expected ratio = 1)
        plt.axhline(1.0, color='grey', linestyle='--', linewidth=1, label='Expected (O/E = 1)')

        plt.title('Relative Dinucleotide Abundance (Observed/Expected)', fontsize=14)
        plt.xlabel('Dinucleotide', fontsize=12)
        plt.ylabel('Relative Abundance (O/E Ratio)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle=':', alpha=0.7)

        # Adjust legend position
        plt.legend(title='Gene', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout

        output_filename = os.path.join(output_dir, f"relative_dinucleotide_abundance.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Relative dinucleotide abundance plot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating relative dinucleotide abundance plot: {e}", file=sys.stderr)
        # import traceback; traceback.print_exc()
    finally:
        plt.close()

# --- RSCU Distribution Plot per Gene ---
def plot_rscu_distribution_per_gene(usage_df, gene_name: str, output_dir: str, file_format: str, verbose=False):
    """
    Generates a bar plot of RSCU values for a specific gene, with codons
    grouped by amino acid on the x-axis. Highlights preferred/non-preferred codons.

    Args:
        usage_df (pd.DataFrame): DataFrame with aggregate codon usage for ONE gene.
                                 Must contain 'Codon', 'AminoAcid', 'RSCU'.
        gene_name (str): Name of the gene (for title and filename).
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
    """
    required_cols = ['Codon', 'AminoAcid', 'RSCU']
    if usage_df is None or usage_df.empty:
        print(f"Warning: Skipping RSCU distribution plot for '{gene_name}'. Input data is empty.")
        return
    if not all(col in usage_df.columns for col in required_cols):
        print(f"Warning: Skipping RSCU distribution plot for '{gene_name}'. Missing required columns ({required_cols}).", file=sys.stderr)
        return

    # --- Data Preparation ---
    try:
        # Keep only coding codons with valid RSCU and AA
        plot_data = usage_df.dropna(subset=['RSCU', 'AminoAcid']).copy()
        plot_data = plot_data[plot_data['AminoAcid'] != '*'] # Exclude stops

        if plot_data.empty:
             print(f"Warning: Skipping RSCU distribution plot for '{gene_name}'. No valid RSCU data for coding codons.")
             return

        # Get 3-letter AA codes
        plot_data['AA3'] = plot_data['AminoAcid'].map(AA_1_TO_3)

        # Define canonical order for AAs and sort the DataFrame
        plot_data['AminoAcid'] = pd.Categorical(plot_data['AminoAcid'], categories=AA_ORDER, ordered=True)
        plot_data.sort_values(by=['AminoAcid', 'Codon'], inplace=True)
        plot_data.reset_index(drop=True, inplace=True) # Reset index for plotting positions

        # Identify preferred (max RSCU) and least preferred (min RSCU) codons per AA group
        codon_colors = {}
        aa_group_bounds = {} # Store start/end index and midpoint for labels/lines
        current_aa = None
        group_start_idx = 0

        for idx, row in plot_data.iterrows():
             aa = row['AminoAcid']
             # Store group bounds when AA changes
             if aa != current_aa:
                 if current_aa is not None:
                     aa_group_bounds[current_aa]['end'] = idx - 0.5 # Midpoint before this codon
                     aa_group_bounds[current_aa]['mid'] = (group_start_idx + idx -1) / 2
                 # Start new group
                 group_start_idx = idx
                 current_aa = aa
                 aa_group_bounds[current_aa] = {'start': idx - 0.5} # Midpoint before this codon

             # Find min/max within the current AA group (excluding single-codon AAs)
             aa_subset = plot_data[plot_data['AminoAcid'] == aa]
             if len(aa_subset) > 1: # Only for synonymous codons
                 max_rscu = aa_subset['RSCU'].max()
                 min_rscu = aa_subset['RSCU'].min()
                 # Handle potential ties (multiple codons with max/min RSCU) - color all that match
                 if np.isclose(row['RSCU'], max_rscu):
                     codon_colors[row['Codon']] = 'red'
                 elif np.isclose(row['RSCU'], min_rscu):
                     codon_colors[row['Codon']] = 'blue'
                 else:
                     codon_colors[row['Codon']] = 'black'
             else: # Single codon AA (Met/Trp) or error case
                 codon_colors[row['Codon']] = 'black'

        # Store bounds for the last group
        if current_aa is not None:
             aa_group_bounds[current_aa]['end'] = len(plot_data) - 0.5
             aa_group_bounds[current_aa]['mid'] = (group_start_idx + len(plot_data) - 1) / 2

    except Exception as prep_err:
        print(f"Error preparing data for RSCU distribution plot for '{gene_name}': {prep_err}", file=sys.stderr)
        return


    # --- Plotting ---
    try:
        fig, ax1 = plt.subplots(figsize=(18, 7))

        # Bar plot for RSCU values
        bar_colors = [codon_colors.get(codon, 'black') for codon in plot_data['Codon']]
        ax1.bar(plot_data.index, plot_data['RSCU'], color=bar_colors, width=0.7)

        # Set primary X-axis ticks and labels (Codons)
        ax1.set_xticks(plot_data.index)
        ax1.set_xticklabels(plot_data['Codon'], rotation=90, fontsize=8)

        # Color the tick labels
        for ticklabel, codon in zip(ax1.get_xticklabels(), plot_data['Codon']):
            ticklabel.set_color(codon_colors.get(codon, 'black'))

        ax1.set_ylabel('RSCU Value', fontsize=12)
        ax1.set_xlabel('Codon', fontsize=12)
        ax1.set_title(f'RSCU Distribution for Gene: {gene_name}', fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_xlim(-0.7, len(plot_data) - 0.3) # Adjust x limits slightly
        ax1.set_ylim(bottom=0)

        # Add vertical separator lines between AA groups
        for aa, bounds in aa_group_bounds.items():
             # Draw line at the start boundary (except for the very first one)
             if bounds['start'] > -0.5:
                  ax1.axvline(x=bounds['start'], color='grey', linestyle=':', linewidth=0.8)
        # Add a final line at the end
        ax1.axvline(x=plot_data.index.max() + 0.5, color='grey', linestyle=':', linewidth=0.8)

        # --- Add centered AA labels using a secondary axis ---
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        # Get positions and 3-letter labels (using AA keys from aa_group_bounds which are 1-letter)
        aa_ticks = []
        aa_labels_3_letter = []
        for aa_1_letter, bounds in aa_group_bounds.items():
            # Ensure the AA is one we expect and has a midpoint calculated
            if aa_1_letter in AA_1_TO_3 and 'mid' in bounds:
                aa_ticks.append(bounds['mid'])
                aa_labels_3_letter.append(AA_1_TO_3[aa_1_letter])

        if aa_ticks and aa_labels_3_letter:
            # Create pairs of (tick_position, 3_letter_label)
            ticks_labels_pairs = list(zip(aa_ticks, aa_labels_3_letter))

            # --- Sort based on the canonical 1-letter AA order ---
            try:
                 ordered_ticks_labels = sorted(
                     ticks_labels_pairs,
                     key=lambda item: AA_ORDER.index(AA_3_TO_1.get(item[1], '?'))
                     # 1. Get 3-letter label (item[1], e.g., 'Ala')
                     # 2. Find corresponding 1-letter code using AA_3_TO_1 (e.g., 'A')
                     # 3. Find the index of that 1-letter code in AA_ORDER
                 )
            except ValueError as sort_err:
                 print(f"Warning: Error sorting AA labels for plot '{gene_name}': {sort_err}. Labels might be unordered.", file=sys.stderr)
                 # Use unsorted labels as fallback
                 ordered_ticks_labels = ticks_labels_pairs
        
        if ordered_ticks_labels:
             final_ticks, final_labels = zip(*ordered_ticks_labels)
             ax2.set_xticks(final_ticks)
             ax2.set_xticklabels(final_labels, fontsize=10, fontweight='bold')
        else: # Handle case where no ticks/labels were generated
             ax2.set_xticks([])
             ax2.set_xticklabels([])


        # Style the secondary axis (remove ticks, line)
        ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=True) # Show labels on top
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout() # Adjust layout

        # Save the plot
        safe_gene_name = sanitize_filename(gene_name) # Use sanitized name for file
        output_filename = os.path.join(output_dir, f"RSCU_distribution_{safe_gene_name}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"RSCU distribution plot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating RSCU distribution plot for '{gene_name}': {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        plt.close()

# --- RSCU Boxplot per Gene with Label Highlighting ---
def plot_rscu_boxplot_per_gene(
    long_rscu_df: pd.DataFrame,   # Data for boxplots (SeqID, Codon, RSCU, AminoAcid, Gene)
    agg_rscu_df: pd.DataFrame,    # Data for label coloring (Codon, AminoAcid, RSCU - mean values)
    gene_name: str,
    output_dir: str,
    file_format: str,
    verbose=False
    ):
    """
    Generates a box plot of RSCU value distributions for a specific gene,
    with codons grouped by amino acid on the x-axis.
    Highlights x-axis labels for codons with highest (red) / lowest (blue)
    MEAN RSCU within each amino acid group.

    Args:
        long_rscu_df (pd.DataFrame): DataFrame in long format with RSCU values
                                     per sequence per codon.
        agg_rscu_df (pd.DataFrame): DataFrame with AGGREGATE codon usage (including mean RSCU)
                                    for the SAME gene. Used for label coloring.
        gene_name (str): Name of the gene (for title and filename).
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
    """
    required_long_cols = ['Codon', 'AminoAcid', 'RSCU']
    required_agg_cols = ['Codon', 'AminoAcid', 'RSCU']

    # --- Input Data Validation ---
    if long_rscu_df is None or long_rscu_df.empty:
        print(f"Warning: Skipping RSCU boxplot for '{gene_name}'. Input distribution data is empty.")
        return
    if agg_rscu_df is None or agg_rscu_df.empty:
        print(f"Warning: Skipping RSCU boxplot for '{gene_name}'. Input aggregate data for label coloring is empty.")
        return
    if not all(col in long_rscu_df.columns for col in required_long_cols):
        print(f"Warning: Skipping RSCU boxplot for '{gene_name}'. Missing required columns in long data ({required_long_cols}).", file=sys.stderr)
        return
    if not all(col in agg_rscu_df.columns for col in required_agg_cols):
        print(f"Warning: Skipping RSCU boxplot for '{gene_name}'. Missing required columns in aggregate data ({required_agg_cols}).", file=sys.stderr)
        return

    # --- Data Preparation ---
    try:
        # Prepare data for boxplot (use long format)
        plot_data = long_rscu_df.dropna(subset=['RSCU', 'AminoAcid']).copy()
        plot_data = plot_data[plot_data['AminoAcid'] != '*'] # Exclude stops
        if plot_data.empty:
             print(f"Warning: Skipping RSCU boxplot for '{gene_name}'. No valid RSCU data for coding codons in long format.")
             return

        # Prepare aggregate data for label coloring
        color_ref_data = agg_rscu_df.dropna(subset=['RSCU', 'AminoAcid']).copy()
        color_ref_data = color_ref_data[color_ref_data['AminoAcid'] != '*']
        if color_ref_data.empty:
             print(f"Warning: Skipping RSCU label coloring for '{gene_name}'. No valid aggregate RSCU data.")
             # Proceed with boxplot but without label coloring
             codon_colors = {} # Empty dict means default color
        else:
             # Ensure RSCU is numeric in aggregate data
             color_ref_data['RSCU'] = pd.to_numeric(color_ref_data['RSCU'], errors='coerce')
             color_ref_data.dropna(subset=['RSCU'], inplace=True)

             # Identify preferred/least preferred based on MEAN RSCU from aggregate data
             codon_colors = {}
             aa_groups_for_color = color_ref_data.groupby('AminoAcid')
             for aa, group in aa_groups_for_color:
                 if len(group) > 1 and group['RSCU'].notna().any(): # Synonymous codons with valid data
                     max_rscu = group['RSCU'].max()
                     min_rscu = group['RSCU'].min()
                     # Check if max and min are distinct
                     if not np.isclose(max_rscu, min_rscu):
                         for codon in group['Codon']:
                              rscu_val = group.loc[group['Codon'] == codon, 'RSCU'].iloc[0]
                              if pd.notna(rscu_val):
                                   if np.isclose(rscu_val, max_rscu):
                                       codon_colors[codon] = 'red'
                                   elif np.isclose(rscu_val, min_rscu):
                                       codon_colors[codon] = 'blue'
                                   else:
                                       codon_colors[codon] = 'black'
                              else:
                                   codon_colors[codon] = 'black' # Default if RSCU is NaN somehow
                     else: # All values same
                          for codon in group['Codon']: codon_colors[codon] = 'black'
                 else: # Single codon or no valid RSCU
                     for codon in group['Codon']: codon_colors[codon] = 'black'

        # Get 3-letter AA codes and sort plot_data for axis order
        plot_data['AA3'] = plot_data['AminoAcid'].map(AA_1_TO_3)
        plot_data['AminoAcid'] = pd.Categorical(plot_data['AminoAcid'], categories=AA_ORDER, ordered=True)
        plot_data.sort_values(by=['AminoAcid', 'Codon'], inplace=True)
        codon_order = plot_data['Codon'].unique() # Use sorted unique codons for plot order

        # Calculate bounds for separator lines and AA labels based on codon_order
        # (Logic for aa_group_bounds remains the same as in previous boxplot version)
        aa_group_bounds = {}
        current_aa = None
        temp_aa_map = plot_data.drop_duplicates(subset=['Codon'])[['Codon', 'AminoAcid']].set_index('Codon')['AminoAcid']
        for i, codon in enumerate(codon_order):
            aa = temp_aa_map.get(codon)
            if aa is None: continue
            if aa != current_aa:
                 if current_aa is not None and current_aa in aa_group_bounds:
                     aa_group_bounds[current_aa]['end'] = i - 0.5
                     aa_group_bounds[current_aa]['mid'] = (aa_group_bounds[current_aa]['start_idx'] + i - 1) / 2
                 current_aa = aa
                 if current_aa not in aa_group_bounds: aa_group_bounds[current_aa] = {}
                 aa_group_bounds[current_aa]['start'] = i - 0.5
                 aa_group_bounds[current_aa]['start_idx'] = i
        if current_aa is not None and current_aa in aa_group_bounds:
            aa_group_bounds[current_aa]['end'] = len(codon_order) - 0.5
            if 'start_idx' in aa_group_bounds[current_aa]:
                 aa_group_bounds[current_aa]['mid'] = (aa_group_bounds[current_aa]['start_idx'] + len(codon_order) - 1) / 2

    except Exception as prep_err:
        print(f"Error preparing data for RSCU boxplot for '{gene_name}': {prep_err}", file=sys.stderr)
        traceback.print_exc()
        return

    # --- Plotting ---
    try:
        fig, ax1 = plt.subplots(figsize=(18, 7))

        # Box plot for RSCU distributions
        sns.boxplot(
            data=plot_data,
            x='Codon',
            y='RSCU',
            order=codon_order, # Use the determined order
            ax=ax1,
            palette="vlag", # Choose a palette suitable for distributions
            hue='Codon',
            legend=False,
            fliersize=2,
            linewidth=0.8,
            showmeans=False # Optionally show means on boxplot: showmeans=True, meanprops={"marker":"o", ...}
            )

        # Set primary X-axis ticks and labels (Codons)
        ax1.set_xticks(np.arange(len(codon_order))) # Ensure ticks match indices
        ax1.set_xticklabels(codon_order, rotation=90, fontsize=8)

        # --- Color the tick labels using codon_colors from aggregate data ---
        for ticklabel, codon in zip(ax1.get_xticklabels(), codon_order):
            ticklabel.set_color(codon_colors.get(codon, 'black')) # Use black as default
        # ---

        ax1.set_ylabel('RSCU Value Distribution', fontsize=12) # Update label
        ax1.set_xlabel('Codon', fontsize=12)
        ax1.set_title(f'RSCU Distribution for Gene: {gene_name}', fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_xlim(-0.7, len(codon_order) - 0.3)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='x', which='major', pad=1)

        # Add vertical separator lines between AA groups
        # ... (ax1.axvline(...) logic using aa_group_bounds) ...
        valid_bounds = [b for b in aa_group_bounds.values() if 'start' in b]
        for bounds in valid_bounds:
             if bounds['start'] > -0.5: ax1.axvline(x=bounds['start'], color='grey', linestyle=':', linewidth=1.2)
        ax1.axvline(x=len(codon_order) - 0.5, color='grey', linestyle=':', linewidth=1.2)


        # Add centered AA labels using a secondary axis (same logic as before)
        # ... (ax2 = ax1.twiny(), calculate final_ticks/final_labels using aa_group_bounds, set ticks/labels, style ax2) ...
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        aa_ticks = []
        aa_labels_3_letter = []
        for aa_1_letter, bounds in aa_group_bounds.items():
            if aa_1_letter in AA_1_TO_3 and 'mid' in bounds and pd.notna(bounds['mid']):
                aa_ticks.append(bounds['mid'])
                aa_labels_3_letter.append(AA_1_TO_3[aa_1_letter])
        if aa_ticks and aa_labels_3_letter:
            ticks_labels_pairs = list(zip(aa_ticks, aa_labels_3_letter))
            try:
                 ordered_ticks_labels = sorted(ticks_labels_pairs, key=lambda item: AA_ORDER.index(AA_3_TO_1.get(item[1], '?')))
                 if ordered_ticks_labels:
                     final_ticks, final_labels = zip(*ordered_ticks_labels)
                     ax2.set_xticks(final_ticks)
                     ax2.set_xticklabels(final_labels, fontsize=10, fontweight='bold')
                 else: ax2.set_xticks([]) ; ax2.set_xticklabels([])
            except ValueError as sort_err:
                 print(f"Warning: Error sorting AA labels for plot '{gene_name}': {sort_err}. Labels unordered.", file=sys.stderr)
                 ax2.set_xticks(aa_ticks); ax2.set_xticklabels(aa_labels_3_letter, fontsize=10, fontweight='bold')
        else: ax2.set_xticks([]) ; ax2.set_xticklabels([])
        ax2.tick_params(axis='x', which='both', length=0, top=False, labeltop=True)
        for spine in ax2.spines.values(): spine.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save the plot
        safe_gene_name = sanitize_filename(gene_name)
        output_filename = os.path.join(output_dir, f"RSCU_boxplot_{safe_gene_name}.{file_format}") # Use boxplot in name
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"RSCU boxplot saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating RSCU boxplot for '{gene_name}': {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        plt.close()

# --- Correlation Heatmap Plot ---
def plot_correlation_heatmap(df: pd.DataFrame, features: list[str], output_dir: str, file_format: str,
                             method: str = 'spearman', verbose=False):
    """
    Generates a heatmap of the correlation matrix for selected features.

    Args:
        df (pd.DataFrame): DataFrame containing the features (e.g., combined per-sequence metrics).
        features (list[str]): List of column names to include in the correlation.
        output_dir (str): Directory to save the plot.
        file_format (str): Plot file format.
        method (str): Correlation method ('spearman' or 'pearson'). Default 'spearman'.
    """
    if df is None or df.empty:
         print("Warning: Input DataFrame is empty. Cannot plot correlation heatmap.")
         return
    # Filter features that actually exist in the DataFrame
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
         print("Warning: Need at least two available features to calculate correlation.", file=sys.stderr)
         return

    try:
        # Select only the features to correlate and ensure they are numeric
        corr_data = df[available_features].copy()
        for col in available_features:
             corr_data[col] = pd.to_numeric(corr_data[col], errors='coerce')
        # Drop rows with NaNs in any of the selected columns to ensure fair comparison
        corr_data.dropna(inplace=True)

        if len(corr_data) < 2:
             print("Warning: Not enough data rows remaining after handling NaNs for correlation.", file=sys.stderr)
             return

        # Calculate correlation matrix
        corr_matrix = corr_data.corr(method=method)

        # Plot heatmap
        plt.figure(figsize=(max(8, len(available_features)*0.8), max(6, len(available_features)*0.6))) # Adjust size
        sns.heatmap(
            corr_matrix,
            annot=True,       # Show correlation values
            cmap='coolwarm',  # Choose a diverging colormap
            fmt=".2f",        # Format annotations to 2 decimal places
            linewidths=.5,
            linecolor='lightgray',
            cbar=True,        # Show color bar
            square=False,     # Allow rectangle if many features
            annot_kws={"size": 8} # Adjust annotation font size if needed
        )
        plt.title(f'{method.capitalize()} Correlation Between Features', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"feature_correlation_heatmap_{method}.{file_format}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Feature correlation heatmap saved to: {output_filename}")

    except Exception as e:
        print(f"Error generating correlation heatmap: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        plt.close()