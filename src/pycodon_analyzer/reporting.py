# src/pycodon_analyzer/reporting.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
This module provides functionalities for generating comprehensive HTML reports
from codon analysis results using Jinja2 templates and Pandas DataFrames.
It handles data aggregation, plot path management, and dynamic page generation.
"""

import os
import shutil
from pathlib import Path
import traceback
from typing import Dict, Any, List, Optional
import logging
import pandas as pd

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from . import utils

logger = logging.getLogger(__name__)

# Define a placeholder for plot paths if a plot is not available
PLOT_NOT_AVAILABLE_PLACEHOLDER = "plot_not_available.png" # Or None, handled in template

# Custom Jinja filter for sanitize_filename
def jinja_sanitize_filename(text: str) -> str:
    """
    Jinja2 filter to sanitize a string for use in filenames or HTML IDs.
    This function will be available as a filter in Jinja templates.

    Args:
        text (str): The input string to sanitize.

    Returns:
        str: The sanitized string.
    """
    return utils.sanitize_filename(text)

def df_to_html_custom(df: Optional[pd.DataFrame],
                      table_id: Optional[str] = None,
                      classes: Optional[List[str]]=None,
                      display_index: bool = False) -> str:
    """
    Converts a Pandas DataFrame to an HTML table with custom styling options.

    Args:
        df (Optional[pd.DataFrame]): The DataFrame to convert. If None or empty,
                                     a placeholder message is returned.
        table_id (Optional[str]): An optional HTML 'id' attribute for the table.
        classes (Optional[List[str]]): A list of CSS classes to apply to the table.
        display_index (bool): Whether to include the DataFrame index in the HTML table.

    Returns:
        str: An HTML string representing the table or an error/unavailable message.
    """
    if df is None or df.empty:
        return "<p class='unavailable'>Data table is not available or empty.</p>"
    
    table_classes = ['dataframe'] # Default class for all generated tables
    if classes:
        table_classes.extend(classes)

    # Create a copy to avoid modifying the original DataFrame
    df_display = df.copy()
    # Truncate long string values for better display in HTML tables,
    # preventing overly wide columns.
    for col in df_display.select_dtypes(include=['object', 'string']).columns:
        df_display[col] = df_display[col].apply(lambda x: (str(x)[:75] + '...') if isinstance(x, str) and len(x) > 75 else x)
    
    try:
        html_table = df_display.to_html(
            classes=table_classes,
            escape=True,  # Escape HTML characters in data
            index=display_index,
            na_rep='N/A', # Representation for NaN/None values
            table_id=table_id
        )
        return html_table
    except Exception as e:
        logger.error(f"Error converting DataFrame to HTML: {e}")
        return "<p class='unavailable'>Error displaying data table.</p>"


class HTMLReportGenerator:
    """
    Generates a comprehensive HTML report for codon analysis results.

    This class manages the data, templates, and output structure for the report,
    allowing for the inclusion of summary statistics, data tables (as HTML and CSV links),
    and various plots. It leverages Jinja2 for templating and Pandas for data handling.
    """
    def __init__(self,
                 output_dir_root: Path,
                 run_params: Dict[str, Any]):
        """
        Initializes the HTMLReportGenerator.

        Args:
            output_dir_root (Path): The root directory where the report and its assets
                                    (HTML pages, plots, data) will be generated.
            run_params (Dict[str, Any]): A dictionary containing parameters used for the
                                         analysis run, to be displayed in the report.

        Raises:
            ImportError: If Jinja2 is not installed.
        """
        if not JINJA2_AVAILABLE: # pragma: no cover
            logger.error("Jinja2 is not installed. Cannot generate HTML report. Please install with 'pip install Jinja2'")
            raise ImportError("Jinja2 is required for HTML report generation.")

        self.output_dir_root = Path(output_dir_root) # e.g., codon_analysis_results/

        # Define paths for the main report file and secondary HTML pages directory
        self.main_report_file_path = self.output_dir_root / "report.html"
        self.secondary_html_pages_dir = self.output_dir_root / "html"

        self.run_params = run_params
        # Determine the directory where Jinja2 templates are located
        self.template_dir = Path(__file__).parent / "templates"
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml']) # Automatically escape HTML/XML to prevent XSS
        )
        # Register custom Jinja2 filters
        self.env.filters['df_to_html'] = df_to_html_custom
        self.env.filters['sanitize_filename'] = jinja_sanitize_filename

        # Initialize the main data structure that will be passed to Jinja2 templates
        self.report_data: Dict[str, Any] = {
            "params": self.run_params,  # Parameters of the analysis run
            "summary_stats": {},        # General summary statistics (e.g., number of genes)
            "tables": {},               # Stores HTML strings of DataFrames and paths to CSVs
            "plot_paths": {             # Stores paths to generated plots, categorized
                "combined_plots": {},           # Plots combining data from multiple genes/analyses
                "per_gene_rscu_boxplots": {},   # RSCU box plots for individual genes (gene_name -> {plot_key: path})
                "per_gene_metadata_plots": {}   # Plots related to metadata columns (metadata_col -> gene_name -> {plot_key: path})
            },
            "metadata_info": {},        # Information about metadata used in analysis (e.g., column for coloring)
            "navigation_items": [],     # List of items for the report's navigation sidebar
            "report_main_file_name": self.main_report_file_path.name,      # "report.html"
            "secondary_pages_dirname": self.secondary_html_pages_dir.name, # "html"
            "output_dir_root_name": self.output_dir_root.name              # Name of the root output directory
        }
        # List of pages to be generated, including their template, output file, and depth
        self.pages_to_generate: List[Dict[str,str]] = []

        # Set up the report directory structure and prepare navigation items
        self._setup_report_directory()
        self._prepare_navigation()


    def _setup_report_directory(self):
        """
        Creates the main HTML report directory and subdirectories for assets.
        Ensures the output structure is ready before report generation.
        """
        try:
            # The main output directory (where report.html will reside)
            # should either exist or be creatable by a parent process.
            self.output_dir_root.mkdir(parents=True, exist_ok=True)
            
            # Create the 'html' subdirectory for secondary pages and assets like images/data.
            # This directory will contain all linked resources from the main report.
            self.secondary_html_pages_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Main report file will be at: {self.main_report_file_path}")
            logger.info(f"Secondary HTML pages and assets will be in: {self.secondary_html_pages_dir}")

        except Exception as e:
            logger.error(f"Could not create report directory structure under {self.output_dir_root}: {e}")
            raise

    def _prepare_navigation(self):
        """
        Prepares the list of navigation items for the report's sidebar and
        defines the structure of pages to be generated.
        """
        # Define static navigation items and their corresponding output pages.
        # Each item includes an 'id' for active page highlighting, a 'title' for display,
        # and a 'url' relative to the report root.
        self.report_data["navigation_items"] = [
            {"id": "summary", "title": "Summary & Overview", "url": "report.html"}, # Main landing page
            {"id": "seq_metrics", "title": "Per-Sequence Metrics", "url": "html/sequence_metrics.html"},
            {"id": "gene_agg", "title": "Per-Gene Aggregates", "url": "html/gene_aggregates.html"},
            {"id": "rscu_plots", "title": "Per-Gene RSCU Plots", "url": "html/per_gene_rscu_plots.html"},
            {"id": "stats_comp", "title": "Statistical Comparisons", "url": "html/statistical_comparisons.html"},
            {"id": "combined_ca", "title": "Combined CA Results", "url": "html/combined_ca.html"},
            {"id": "correlations", "title": "Correlation Heatmaps", "url": "html/correlations.html"},
        ]
        # Define the list of pages to be generated. Each entry specifies:
        # - 'template': The Jinja2 template file to use.
        # - 'output_file': The path where the generated HTML will be saved, relative to output_dir_root.
        # - 'page_id': A unique identifier for the page, used for navigation.
        # - 'depth': The directory depth relative to output_dir_root (0 for root, 1 for html/ subfolder).
        self.pages_to_generate = [
            {"template": "index_page_template.html", "output_file": "report.html", "page_id": "summary", "depth": 0},
            {"template": "sequence_metrics_page_template.html", "output_file": "html/sequence_metrics.html", "page_id": "seq_metrics", "depth": 1},
            {"template": "gene_aggregates_page_template.html", "output_file": "html/gene_aggregates.html", "page_id": "gene_agg", "depth": 1},
            {"template": "per_gene_rscu_page_template.html", "output_file": "html/per_gene_rscu_plots.html", "page_id": "rscu_plots", "depth": 1},
            {"template": "statistical_comparisons_page_template.html", "output_file": "html/statistical_comparisons.html", "page_id": "stats_comp", "depth": 1},
            {"template": "combined_ca_page_template.html", "output_file": "html/combined_ca.html", "page_id": "combined_ca", "depth": 1},
            {"template": "correlation_heatmaps_page_template.html", "output_file": "html/correlations.html", "page_id": "correlations", "depth": 1},
        ]
        # Initialize the navigation for metadata plots as inactive.
        # This will be dynamically updated later if metadata plots are generated.
        self._update_nav_for_metadata_plots(is_active=False)

    def set_ca_performed_status(self, was_performed: bool):
        """
        Sets a flag in the report data indicating whether Correspondence Analysis (CA)
        was performed. This flag can be used by templates for conditional rendering
        of CA-related sections.

        Args:
            was_performed (bool): True if CA was performed, False otherwise.
        """
        self.report_data["ca_performed"] = was_performed

    def _update_nav_for_metadata_plots(self, is_active: bool, metadata_col_name: Optional[str]=None):
        """
        Adds or removes the metadata plots link from the navigation and page generation list.
        This allows the report to dynamically include a section for metadata-driven plots
        only when relevant data is available.

        Args:
            is_active (bool): If True, the metadata plots link will be added/updated.
                              If False, it will be removed.
            metadata_col_name (Optional[str]): The name of the metadata column used for
                                               coloring, used in the navigation title.
                                               Required if `is_active` is True.
        """
        page_id_to_check = "meta_plots"
        
        # Remove any existing metadata plots entry to prevent duplicates
        # if this method is called multiple times (e.g., during report generation).
        self.report_data["navigation_items"] = [item for item in self.report_data["navigation_items"] if item.get("id") != page_id_to_check]
        self.pages_to_generate = [p for p in self.pages_to_generate if p.get("page_id") != page_id_to_check]

        if is_active and metadata_col_name:
            # Sanitize the column name for display in the navigation title and URL.
            sanitized_metadata_col_name = utils.sanitize_filename(metadata_col_name)
            self.report_data["navigation_items"].append(
                {"id": page_id_to_check,
                 "title": f"Plots by '{sanitized_metadata_col_name}'",
                 "url": "html/metadata_plots.html"} # URL relative to output_dir_root
            )
            self.pages_to_generate.append(
                {"template": "metadata_plots_page_template.html",
                 "output_file": "html/metadata_plots.html", # Output file path within the 'html/' subdirectory
                 "page_id": page_id_to_check,
                 "depth": 1} # Depth is 1 as it's located in the 'html/' subdirectory
            )
            logger.info(f"Added 'Plots by {sanitized_metadata_col_name}' to report navigation.")


    def add_summary_data(self, num_genes_processed: int, total_valid_sequences: int):
        """
        Adds summary statistics to the report data.

        Args:
            num_genes_processed (int): The total number of genes successfully processed.
            total_valid_sequences (int): The total number of valid sequences extracted
                                         across all processed genes.
        """
        self.report_data["summary_stats"]["num_genes_processed"] = num_genes_processed
        self.report_data["summary_stats"]["total_valid_sequences"] = total_valid_sequences

    def add_table(self, table_name: str,
                  df: Optional[pd.DataFrame],
                  table_csv_path_relative_to_outdir: Optional[str],
                  table_id: Optional[str] = None,
                  classes: Optional[List[str]]=None,
                  display_in_html: bool = True,
                  display_index: bool = False):
        """
        Adds a Pandas DataFrame to the report data.
        
        The DataFrame can be converted to an HTML table for direct display in the report,
        and/or its CSV path can be stored for linking.

        Args:
            table_name (str): A descriptive name for the table. Used for keys in report_data
                              and potentially for display.
            df (Optional[pd.DataFrame]): The DataFrame to add. If None or empty,
                                         appropriate placeholder messages are used.
            table_csv_path_relative_to_outdir (Optional[str]): The path to the CSV file
                                                                containing the table data,
                                                                relative to `output_dir_root`.
                                                                This is used for download links.
            table_id (Optional[str]): An optional HTML 'id' attribute for the generated table.
            classes (Optional[List[str]]): A list of CSS classes to apply to the HTML table.
            display_in_html (bool): If True, the DataFrame will be converted to an HTML string
                                    and stored in `report_data`. If False, a message indicating
                                    the table is not displayed will be stored, along with a CSV link.
            display_index (bool): Whether to include the DataFrame index when converting to HTML.
        """
        # Sanitize table_name for use as a dictionary key and potential filename part.
        sane_table_key = utils.sanitize_filename(table_name).lower().replace('-', '_')
        
        if table_csv_path_relative_to_outdir:
            # Store the path relative to output_dir_root, e.g., "data/table.csv".
            # This path is used to construct download links in the HTML report.
            self.report_data["tables"][f"{sane_table_key}_csv_path_from_root"] = table_csv_path_relative_to_outdir
            # Also store just the filename for display purposes in templates.
            self.report_data["tables"][f"{sane_table_key}_csv_filename"] = Path(table_csv_path_relative_to_outdir).name
            logger.info(f"Table '{table_name}' (CSV to {table_csv_path_relative_to_outdir}) added to the report context.")
        else:
            self.report_data["tables"][f"{sane_table_key}_csv_path_from_root"] = None
            self.report_data["tables"][f"{sane_table_key}_csv_filename"] = None
            logger.warning(f"No CSV path provided for table '{table_name}'.")

        if display_in_html:
            if df is not None and not df.empty:
                 # Convert DataFrame to HTML and store it for direct embedding.
                 self.report_data["tables"][f"{sane_table_key}_html"] = df_to_html_custom(df, table_id, classes, display_index=display_index)
            else:
                 self.report_data["tables"][f"{sane_table_key}_html"] = "<p class='unavailable'>Data table is not available or empty for HTML display.</p>"
        else:
            # If not displaying in HTML, provide a message and a link to the CSV if available.
            link_text = self.report_data["tables"].get(f"{sane_table_key}_csv_path_from_root", "CSV link unavailable")
            # Construct the HTML string for the link, handling cases where CSV path might be missing.
            self.report_data["tables"][f"{sane_table_key}_html"] = \
                f"<p>Table '{table_name}' is intentionally not displayed here. See CSV for full data: " \
                f"<a href='{{{{ report_data.base_path_to_root }}}}{link_text}'>{Path(link_text).name if link_text != 'CSV link unavailable' else link_text}</a></p>" \
                if table_csv_path_relative_to_outdir else \
                f"<p>Table '{table_name}' is intentionally not displayed here. CSV link unavailable.</p>"
            

    def add_plot(self, plot_key: str,
                 plot_path_relative_to_outdir: Optional[str],
                 category: str = "combined_plots",
                 plot_dict_target: Optional[Dict[str, Any]] = None
                ):
        """
        Adds a plot's relative path to the report data.
        The plot image file is expected to be copied to the report's asset directory
        by the calling function. This method only registers its path for templating.

        Args:
            plot_key (str): A unique key to identify the plot within its category.
            plot_path_relative_to_outdir (Optional[str]): The path to the plot image file,
                                                          relative to `output_dir_root`.
                                                          If None, a placeholder will be used.
            category (str): The category under which to store the plot (e.g., "combined_plots",
                            "per_gene_rscu_boxplots"). Defaults to "combined_plots".
            plot_dict_target (Optional[Dict[str, Any]]): An optional specific dictionary
                                                          within `report_data["plot_paths"]`
                                                          to target. If None, the `category`
                                                          parameter is used to determine the target.
        """
        if plot_path_relative_to_outdir:
            logger.debug(f"Adding plot '{plot_key}' to report with relative path (from output_dir_root): '{plot_path_relative_to_outdir}'")
        else:
            logger.warning(f"No plot path provided for {category} - {plot_key}. A placeholder might be displayed.")

        # Determine the target dictionary for the plot path.
        # If plot_dict_target is provided, use it directly. Otherwise, use the specified category.
        target_dict = plot_dict_target
        if target_dict is None:
            target_dict = self.report_data["plot_paths"].setdefault(category, {})
        
        # Store the plot path (can be None if plot is unavailable).
        target_dict[plot_key] = plot_path_relative_to_outdir

    def generate_report(self):
        """
        Generates all HTML report pages based on the collected data and templates.
        Iterates through `pages_to_generate`, renders each template with the
        `report_data`, and writes the output to the specified file paths.
        """
        if not JINJA2_AVAILABLE: # pragma: no cover
            logger.error("Cannot generate HTML report because Jinja2 is not available.")
            return

        logger.info("Generating HTML report...")
        
        # Update navigation for metadata plots based on whether metadata info is present.
        if self.report_data.get("metadata_info", {}).get("column_used_for_coloring"):
            self._update_nav_for_metadata_plots(True, self.report_data["metadata_info"]["column_used_for_coloring"])
        else:
            self._update_nav_for_metadata_plots(False)

        # Iterate through each page defined in pages_to_generate and render it.
        for page_info in self.pages_to_generate:
            try:
                template = self.env.get_template(page_info["template"])
                current_page_depth = page_info.get("depth", 0)
                # Calculate the relative path from the current page's directory to the report root.
                # This is crucial for correct linking of assets (CSS, JS, images) from sub-pages.
                self.report_data["base_path_to_root"] = "../" * current_page_depth
                
                # Render the template with the accumulated report data.
                html_content = template.render(
                    report_data=self.report_data,
                    navigation_items=self.report_data["navigation_items"],
                    active_page=page_info["page_id"] # Used to highlight the active page in navigation
                )
                
                # Determine the full output path for the current page and ensure its parent directory exists.
                full_output_path = self.output_dir_root / page_info["output_file"]
                full_output_path.parent.mkdir(parents=True, exist_ok=True)

                # Write the rendered HTML content to the output file.
                with open(full_output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"Generated report page: {full_output_path}")
                
            except TemplateNotFound: # pragma: no cover
                 logger.error(f"HTML template not found: {page_info['template']}. Skipping page '{page_info['output_file']}'.")
            except Exception as e: # pragma: no cover
                logger.error(f"Error generating report page {page_info['output_file']} from template {page_info['template']}: {e}")
                logger.debug(traceback.format_exc()) # Log full traceback for debugging.

        logger.info(f"HTML report generation complete. Open '{self.main_report_file_path}' to view.")