# src/pycodon_analyzer/reporting.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

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
    # This function will be available as a filter in Jinja templates
    return utils.sanitize_filename(text)

def df_to_html_custom(df: Optional[pd.DataFrame], 
                      table_id: Optional[str] = None, 
                      classes: Optional[List[str]]=None, 
                      display_index: bool = False) -> str:
    """Converts a Pandas DataFrame to an HTML table with custom styling options."""
    if df is None or df.empty:
        return "<p class='unavailable'>Data table is not available or empty.</p>"
    
    table_classes = ['dataframe'] # Default class
    if classes:
        table_classes.extend(classes)

    # Truncate long string values for better display in HTML
    df_display = df.copy()
    for col in df_display.select_dtypes(include=['object', 'string']).columns:
        df_display[col] = df_display[col].apply(lambda x: (str(x)[:75] + '...') if isinstance(x, str) and len(x) > 75 else x)
    
    try:
        html_table = df_display.to_html(
            classes=table_classes,
            escape=True,
            index=display_index,
            na_rep='N/A',
            table_id=table_id
        )
        return html_table
    except Exception as e:
        logger.error(f"Error converting DataFrame to HTML: {e}")
        return "<p class='unavailable'>Error displaying data table.</p>"


class HTMLReportGenerator:
    def __init__(self, output_dir_root: Path, run_params: Dict[str, Any]): # output_dir_root is the main analysis output
        if not JINJA2_AVAILABLE: # pragma: no cover
            logger.error("Jinja2 is not installed. Cannot generate HTML report. Please install with 'pip install Jinja2'")
            raise ImportError("Jinja2 is required for HTML report generation.")

        self.output_dir_root = Path(output_dir_root) # e.g., codon_analysis_results/

        self.main_report_file_path = self.output_dir_root / "report.html"
        self.secondary_html_pages_dir = self.output_dir_root / "html"
        self.report_img_dir = self.secondary_html_pages_dir / "images"
        self.report_data_dir = self.secondary_html_pages_dir / "data"

        self.run_params = run_params
        self.template_dir = Path(__file__).parent / "templates"
        
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        self.env.filters['df_to_html'] = df_to_html_custom
        self.env.filters['sanitize_filename'] = jinja_sanitize_filename # Register custom filter

        self.report_data: Dict[str, Any] = {
            "params": self.run_params,
            "summary_stats": {},
            "tables": {}, # Will store HTML strings of DataFrames AND paths to CSVs
            "plot_paths": {
                "combined_plots": {},
                "per_gene_rscu_boxplots": {}, # gene_name -> {plot_key: path}
                "per_gene_metadata_plots": {} # metadata_col -> gene_name -> {plot_key: path}
            },
            "metadata_info": {},
            "navigation_items": [],
            "report_main_file_name": self.main_report_file_path.name, # "report.html"
            "secondary_pages_dirname": self.secondary_html_pages_dir.name, # "html"
            # output_dir_root_name is still useful for context if needed in templates
            "output_dir_root_name": self.output_dir_root.name
        }
        self.pages_to_generate: List[Dict[str,str]] = []

        self._setup_report_directory() # Call this early
        self._prepare_navigation()


    def _setup_report_directory(self):
        """Creates the main HTML report directory and subdirectories for assets."""
        try:
            # Main output directory (where report.html will be) should already exist or be creatable by parent process
            self.output_dir_root.mkdir(parents=True, exist_ok=True)
            
            # Create the 'html' subdirectory for secondary pages and assets
            # Clean up previous 'html' dir if it exists to avoid stale files
            # if self.secondary_html_pages_dir.exists():
            #     logger.info(f"Cleaning up existing secondary report directory: {self.secondary_html_pages_dir}")
            #     shutil.rmtree(self.secondary_html_pages_dir) # Deactivated for safety

            self.secondary_html_pages_dir.mkdir(parents=True, exist_ok=True)
            self.report_img_dir.mkdir(exist_ok=True)
            self.report_data_dir.mkdir(exist_ok=True)
            logger.info(f"Main report file will be at: {self.main_report_file_path}")
            logger.info(f"Secondary HTML pages and assets will be in: {self.secondary_html_pages_dir}")

        except Exception as e:
            logger.error(f"Could not create report directory structure under {self.output_dir_root}: {e}")
            raise

    def _prepare_navigation(self):
        """Prepares the list of navigation items for the sidebar."""
        # This will be populated as we define sections
        self.report_data["navigation_items"] = [
            {"id": "summary", "title": "Summary & Overview", "url": "report.html"}, # Main page
            {"id": "seq_metrics", "title": "Per-Sequence Metrics", "url": "html/sequence_metrics.html"},
            {"id": "gene_agg", "title": "Per-Gene Aggregates", "url": "html/gene_aggregates.html"},
            {"id": "rscu_plots", "title": "Per-Gene RSCU Plots", "url": "html/per_gene_rscu_plots.html"},
            {"id": "stats_comp", "title": "Statistical Comparisons", "url": "html/statistical_comparisons.html"},
            {"id": "combined_ca", "title": "Combined CA Results", "url": "html/combined_ca.html"},
            {"id": "correlations", "title": "Correlation Heatmaps", "url": "html/correlations.html"},
        ]
        self.pages_to_generate = [
            {"template": "index_page_template.html", "output_file": "report.html", "page_id": "summary", "depth": 0}, # Main page
            {"template": "sequence_metrics_page_template.html", "output_file": "html/sequence_metrics.html", "page_id": "seq_metrics", "depth": 1},
            {"template": "gene_aggregates_page_template.html", "output_file": "html/gene_aggregates.html", "page_id": "gene_agg", "depth": 1},
            {"template": "per_gene_rscu_page_template.html", "output_file": "html/per_gene_rscu_plots.html", "page_id": "rscu_plots", "depth": 1},
            {"template": "statistical_comparisons_page_template.html", "output_file": "html/statistical_comparisons.html", "page_id": "stats_comp", "depth": 1},
            {"template": "combined_ca_page_template.html", "output_file": "html/combined_ca.html", "page_id": "combined_ca", "depth": 1},
            {"template": "correlation_heatmaps_page_template.html", "output_file": "html/correlations.html", "page_id": "correlations", "depth": 1},
        ]
        # Placeholder for dynamic addition of metadata plots link
        # This will be updated in generate_report if metadata plots are active
        self._update_nav_for_metadata_plots(is_active=False) # Initialize with no metadata plots link

    def set_ca_performed_status(self, was_performed: bool):
        """Sets a flag indicating if CA was performed, for conditional rendering in templates."""
        self.report_data["ca_performed"] = was_performed

    def _update_nav_for_metadata_plots(self, is_active: bool, metadata_col_name: Optional[str]=None):
        """Adds or removes the metadata plots link from navigation and page generation list."""
        page_id_to_check = "meta_plots"
        
        # Remove existing entry if present to avoid duplicates if called multiple times
        self.report_data["navigation_items"] = [item for item in self.report_data["navigation_items"] if item.get("id") != page_id_to_check]
        self.pages_to_generate = [p for p in self.pages_to_generate if p.get("page_id") != page_id_to_check]

        if is_active and metadata_col_name:
            sanitized_metadata_col_name = utils.sanitize_filename(metadata_col_name) # Sanitize for display and URL
            self.report_data["navigation_items"].append(
                {"id": page_id_to_check,
                 "title": f"Plots by '{sanitized_metadata_col_name}'",
                 "url": "html/metadata_plots.html"} # Relative to output_dir_root
            )
            self.pages_to_generate.append(
                {"template": "metadata_plots_page_template.html",
                 "output_file": "html/metadata_plots.html", # Placed in html/
                 "page_id": page_id_to_check,
                 "depth": 1} # Depth is 1 as it's in html/
            )
            logger.info(f"Added 'Plots by {sanitized_metadata_col_name}' to report navigation.")


    def _copy_and_get_relative_path(self, 
                                    plot_abs_path: Optional[str], 
                                    plot_category: str, 
                                    plot_name: str, 
                                    current_page_depth: int) -> Optional[str]:
        """
        Copies a plot to the report's image directory and returns its relative path
        for use in HTML. Handles cases where the plot might not exist.
        If plot_abs_path is None or file not found, returns placeholder or None.
        """
        if plot_abs_path is None or not Path(plot_abs_path).is_file():
            logger.warning(f"Plot file not found or not specified for {plot_category} - {plot_name}. Will use placeholder or skip in report.")
            # You could copy a standard "plot_not_available.png" to self.report_img_dir
            # and return "images/plot_not_available.png"
            return None # Or return "images/" + PLOT_NOT_AVAILABLE_PLACEHOLDER if you have one

        try:
            src_path = Path(plot_abs_path)
            dest_filename_in_images_dir = src_path.name
            dest_path_in_report_images = self.report_img_dir / dest_filename_in_images_dir
            
            shutil.copy(src_path, dest_path_in_report_images)
            
            if current_page_depth == 0: # Page is at output_dir_root (report.html)
                relative_path_for_html = f"html/images/{dest_filename_in_images_dir}"
            else: # Page is inside output_dir_root/html/
                relative_path_for_html = f"images/{dest_filename_in_images_dir}"
                logger.debug(f"Copied plot '{src_path.name}' to '{dest_path_in_report_images}'. Relative HTML path: '{relative_path_for_html}'")
            return relative_path_for_html
        except Exception as e: # pragma: no cover
            logger.error(f"Could not copy plot {plot_abs_path} to report directory {self.report_img_dir}: {e}")
            return None


    def add_summary_data(self, num_genes_processed: int, total_valid_sequences: int):
        self.report_data["summary_stats"]["num_genes_processed"] = num_genes_processed
        self.report_data["summary_stats"]["total_valid_sequences"] = total_valid_sequences

    def add_table(self, table_name: str, df: Optional[pd.DataFrame], 
                  table_id: Optional[str] = None, classes: Optional[List[str]]=None, 
                  display_in_html: bool = True, 
                  display_index: bool = False):
        """
        Adds a DataFrame to the report data.
        Converts to HTML if display_in_html is True.
        Always saves the CSV to the report's data directory.
        """
        # Sanitize table_name for use as a key and filename part
        sane_table_key = utils.sanitize_filename(table_name).lower().replace('-', '_')        
        
        if df is not None and not df.empty:
            logger.debug(f"Processing table '{table_name}' for HTML report data.")
            
            csv_filename = f"{sane_table_key}.csv"
            csv_path = self.report_data_dir / csv_filename
            try:
                # Save the index for CA tables if it is named (eg: 'Codon') or if it is not a simple RangeIndex
                should_save_index_csv = display_index or (df.index.name is not None) or not isinstance(df.index, pd.RangeIndex)
                df.to_csv(csv_path, 
                          index=should_save_index_csv, 
                          float_format='%.5g')
                self.report_data["tables"][f"{sane_table_key}_csv_path"] = f"data/{csv_filename}"
                logger.info(f"Saved table {table_name} to {csv_path} for the report (index saved: {should_save_index_csv}).")
            except Exception as e: # pragma: no cover
                logger.error(f"Could not save CSV for table {table_name}: {e}")
                self.report_data["tables"][f"{sane_table_key}_csv_path"] = None            
            
            if display_in_html:
                self.report_data["tables"][f"{sane_table_key}_html"] = df_to_html_custom(df, 
                                                                                         table_id, 
                                                                                         classes, 
                                                                                         display_index=display_index)
            else:
                link_path_placeholder = f"{self.secondary_html_pages_dir.name}/data/{csv_filename}" if self.report_data["tables"].get(f"{sane_table_key}_csv_filename") else "#"
                display_link = f"data/{csv_filename}" # This is for pages inside html/ directory
                
                # The actual href construction needs to be smart based on page depth
                # For now, provide the raw path and let template decide based on page_depth
                self.report_data["tables"][f"{sane_table_key}_html"] = \
                    f"<p>Table '{table_name}' is intentionally not displayed here. See CSV for full data: " \
                    f"<a href='{display_link}' data-csv-filename='{csv_filename}'>data/{csv_filename}</a></p>" \
                    if self.report_data["tables"].get(f"{sane_table_key}_csv_filename") else \
                    f"<p>Table '{table_name}' is intentionally not displayed here. CSV link unavailable.</p>"
            
        else:
            logger.warning(f"DataFrame for table '{table_name}' is None or empty. It will be marked as unavailable in the report.")
            self.report_data["tables"][f"{sane_table_key}_html"] = df_to_html_custom(None) # display_index n'est pas pertinent ici
            self.report_data["tables"][f"{sane_table_key}_csv_path"] = None

    def add_plot(self, plot_key: str, 
                 plot_abs_path: Optional[str], 
                 category: str = "combined_plots",
                 plot_dict_target: Optional[Dict[str, Any]] = None,
                 current_page_depth: int = 1 # Default to depth 1 (inside html/) for most plots
                ):
        """
        Adds a plot by copying it and storing its relative path, adjusted for page depth.
        """
        copied_filename: Optional[str] = None
        if plot_abs_path is not None and Path(plot_abs_path).is_file():
            try:
                src_path = Path(plot_abs_path)
                dest_filename = src_path.name
                # self.report_img_dir is output_root/html/images/
                dest_path_in_report_img_dir = self.report_img_dir / dest_filename
                shutil.copy(src_path, dest_path_in_report_img_dir)
                copied_filename = dest_filename
                logger.debug(f"Copied plot '{src_path.name}' to '{dest_path_in_report_img_dir}'.")
            except Exception as e: # pragma: no cover
                logger.error(f"Could not copy plot {plot_abs_path} to {self.report_img_dir}: {e}")
        else:
            logger.warning(f"Plot file not found or not specified for {category} - {plot_key}.")

        # Path to be stored in report_data, always relative to output_dir_root
        stored_path: Optional[str] = None
        if copied_filename:
            stored_path = f"{self.secondary_html_pages_dir.name}/images/{copied_filename}"
            # e.g. "html/images/myplot.png"

        target_dict = plot_dict_target
        if target_dict is None:
            target_dict = self.report_data["plot_paths"].setdefault(category, {})
        target_dict[plot_key] = stored_path

    def generate_report(self):
        if not JINJA2_AVAILABLE: # pragma: no cover
            logger.error("Cannot generate HTML report because Jinja2 is not available.")
            return

        logger.info("Generating HTML report...")
        # Directories should already be created by _setup_report_directory
        # self.secondary_html_pages_dir.mkdir(parents=True, exist_ok=True)
        # self.report_img_dir.mkdir(parents=True, exist_ok=True)
        # self.report_data_dir.mkdir(parents=True, exist_ok=True)

        if self.report_data.get("metadata_info", {}).get("column_used_for_coloring"):
            self._update_nav_for_metadata_plots(True, self.report_data["metadata_info"]["column_used_for_coloring"])
        else:
            self._update_nav_for_metadata_plots(False)

        for page_info in self.pages_to_generate:
            try:
                template = self.env.get_template(page_info["template"])
                
                # Determine base path for relative links in this page
                # This path is from the current HTML file to the output_dir_root
                current_page_depth = page_info.get("depth", 0)
                base_path_to_root = "../" * current_page_depth
                self.report_data["base_path_to_root"] = base_path_to_root
                
                # Specific base path for assets (images, data) which are under "html/"
                # If current page is at root (depth 0), path to assets is "html/"
                # If current page is in "html/" (depth 1), path to assets is "" (already in correct relative scope)
                self.report_data["base_path_to_html_assets"] = "html/" if current_page_depth == 0 else ""


                html_content = template.render(
                    report_data=self.report_data, # Contains all data including plot paths relative to root
                    navigation_items=self.report_data["navigation_items"], # URLs are relative to root
                    active_page=page_info["page_id"]
                )
                
                # Determine full output path for the HTML file
                # page_info["output_file"] is already "report.html" or "html/secondary.html"
                full_output_path = self.output_dir_root / page_info["output_file"]
                
                # Ensure parent directory exists for secondary pages
                full_output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"Generated report page: {full_output_path}")
            except TemplateNotFound: # pragma: no cover
                 logger.error(f"HTML template not found: {page_info['template']}. Skipping page '{page_info['output_file']}'.")
            except Exception as e: # pragma: no cover
                logger.error(f"Error generating report page {page_info['output_file']} from template {page_info['template']}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"HTML report generation complete. Open '{self.main_report_file_path}' to view.")