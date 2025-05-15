# src/pycodon_analyzer/extraction.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Module for extracting gene alignments from whole genome MSAs.
Based on the logic from the original extract_genes_aln.py script.
"""
import argparse
import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)

# --- Annotation Parsing Functions (from extract_genes_aln.py) ---

def parse_genbank_location_string(location_str: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Parses GenBank feature location strings.
    Handles formats like 'start..end', 'complement(start..end)',
    and basic boundary markers ('<', '>'). It extracts the outermost
    coordinates from 'join' or 'order' statements.

    Args:
        location_str: The location string (e.g., "100..250", "complement(500..300)").

    Returns:
        A tuple (start, end, strand). Start and end are 1-based inclusive.
        Returns (None, None, strand) if parsing fails. Ensures start <= end.
    """
    strand = '+'
    original_input = location_str

    match_complement = re.match(r'complement\((.*)\)', location_str)
    if match_complement:
        strand = '-'
        location_str = match_complement.group(1)

    location_str = location_str.replace('join(', '').replace('order(', '').replace(')', '')
    coords = re.findall(r'<?(\d+)\.\.>?(\d+)', location_str)

    if not coords:
        single_point = re.search(r'<?(\d+)>?', location_str)
        if single_point:
             pos = int(single_point.group(1))
             return pos, pos, strand
        else:
             logger.warning(f"Could not parse coordinates from location string: '{original_input}'")
             return None, None, strand

    try:
        all_pos: List[int] = []
        for start_str, end_str in coords:
            all_pos.append(int(start_str))
            all_pos.append(int(end_str))

        start: Optional[int] = min(all_pos) if all_pos else None
        end: Optional[int] = max(all_pos) if all_pos else None

        if start is not None and end is not None and start > end:
            start, end = end, start # Swap

        return start, end, strand
    except ValueError:
        logger.warning(f"Error converting coordinates to numbers in location: '{original_input}'")
        return None, None, strand


def parse_annotation_fasta_for_extraction(annotation_path: Path) -> List[Dict[str, Any]]:
    """
    Reads a multi-FASTA reference gene file and extracts gene information.
    Parses headers like '>lcl|ID [gene=NAME] ... [location=LOC]' or using '[locus_tag=...]'.

    Args:
        annotation_path: Path to the annotation FASTA file.

    Returns:
        A list of dictionaries, each containing:
            'GeneName' (str), 'Start' (int, 1-based), 'End' (int, 1-based),
            'Strand' (str, '+' or '-'), 'OriginalLocationStr' (str).
        Returns an empty list if the file is not found or no valid annotations are parsed.
    """
    gene_annotations: List[Dict[str, Any]] = []
    required_fields_found = 0

    if not annotation_path.is_file():
         logger.error(f"Annotation file not found: {annotation_path}")
         raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    logger.info(f"Parsing annotations from: {annotation_path}...")
    try:
        for record in SeqIO.parse(str(annotation_path), "fasta"):
            header = record.description
            gene_name: Optional[str] = None
            location_str: Optional[str] = None

            gene_match = re.search(r'\[(?:gene|locus_tag)=([^\]]+)\]', header)
            if gene_match: gene_name = gene_match.group(1)

            location_match = re.search(r'\[location=([^\]]+)\]', header)
            if location_match: location_str = location_match.group(1)

            if gene_name and location_str:
                start, end, strand = parse_genbank_location_string(location_str)
                if start is not None and end is not None:
                    gene_annotations.append({
                        'GeneName': gene_name, 'Start': start, 'End': end,
                        'Strand': strand, 'OriginalLocationStr': location_str
                    })
                    required_fields_found += 1
                else:
                    logger.warning(f"Skipping record '{record.id}': Could not parse location '{location_str}'.")
            # else: logger.debug(f"Gene name or location tag missing for record '{record.id}'.")

    except Exception as e:
        logger.exception(f"Error parsing annotation file {annotation_path}: {e}")
        raise ValueError(f"Error parsing annotation file {annotation_path}") from e

    if required_fields_found == 0:
        logger.warning(f"No annotations with gene name/locus_tag and parsable location found in {annotation_path}.")
    else:
        logger.info(f"Found {required_fields_found} potential gene annotations.")
    return gene_annotations


# --- Coordinate Mapping Function (from extract_genes_aln.py) ---

def map_coordinates_to_alignment_for_extraction(
    genes_info_list: List[Dict[str, Any]],
    ref_aligned_record: SeqRecord
) -> List[Dict[str, Any]]:
    """
    Maps ungapped 1-based gene coordinates to 0-based indices in the aligned reference sequence.
    Args:
        genes_info_list: List of gene info dicts from parse_annotation_fasta_for_extraction.
        ref_aligned_record: The reference SeqRecord *from the alignment*.
    Returns:
        Updated list of gene info dictionaries including 'Aln_Start_0based' and 'Aln_End_Exclusive'.
    """
    if not ref_aligned_record:
        logger.error("Reference sequence record is missing, cannot map coordinates.")
        raise ValueError("Reference sequence record is missing for coordinate mapping.")

    ref_aligned_seq = str(ref_aligned_record.seq)
    mapped_genes_info: List[Dict[str, Any]] = []
    ungapped_to_aligned_map: Dict[int, int] = {}
    ungapped_pos = 0
    for i, char in enumerate(ref_aligned_seq):
        if char != '-':
            ungapped_pos += 1
            ungapped_to_aligned_map[ungapped_pos] = i
    max_ref_ungapped_len = ungapped_pos

    logger.info(f"Mapping coordinates relative to aligned reference '{ref_aligned_record.id}' (ungapped length: {max_ref_ungapped_len})...")
    skipped_outside = 0; skipped_mapping = 0

    for gene_info in genes_info_list:
        start_orig = gene_info['Start']
        end_orig = gene_info['End']
        if not (0 < start_orig <= max_ref_ungapped_len and 0 < end_orig <= max_ref_ungapped_len):
            logger.warning(f"Original coordinates {start_orig}..{end_orig} for gene '{gene_info['GeneName']}' "
                           f"fall outside reference ungapped length ({max_ref_ungapped_len}). Skipping.")
            skipped_outside += 1; continue
        try:
            aln_start_idx_0based = ungapped_to_aligned_map[start_orig]
            aln_end_idx_0based = ungapped_to_aligned_map[end_orig]
            if aln_start_idx_0based <= aln_end_idx_0based:
                 gene_info['Aln_Start_0based'] = aln_start_idx_0based
                 gene_info['Aln_End_Exclusive'] = aln_end_idx_0based + 1
                 mapped_genes_info.append(gene_info)
            else:
                 logger.warning(f"Mapped alignment indices reversed for gene '{gene_info['GeneName']}'. Skipping."); skipped_mapping += 1
        except KeyError as e:
             logger.warning(f"Could not map coordinate {e} for gene '{gene_info['GeneName']}'. Skipping."); skipped_mapping += 1
    logger.info(f"Coordinate mapping done: Mapped: {len(mapped_genes_info)}, Skipped (outside): {skipped_outside}, Skipped (mapping issue): {skipped_mapping}.")
    return mapped_genes_info

# --- Sanitization Function (can be moved to utils if preferred and imported here) ---
def _sanitize_filename_for_extraction(name: str) -> str:
    """Sanitizes a gene name to be filesystem-safe for extraction output."""
    name = re.sub(r'[\[\]()/:\\\'"]', '', name)
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-.]', '', name)
    if name.startswith('.') or name.startswith('-'): name = '_' + name
    return name if name else "unnamed_gene"


# --- Main Orchestration Function for 'extract' subcommand ---
def extract_gene_alignments_from_genome_msa(
    annotations_path: Path,
    alignment_path: Path,
    ref_id: str,
    output_dir: Path
    ) -> None:
    """
    Main logic from extract_genes_aln.py.
    Reads annotations, maps to alignment, extracts gene segments, writes to files.
    Raises FileNotFoundError or ValueError on critical errors.
    """
    logger.info(f"Starting gene extraction: Annotations='{annotations_path}', Alignment='{alignment_path}', RefID='{ref_id}'")

    # 1. Validate inputs (FileNotFoundError raised by functions below if path invalid)
    # 2. Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory for extracted genes: {output_dir}")
    except OSError as e:
        raise OSError(f"Error creating output directory '{output_dir}': {e}") from e

    # 3. Parse Annotations
    gene_annotations = parse_annotation_fasta_for_extraction(annotations_path)
    if not gene_annotations:
        raise ValueError("No valid annotations parsed from the annotation file. Cannot proceed with extraction.")

    # 4. Read Genome Alignment and Find Reference
    logger.info(f"Reading genome alignment: {alignment_path}...")
    ref_aligned_record: Optional[SeqRecord] = None
    genome_alignment_records: List[SeqRecord] = []
    try:
        temp_records = list(SeqIO.parse(str(alignment_path), "fasta"))
        if not temp_records: raise ValueError(f"No sequences found in genome alignment file: {alignment_path}")
        for record in temp_records:
            genome_alignment_records.append(record)
            if record.id == ref_id: ref_aligned_record = record
        if not ref_aligned_record:
            available_ids_sample = [rec.id for rec in genome_alignment_records[:5]]
            logger.error(f"Reference sequence ID '{ref_id}' not found in alignment. Available sample IDs: {available_ids_sample}...")
            raise ValueError(f"Reference sequence ID '{ref_id}' not found in alignment file.")
        logger.info(f"Read {len(genome_alignment_records)} genome sequences. Found reference '{ref_id}'.")
    except ValueError as e: # Catch value error from no sequences or ref_id not found
        raise e
    except Exception as e: # Catch other parsing/IO errors
        raise ValueError(f"Error reading or parsing genome alignment file {alignment_path}: {e}") from e

    # 5. Map Coordinates
    aligned_genes_info = map_coordinates_to_alignment_for_extraction(gene_annotations, ref_aligned_record)
    if not aligned_genes_info:
        raise ValueError("Could not map coordinates for any gene. Check reference ID and annotation formats.")

    # 6. Extract and Write Gene Alignments
    logger.info("Extracting and writing gene alignments...")
    genes_written = 0; genes_failed = 0; genes_skipped_extraction = 0
    for gene_info in aligned_genes_info:
        gene_name = gene_info['GeneName']
        safe_gene_name = _sanitize_filename_for_extraction(gene_name) # Use local sanitize
        output_filename = output_dir / f"gene_{safe_gene_name}.fasta"
        logger.debug(f"Processing {gene_name} -> {output_filename} ...")

        gene_specific_records: List[SeqRecord] = []
        start_aln = gene_info['Aln_Start_0based']; end_aln = gene_info['Aln_End_Exclusive']; strand = gene_info['Strand']
        if start_aln < 0 or end_aln <= start_aln:
            logger.warning(f"Invalid mapped coords for gene {gene_name}. Skipping."); genes_failed += 1; continue

        for genome_record in genome_alignment_records:
            genome_seq = genome_record.seq; genome_id = genome_record.id; genome_len = len(genome_seq)
            if end_aln > genome_len:
                logger.debug(f"Coords for gene {gene_name} exceed length of genome {genome_id}. Skipping this genome for this gene.")
                continue
            sub_sequence: Seq = genome_seq[start_aln:end_aln]
            if strand == '-':
                try: sub_sequence = sub_sequence.reverse_complement()
                except Exception as rc_err: logger.warning(f"Could not reverse complement for {genome_id} gene {gene_name}: {rc_err}. Using forward.")
            extracted_record = SeqRecord(sub_sequence, id=genome_id, description=f"gene={gene_name} | source_location={gene_info['OriginalLocationStr']}")
            gene_specific_records.append(extracted_record)

        if gene_specific_records:
            try:
                with open(output_filename, "w") as outfile: SeqIO.write(gene_specific_records, outfile, "fasta")
                genes_written += 1
            except IOError as e: logger.error(f"Error writing output file {output_filename}: {e}"); genes_failed += 1
            except Exception as e: logger.exception(f"Unexpected error writing {output_filename}: {e}"); genes_failed += 1
        else:
             genes_skipped_extraction += 1
             logger.warning(f"No sequences could be extracted for gene {gene_name}. No output file generated.")

    logger.info("--- Gene Extraction Summary ---")
    logger.info(f"  Total annotations parsed:         {len(gene_annotations)}")
    logger.info(f"  Annotations successfully mapped:  {len(aligned_genes_info)}")
    logger.info(f"  Gene alignments written:          {genes_written}")
    if genes_skipped_extraction > 0: logger.info(f"  Genes skipped (no seq extracted): {genes_skipped_extraction}")
    if genes_failed > 0: logger.info(f"  Genes failed (error):             {genes_failed}")
    if genes_written == 0 and len(aligned_genes_info) > 0: logger.warning("No gene alignment files written despite mapping. Check logs.")