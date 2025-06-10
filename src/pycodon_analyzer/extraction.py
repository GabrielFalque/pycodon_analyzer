# src/pycodon_analyzer/extraction.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Module for extracting gene alignments from whole genome multiple sequence alignments (MSAs).

This module provides functionality to extract individual gene alignments from a whole
genome multiple sequence alignment based on gene annotations. It handles coordinate
mapping between ungapped reference sequences and gapped alignment sequences, and
supports various GenBank location formats including complementary strands and
complex location descriptors.

Key functions:
- parse_genbank_location_string: Parses GenBank feature location strings
- parse_annotation_fasta_for_extraction: Extracts gene information from FASTA annotations
- map_coordinates_to_alignment_for_extraction: Maps gene coordinates to alignment positions
- extract_gene_alignments_from_genome_msa: Main orchestration function

Based on the logic from the original extract_genes_aln.py script.
"""

import argparse
import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import utils  # For sanitize_filename

# Configure module logger
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ANNOTATION PARSING FUNCTIONS
# -----------------------------------------------------------------------------

def parse_genbank_location_string(location_str: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Parses GenBank feature location strings into coordinates and strand information.
    
    Handles various formats including:
    - Simple ranges: '100..250'
    - Complementary strand: 'complement(500..300)'
    - Boundary markers: '<100..>250'
    - Single positions: '42'
    - Complex locations: 'join(100..200,300..400)'
    
    For complex locations (join/order), extracts the outermost coordinates.
    
    Args:
        location_str: The GenBank location string to parse
        
    Returns:
        Tuple of (start, end, strand):
            - start: 1-based inclusive start position (None if parsing fails)
            - end: 1-based inclusive end position (None if parsing fails)
            - strand: '+' for forward strand, '-' for reverse/complement strand
            
    Note:
        For single position features, start and end will be the same value.
        The function ensures start <= end regardless of input order.
    """
    # Default to forward strand
    strand = '+'
    original_input = location_str
    
    # Check for complement notation which indicates reverse strand
    match_complement = re.match(r'complement\((.*)\)', location_str)
    if match_complement:
        strand = '-'
        location_str = match_complement.group(1)
    
    # Remove join/order notation but keep the coordinates inside
    location_str = location_str.replace('join(', '').replace('order(', '').replace(')', '')
    
    # Find all coordinate pairs (handles multiple ranges in join/order statements)
    coords = re.findall(r'<?(\d+)\.\.>?(\d+)', location_str)
    
    # Handle single position features (no range)
    if not coords:
        single_point = re.search(r'<?(\d+)>?', location_str)
        if single_point:
            pos = int(single_point.group(1))
            return pos, pos, strand
        else:
            logger.warning(f"Could not parse coordinates from location string: '{original_input}'")
            return None, None, strand
    
    # Process all coordinate pairs to find outermost positions
    try:
        all_pos: List[int] = []
        for start_str, end_str in coords:
            all_pos.append(int(start_str))
            all_pos.append(int(end_str))
        
        # Get outermost positions
        start: Optional[int] = min(all_pos) if all_pos else None
        end: Optional[int] = max(all_pos) if all_pos else None
        
        # Ensure start <= end
        if start is not None and end is not None and start > end:
            start, end = end, start  # Swap to ensure correct order
        
        return start, end, strand
    except ValueError:
        logger.warning(f"Error converting coordinates to numbers in location: '{original_input}'")
        return None, None, strand


def parse_annotation_fasta_for_extraction(annotation_path: Path) -> List[Dict[str, Any]]:
    """
    Reads a multi-FASTA reference gene file and extracts gene information from headers.
    
    Parses FASTA headers with format like:
    '>lcl|ID [gene=NAME] ... [location=LOC]' or using '[locus_tag=...]'
    
    Args:
        annotation_path: Path to the annotation FASTA file
        
    Returns:
        List of dictionaries, each containing gene information:
            - 'GeneName': Gene name or locus tag (str)
            - 'Start': 1-based start position (int)
            - 'End': 1-based end position (int)
            - 'Strand': Strand direction ('+' or '-')
            - 'OriginalLocationStr': Original location string from header
            
    Raises:
        FileNotFoundError: If annotation file doesn't exist
        ValueError: If there's an error parsing the annotation file
    """
    gene_annotations: List[Dict[str, Any]] = []
    required_fields_found = 0
    
    # Verify file exists
    if not annotation_path.is_file():
        logger.error(f"Annotation file not found: {annotation_path}")
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    logger.info(f"Parsing annotations from: {annotation_path}...")
    try:
        # Parse each record in the FASTA file
        for record in SeqIO.parse(str(annotation_path), "fasta"):
            header = record.description
            gene_name: Optional[str] = None
            location_str: Optional[str] = None
            
            # Extract gene name or locus tag
            gene_match = re.search(r'\[(?:gene|locus_tag)=([^\]]+)\]', header)
            if gene_match:
                gene_name = gene_match.group(1)
            
            # Extract location information
            location_match = re.search(r'\[location=([^\]]+)\]', header)
            if location_match:
                location_str = location_match.group(1)
            
            # Process records with both gene name and location
            if gene_name and location_str:
                # Parse location string to get coordinates and strand
                start, end, strand = parse_genbank_location_string(location_str)
                if start is not None and end is not None:
                    gene_annotations.append({
                        'GeneName': gene_name,
                        'Start': start,
                        'End': end,
                        'Strand': strand,
                        'OriginalLocationStr': location_str
                    })
                    required_fields_found += 1
                else:
                    logger.warning(f"Skipping record '{record.id}': Could not parse location '{location_str}'.")
            # Uncomment for more verbose logging
            # else:
            #     logger.debug(f"Gene name or location tag missing for record '{record.id}'.")
    
    except Exception as e:
        logger.exception(f"Error parsing annotation file {annotation_path}: {e}")
        raise ValueError(f"Error parsing annotation file {annotation_path}") from e
    
    # Log summary of parsing results
    if required_fields_found == 0:
        logger.warning(f"No annotations with gene name/locus_tag and parsable location found in {annotation_path}.")
    else:
        logger.info(f"Found {required_fields_found} potential gene annotations.")
    
    return gene_annotations


# -----------------------------------------------------------------------------
# COORDINATE MAPPING FUNCTIONS
# -----------------------------------------------------------------------------

def map_coordinates_to_alignment_for_extraction(
    genes_info_list: List[Dict[str, Any]],
    ref_aligned_record: SeqRecord
) -> List[Dict[str, Any]]:
    """
    Maps ungapped 1-based gene coordinates to 0-based indices in the aligned reference sequence.
    
    This function creates a mapping between positions in the ungapped reference sequence
    and positions in the aligned (gapped) reference sequence, then uses this mapping
    to convert gene coordinates.
    
    Args:
        genes_info_list: List of gene info dictionaries from parse_annotation_fasta_for_extraction
        ref_aligned_record: The reference SeqRecord from the alignment (with gap characters)
        
    Returns:
        Updated list of gene info dictionaries with additional keys:
            - 'Aln_Start_0based': 0-based start index in the alignment
            - 'Aln_End_Exclusive': Exclusive end index in the alignment (for slicing)
            
    Raises:
        ValueError: If reference sequence record is missing
    """
    # Validate input
    if not ref_aligned_record:
        logger.error("Reference sequence record is missing, cannot map coordinates.")
        raise ValueError("Reference sequence record is missing for coordinate mapping.")
    
    # Get the aligned reference sequence as a string
    ref_aligned_seq = str(ref_aligned_record.seq)
    mapped_genes_info: List[Dict[str, Any]] = []
    
    # Create mapping from ungapped positions (1-based) to aligned positions (0-based)
    ungapped_to_aligned_map: Dict[int, int] = {}
    ungapped_pos = 0
    for i, char in enumerate(ref_aligned_seq):
        if char != '-':  # Skip gap characters
            ungapped_pos += 1
            ungapped_to_aligned_map[ungapped_pos] = i
    
    # Store the total ungapped length for validation
    max_ref_ungapped_len = ungapped_pos
    
    logger.info(f"Mapping coordinates relative to aligned reference '{ref_aligned_record.id}' "
                f"(ungapped length: {max_ref_ungapped_len})...")
    
    # Counters for skipped genes
    skipped_outside = 0
    skipped_mapping = 0
    
    # Process each gene and map its coordinates
    for gene_info in genes_info_list:
        start_orig = gene_info['Start']
        end_orig = gene_info['End']
        
        # Check if coordinates are within the reference sequence
        if not (0 < start_orig <= max_ref_ungapped_len and 0 < end_orig <= max_ref_ungapped_len):
            logger.warning(f"Original coordinates {start_orig}..{end_orig} for gene '{gene_info['GeneName']}' "
                          f"fall outside reference ungapped length ({max_ref_ungapped_len}). Skipping.")
            skipped_outside += 1
            continue
        
        try:
            # Map the coordinates to alignment positions
            aln_start_idx_0based = ungapped_to_aligned_map[start_orig]
            aln_end_idx_0based = ungapped_to_aligned_map[end_orig]
            
            # Ensure mapped coordinates are in correct order
            if aln_start_idx_0based <= aln_end_idx_0based:
                # Add mapped coordinates to gene info
                gene_info['Aln_Start_0based'] = aln_start_idx_0based
                gene_info['Aln_End_Exclusive'] = aln_end_idx_0based + 1  # Make end exclusive for slicing
                mapped_genes_info.append(gene_info)
            else:
                logger.warning(f"Mapped alignment indices reversed for gene '{gene_info['GeneName']}'. Skipping.")
                skipped_mapping += 1
        except KeyError as e:
            logger.warning(f"Could not map coordinate {e} for gene '{gene_info['GeneName']}'. Skipping.")
            skipped_mapping += 1
    
    # Log mapping summary
    logger.info(f"Coordinate mapping done: Mapped: {len(mapped_genes_info)}, "
               f"Skipped (outside): {skipped_outside}, Skipped (mapping issue): {skipped_mapping}.")
    
    return mapped_genes_info


# -----------------------------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# -----------------------------------------------------------------------------

def extract_gene_alignments_from_genome_msa(
    annotations_path: Path,
    alignment_path: Path,
    ref_id: str,
    output_dir: Path
) -> None:
    """
    Extract individual gene alignments from a whole genome multiple sequence alignment.
    
    This function orchestrates the entire extraction process:
    1. Validates inputs and creates output directory
    2. Parses gene annotations from the annotation file
    3. Reads the genome alignment and identifies the reference sequence
    4. Maps gene coordinates from the reference to the alignment
    5. Extracts gene sequences for each genome in the alignment
    6. Writes individual gene alignment files
    
    Args:
        annotations_path: Path to the annotation file (FASTA format with location tags)
        alignment_path: Path to the whole genome alignment file (FASTA format)
        ref_id: ID of the reference sequence in the alignment
        output_dir: Directory where extracted gene alignments will be written
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: For parsing errors or if reference sequence is not found
        OSError: If output directory cannot be created
    """
    logger.info(f"Starting gene extraction: Annotations='{annotations_path}', "
               f"Alignment='{alignment_path}', RefID='{ref_id}'")
    
    # 1. Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory for extracted genes: {output_dir}")
    except OSError as e:
        raise OSError(f"Error creating output directory '{output_dir}': {e}") from e
    
    # 2. Parse Annotations
    gene_annotations = parse_annotation_fasta_for_extraction(annotations_path)
    if not gene_annotations:
        raise ValueError("No valid annotations parsed from the annotation file. Cannot proceed with extraction.")
    
    # 3. Read Genome Alignment and Find Reference
    logger.info(f"Reading genome alignment: {alignment_path}...")
    ref_aligned_record: Optional[SeqRecord] = None
    genome_alignment_records: List[SeqRecord] = []
    
    try:
        # Read all sequences from the alignment file
        temp_records = list(SeqIO.parse(str(alignment_path), "fasta"))
        if not temp_records:
            raise ValueError(f"No sequences found in genome alignment file: {alignment_path}")
        
        # Store all records and identify the reference sequence
        for record in temp_records:
            genome_alignment_records.append(record)
            if record.id == ref_id:
                ref_aligned_record = record
        
        # Verify reference sequence was found
        if not ref_aligned_record:
            available_ids_sample = [rec.id for rec in genome_alignment_records[:5]]
            logger.error(f"Reference sequence ID '{ref_id}' not found in alignment. "
                        f"Available sample IDs: {available_ids_sample}...")
            raise ValueError(f"Reference sequence ID '{ref_id}' not found in alignment file.")
        
        logger.info(f"Read {len(genome_alignment_records)} genome sequences. Found reference '{ref_id}'.")
    
    except ValueError as e:
        # Re-raise ValueError for no sequences or missing reference
        raise e
    except Exception as e:
        # Convert other exceptions to ValueError with context
        raise ValueError(f"Error reading or parsing genome alignment file {alignment_path}: {e}") from e
    
    # 4. Map Coordinates
    aligned_genes_info = map_coordinates_to_alignment_for_extraction(gene_annotations, ref_aligned_record)
    if not aligned_genes_info:
        raise ValueError("Could not map coordinates for any gene. Check reference ID and annotation formats.")
    
    # 5. Extract and Write Gene Alignments
    logger.info("Extracting and writing gene alignments...")
    genes_written = 0
    genes_failed = 0
    genes_skipped_extraction = 0
    
    # Process each gene
    for gene_info in aligned_genes_info:
        gene_name = gene_info['GeneName']
        safe_gene_name = utils.sanitize_filename(gene_name)
        output_filename = output_dir / f"gene_{safe_gene_name}.fasta"
        logger.debug(f"Processing {gene_name} -> {output_filename} ...")
        
        # Extract gene-specific records from each genome
        gene_specific_records: List[SeqRecord] = []
        start_aln = gene_info['Aln_Start_0based']
        end_aln = gene_info['Aln_End_Exclusive']
        strand = gene_info['Strand']
        
        # Validate mapped coordinates
        if start_aln < 0 or end_aln <= start_aln:
            logger.warning(f"Invalid mapped coords for gene {gene_name}. Skipping.")
            genes_failed += 1
            continue
        
        # Extract gene sequence from each genome in the alignment
        for genome_record in genome_alignment_records:
            genome_seq = genome_record.seq
            genome_id = genome_record.id
            genome_len = len(genome_seq)
            
            # Skip if coordinates exceed genome length
            if end_aln > genome_len:
                logger.debug(f"Coords for gene {gene_name} exceed length of genome {genome_id}. "
                            f"Skipping this genome for this gene.")
                continue
            
            # Extract subsequence
            sub_sequence: Seq = genome_seq[start_aln:end_aln]
            
            # Handle reverse complement for negative strand
            if strand == '-':
                try:
                    sub_sequence = sub_sequence.reverse_complement()
                except Exception as rc_err:
                    logger.warning(f"Could not reverse complement for {genome_id} gene {gene_name}: "
                                  f"{rc_err}. Using forward.")
            
            # Create record for this gene from this genome
            extracted_record = SeqRecord(
                sub_sequence,
                id=genome_id,
                description=f"gene={gene_name} | source_location={gene_info['OriginalLocationStr']}"
            )
            gene_specific_records.append(extracted_record)
        
        # Write gene alignment to file if records were extracted
        if gene_specific_records:
            try:
                with open(output_filename, "w") as outfile:
                    SeqIO.write(gene_specific_records, outfile, "fasta")
                genes_written += 1
            except IOError as e:
                logger.error(f"Error writing output file {output_filename}: {e}")
                genes_failed += 1
            except Exception as e:
                logger.exception(f"Unexpected error writing {output_filename}: {e}")
                genes_failed += 1
        else:
            genes_skipped_extraction += 1
            logger.warning(f"No sequences could be extracted for gene {gene_name}. No output file generated.")
    
    # 6. Log Summary Statistics
    logger.info("--- Gene Extraction Summary ---")
    logger.info(f"  Total annotations parsed:         {len(gene_annotations)}")
    logger.info(f"  Annotations successfully mapped:  {len(aligned_genes_info)}")
    logger.info(f"  Gene alignments written:          {genes_written}")
    
    if genes_skipped_extraction > 0:
        logger.info(f"  Genes skipped (no seq extracted): {genes_skipped_extraction}")
    
    if genes_failed > 0:
        logger.info(f"  Genes failed (error):             {genes_failed}")
    
    if genes_written == 0 and len(aligned_genes_info) > 0:
        logger.warning("No gene alignment files written despite mapping. Check logs.")