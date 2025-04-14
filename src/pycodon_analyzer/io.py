# src/pycodon_analyzer/io.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Input/Output operations, primarily for reading sequence files.
"""
import sys
import os
import logging # <-- Import logging
from typing import List, Set # <-- Import typing helpers

# Biopython imports
try:
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
except ImportError:
    # Log error and exit? Or let subsequent code fail?
    # Logging might not be configured yet here. Print and exit might be safer.
    print("ERROR: Biopython is not installed or cannot be found. Please install it (`pip install biopython`).", file=sys.stderr)
    sys.exit(1)

# Import constants from utils (assuming utils.py is correctly structured)
try:
    from .utils import VALID_DNA_CHARS
except ImportError:
    # Fallback if run standalone or structure issue, less robust
    print("Warning: Could not import VALID_DNA_CHARS from .utils. Using basic DNA set.", file=sys.stderr)
    VALID_DNA_CHARS: Set[str] = set('ATCGN-')

# --- Configure logging for this module ---
logger = logging.getLogger(__name__)


def read_fasta(filepath: str) -> List[SeqRecord]:
    """
    Reads sequences from a FASTA file.

    Performs basic validation:
    - Checks for file existence.
    - Converts sequences to uppercase.
    - Warns about non-DNA characters found (based on VALID_DNA_CHARS).
    - Skips empty sequences.

    Args:
        filepath (str): Path to the FASTA file.

    Returns:
        List[SeqRecord]: A list of Biopython SeqRecord objects found in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as FASTA, or if no valid
                    sequences are found after basic checks.
    """
    logger.debug(f"Attempting to read FASTA file: {filepath}")
    records: List[SeqRecord] = []

    try:
        with open(filepath, 'r') as handle:
            # Use Biopython's SeqIO parser
            for record in SeqIO.parse(handle, "fasta"):
                # --- Sequence Processing and Validation ---
                try:
                    # Ensure sequence attribute exists and is usable
                    if not hasattr(record, 'seq') or record.seq is None:
                         logger.warning(f"Record '{record.id}' in {os.path.basename(filepath)} has missing sequence data. Skipping.")
                         continue

                    # Convert sequence to uppercase string for consistency
                    seq_str = str(record.seq).upper()
                    record.seq = Seq(seq_str) # Update record's seq object

                    # Check for empty sequences
                    if not record.seq:
                         logger.warning(f"Sequence '{record.id}' in {os.path.basename(filepath)} is empty. Skipping.")
                         continue # Skip empty sequences

                    # Basic validation for non-DNA characters (allows gaps and Ns by default if in VALID_DNA_CHARS)
                    # Log only if invalid characters are found
                    sequence_chars: Set[str] = set(seq_str)
                    invalid_chars: Set[str] = sequence_chars - VALID_DNA_CHARS
                    if invalid_chars:
                        # Log as warning, as the program might still handle some non-standard chars later
                        logger.warning(f"Sequence '{record.id}' in {os.path.basename(filepath)} "
                                       f"contains potentially invalid characters (not in VALID_DNA_CHARS): {invalid_chars}. "
                                       "These might affect downstream analysis.")

                    # If sequence passes checks, add it to the list
                    records.append(record)

                except AttributeError as attr_err:
                    # Handle cases where the record object might be malformed
                    logger.warning(f"Skipping record due to attribute error (likely malformed record) in {os.path.basename(filepath)}: {attr_err}")
                    continue
                except Exception as rec_proc_err:
                    # Catch unexpected errors during processing of a single record
                    logger.exception(f"Error processing record '{record.id}' in {os.path.basename(filepath)}: {rec_proc_err}. Skipping record.")
                    continue

    except FileNotFoundError:
        # Log the error and re-raise specifically for the caller (cli.py)
        logger.error(f"Input FASTA file not found: '{filepath}'")
        raise # Re-raise FileNotFoundError

    except ValueError as parse_err:
        # SeqIO.parse often raises ValueError for format issues
        logger.error(f"Error parsing FASTA file '{os.path.basename(filepath)}'. Check file format. Details: {parse_err}")
        # Raise a new ValueError to indicate parsing failure
        raise ValueError(f"Failed to parse FASTA file '{filepath}'. Ensure it is a valid FASTA file.") from parse_err

    except Exception as e:
        # Catch other potential file reading or unexpected errors
        logger.exception(f"An unexpected error occurred while reading FASTA file '{filepath}': {e}")
        # Raise a generic ValueError
        raise ValueError(f"Failed to read or process FASTA file '{filepath}'.") from e

    # After successfully reading the file, check if any valid records were found
    if not records:
        # Log an error and raise ValueError, as this usually indicates an issue
        # (e.g., file exists but contains only invalid/empty sequences)
        logger.error(f"No valid sequences found in FASTA file: '{filepath}'")
        raise ValueError(f"No valid sequences found in '{filepath}'.")

    logger.debug(f"Successfully read {len(records)} sequences from {filepath}")

    # Optional: Add check for alignment (all sequences same length) if strictly required
    # This might be better handled in the analysis step depending on requirements.
    # if records:
    #     first_len = len(records[0].seq)
    #     if not all(len(rec.seq) == first_len for rec in records):
    #         logger.warning(f"Sequences in file '{os.path.basename(filepath)}' have different lengths. Treating as unaligned.")
    #         # Or raise ValueError("Error: Input sequences are not aligned (different lengths).")

    return records