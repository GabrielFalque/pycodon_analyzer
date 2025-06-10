# src/pycodon_analyzer/io.py
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Handles input/output operations for the pycodon_analyzer, focusing on reading sequence files.
"""
import sys
import os
import logging
from typing import List, Set

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

# Configure logging for this module
logger = logging.getLogger(__name__)

# Import constants from utils (assuming utils.py is correctly structured)
try:
    from .utils import VALID_DNA_CHARS
except ImportError:
    # Fallback if run standalone or structure issue. Use logger here.
    logger.warning(
        "Could not import VALID_DNA_CHARS from .utils. Using a basic DNA character set. "
        "This might affect sequence validation if custom VALID_DNA_CHARS was intended."
    )
    VALID_DNA_CHARS: Set[str] = set('ATCGN-') # Default fallback



def read_fasta(filepath: str) -> List[SeqRecord]:
    """Reads sequences from a FASTA file.

    This function parses a FASTA file, performs basic validation, and returns a list of Biopython SeqRecord objects.

    Validation steps include:
    - Checking for file existence.
    - Converting sequences to uppercase.
    - Warning about non-DNA characters (based on VALID_DNA_CHARS).
    - Skipping empty sequences.

    Args:
        filepath (str): The path to the FASTA file.

    Returns:
        List[SeqRecord]: A list of Biopython SeqRecord objects found in the file.  Returns an empty list if no valid sequences are found or if an error occurs.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be parsed as a FASTA file.
    """
    logger.debug(f"Attempting to read FASTA file: {filepath}")
    records: List[SeqRecord] = []
    file_had_content = False  # Flag to track if the file was opened successfully and had content

    try:
        with open(filepath, 'r') as handle:
            # Check if the file is empty before parsing to avoid errors
            initial_pos = handle.tell()  # Store the current position
            handle.seek(0, os.SEEK_END)  # Go to the end of the file
            if handle.tell() == 0:  # If the file size is 0, it's empty
                logger.warning(f"FASTA file '{filepath}' is empty.")
                return []
            handle.seek(initial_pos)  # Reset file pointer to the beginning

            file_had_content = True  # File was not empty

            for record in SeqIO.parse(handle, "fasta"):  # Parse the FASTA file using Biopython
                # Sequence Processing and Validation
                try:
                    # Ensure sequence attribute exists and is usable; if not, log a warning and skip the record
                    if not hasattr(record, 'seq') or record.seq is None:
                         logger.warning(f"Record '{record.id}' in {os.path.basename(filepath)} has missing sequence data. Skipping record.")
                         continue

                    # Convert sequence to uppercase string for consistency and create a new Seq object
                    seq_str = str(record.seq).upper()
                    record.seq = Seq(seq_str)  # Update record's seq object

                    # Check for empty sequences; if empty, log a warning and skip the record
                    if not record.seq:
                         logger.warning(f"Sequence '{record.id}' in {os.path.basename(filepath)} is empty. Skipping record.")
                         continue  # Skip empty sequences

                    # Basic validation for non-DNA characters (allows gaps and Ns by default if in VALID_DNA_CHARS); log only if invalid characters are found
                    sequence_chars: Set[str] = set(seq_str)
                    invalid_chars: Set[str] = sequence_chars - VALID_DNA_CHARS
                    if invalid_chars:
                        # Log as warning, as the program might still handle some non-standard chars later, as the program might still handle some non-standard chars later
                        logger.warning(
                            f"Sequence '{record.id}' in {os.path.basename(filepath)} "
                            f"contains potentially invalid characters (not in VALID_DNA_CHARS): {invalid_chars}. "
                            "These might affect downstream analysis."
                        )

                    # If sequence passes checks, add it to the list of valid records
                    records.append(record)

                except AttributeError as attr_err:
                    # Handle cases where the record object might be malformed, log a warning, and skip the record
                    logger.warning(f"Skipping record due to attribute error (likely malformed record) in {os.path.basename(filepath)}: {attr_err}")
                    continue
                except Exception as rec_proc_err:
                    # Catch unexpected errors during processing of a single record, log the exception, and skip the record
                    logger.exception(f"Error processing record '{record.id}' in {os.path.basename(filepath)}: {rec_proc_err}. Skipping record.")
                    continue

    except FileNotFoundError:
        logger.error(f"FASTA file not found: '{filepath}'")  # Log the error
        return []
    except ValueError as parse_err:
        logger.error(f"Error parsing FASTA file '{os.path.basename(filepath)}'. Check file format. Details: {parse_err}")  # Log the error
        return []
    except Exception as e:
        logger.exception(f"An unexpected error occurred while reading FASTA file '{filepath}': {e}")  # Log the exception
        return []

    # After attempting to read, if no records were found AND the file was not empty, it indicates a parsing issue
    if not records and file_had_content:
        logger.error(f"FASTA file '{filepath}' contains content but no valid FASTA records could be parsed.")
    elif not records:  # This case handles truly empty files (already logged above)
        pass  # Already handled by the initial empty file check

    logger.info(f"Successfully read {len(records)} sequences from {filepath}")  # Log the successful read

    # Optional: Add check for alignment (all sequences same length) if strictly required
    # This might be better handled in the analysis step depending on requirements.
    # if records:
    #     first_len = len(records[0].seq)
    #     if not all(len(rec.seq) == first_len for rec in records):
    #         logger.warning(f"Sequences in file '{os.path.basename(filepath)}' have different lengths. Treating as unaligned.")
    #         # Or raise ValueError("Error: Input sequences are not aligned (different lengths).")

    return records