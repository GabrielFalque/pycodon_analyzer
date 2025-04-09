# src/pycodon_analyzer/io.py

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

"""
Input/Output operations, primarily for reading sequence files.
"""
import sys
import os
from Bio import SeqIO # type: ignore
from Bio.SeqRecord import SeqRecord # type: ignore
from Bio.Seq import Seq # type: ignore
from .utils import VALID_DNA_CHARS

def read_fasta(filepath, v=False):
    """
    Reads sequences from a FASTA file.
    Performs basic validation to check for non-DNA characters.
    Converts sequences to uppercase.

    Args:
        filepath (str): Path to the FASTA file.

    Returns:
        list[SeqRecord]: A list of Biopython SeqRecord objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If sequences contain invalid characters or are empty,
                    or if no valid sequences are found.
    """
    records = []
    try:
        with open(filepath, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Convert sequence to uppercase for consistency
                record.seq = record.seq.upper()

                # Basic validation for non-DNA characters (allows gaps and Ns)
                # This check might be less critical now if input is assumed clean alignment
                invalid_chars = set(str(record.seq)) - VALID_DNA_CHARS
                if invalid_chars and v:
                    print(f"Warning: Sequence {record.id} in file {os.path.basename(filepath)} "
                          f"contains potentially invalid characters: {invalid_chars}. "
                          "These might affect codon counting.", file=sys.stderr)

                if not record.seq:
                     print(f"Warning: Sequence {record.id} in file {os.path.basename(filepath)} is empty. Skipping.", file=sys.stderr)
                     continue # Skip empty sequences

                records.append(record)

    except FileNotFoundError:
        # Raise specific error to be caught by CLI
        raise FileNotFoundError(f"Error: Input FASTA file not found at '{filepath}'")
    except Exception as e:
        # Catch other potential parsing errors
        raise ValueError(f"Error parsing FASTA file '{filepath}': {e}")

    if not records:
        # Raise ValueError instead of just printing, so CLI knows to exit
        raise ValueError(f"No valid sequences found in '{filepath}'.")

    # Optional: Add check for alignment (all sequences same length) if strictly required
    # first_len = len(records[0].seq)
    # if not all(len(rec.seq) == first_len for rec in records):
    #     print("Warning: Sequences in the input file have different lengths. Treating as unaligned.", file=sys.stderr)
        # Or raise ValueError("Error: Input sequences are not aligned (different lengths).")

    return records