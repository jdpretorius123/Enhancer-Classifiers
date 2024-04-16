"""
Write KNN Output.

This module allows the user to setup
and run the KNN algorithm.

Functions
---------
write(argv: list[str]) -> None:
    Writes classification results.
"""

import os
import numpy as np
from typing import TextIO
from file import FastaFile
from sequence import Sequence


def encodeSeqs(seqs: list[Sequence]) -> None:
    """Encode Sequences using one-hot."""
    for seq in seqs:
        seq.encode()


def prepData(infile: str, outfile: str) -> None:
    """Prepare data for neural network."""
    fasta: FastaFile = FastaFile(infile)
    seqs: list[Sequence] = fasta.getSequences()
    encodeSeqs(seqs)
    file: TextIO = open(outfile, "w")
    for seq in seqs:
        features: list[str] = seq.getFeatures()
        line: str = "\t".join(features)
        file.write(line + "\n")
    file.close()


def convertData(file: str) -> np.ndarray:
    """Convert text file to NumPy array."""
    data: np.ndarray = np.loadtxt(fname=file, delimiter="\t")
    return data


def getFeatures(data: np.ndarray) -> np.ndarray:
    """Return features of data."""
    col: int = data.shape[1]
    col = col - 1
    return data[:, :col]


def getClasses(data: np.ndarray) -> np.ndarray:
    """Return classes of data."""
    col: int = data.shape[1]
    col = col - 1
    return data[:, col:]


def prepAllData(fastafiles: list[str], textfiles: list[str]) -> None:
    """Prepare training, validation, and testing data for neural network."""
    prepData(fastafiles[0], textfiles[0])
    prepData(fastafiles[1], textfiles[1])
    prepData(fastafiles[2], textfiles[2])


def getDataSets(file: str) -> tuple[np.ndarray, np.ndarray]:
    """Return NumPy arrays for neural network."""
    data: np.ndarray = convertData(file)
    features: np.ndarray = getFeatures(data)
    classes: np.ndarray = getClasses(data)
    return features, classes

def reshape(data: np.ndarray) -> np.ndarray:
    """Reshape data structure."""
    ncol: int = data.shape[1] // 4
    reshaped: np.ndarray = data.reshape((-1,) + (ncol, 4))
    return reshaped


# prepData("sim1/train.fasta", "sim1/test.txt")

# data: np.ndarray = convertData("sim1/test.txt")
# x: np.ndarray = getFeatures(data)
# y: np.ndarray = getClasses(data)

# xrow: int = x.shape[0]
# xcol: int = x.shape[1]
# print(f"X rows: {xrow} X cols: {xcol}")
# print(x[0:10,997:])

# yrow: int = y.shape[0]
# ycol: int = y.shape[1]
# print(f"Y rows: {yrow} Y cols: {ycol}")
# print(y[0:10,:])
