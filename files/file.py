"""
File Class.

This module allows the user to process an input
FastaFile and use the file's contents for the
Kmeans and KNN algorithms.

Classes
-------
File
"""

import re
from typing import TextIO
from sequence import Sequence


class FastaFile:
    """A class to represent a fasta file."""

    def __init__(self, path: str) -> None:
        """Construct all attributes for File."""
        self.path = path

    @property
    def path(self) -> str:
        """Path to file."""
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        if isinstance(path, str):
            self._path = path
        else:
            raise ValueError('"path" must be a str')

    def _processIds(self, lines: list[str]) -> list[str]:
        """Process ids from fasta file."""
        ids: list[str] = list()
        for line in lines:
            if line.startswith(">"):
                line = line.strip("\n")
                idx: int = line.find("=")
                start: int = idx + 1
                id: str = line[start:]
                ids.append(id)
        return ids

    def _remove(self, alist: list[str], element: str) -> list[str]:
        """Return alist without element."""
        count: int = alist.count(element)
        for i in range(count):
            alist.remove(element)
        return alist

    def _processSeqs(self, lines: list[str]) -> list[str]:
        """Process sequences from fasta file."""
        seqs: list[str] = list()
        seq: str = ""
        for line in lines:
            if not line.startswith(">"):
                line = re.sub("[\n]", "", line)
                seq = seq + line
            else:
                seqs.append(seq)
                seq = ""
        seqs.append(seq)
        seqs = self._remove(seqs, "")
        return seqs

    def _createSeqs(self, ids: list[str], seqstr: list[str]) -> list[Sequence]:
        """Create Sequences."""
        seqs: list[Sequence] = list()
        for i in range(len(seqstr)):
            seqs.append(Sequence(seqstr[i], ids[i]))
        return seqs

    def getSequences(self) -> list[Sequence]:
        """Process infile for Sequences."""
        file: TextIO = open(self.path, "r")
        lines: list[str] = file.readlines()
        file.close()

        ids: list[str] = self._processIds(lines)
        seqstr: list[str] = self._processSeqs(lines)
        seqs: list[Sequence] = self._createSeqs(ids, seqstr)
        return seqs

    def print(self) -> None:
        """Print Sequence."""
        seqs: list[Sequence] = self.getSequences()
        for seq in seqs:
            print(seq.id)
            seq.print()


# file: FastaFile = FastaFile("sim1/train.fasta")
# seqs: list[Sequence] = file.getSequences()
# for seq in seqs:
#     seq.encode()
# firstSeq: Sequence = seqs[0]
# length: int = firstSeq.getLength()
# onehot: int = len(firstSeq.onehot)
# print(f"Length: {length} Onehot: {onehot}")
