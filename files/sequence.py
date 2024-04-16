"""
Sequence Class.

This module allows the user to store one
sequence from a fasta file in a Sequence instance.

Classes
-------
Sequence
"""

from __future__ import annotations


class Sequence:
    """A class to represent a sequence."""

    def __init__(self, seq: str, id: str) -> None:
        """Construct all attributes for Sequence."""
        self.seq = seq
        self.id = id
        self.predId = ""
        self.onehot = ""

    @property
    def seq(self) -> str:
        """Sequence as string."""
        return self._seq

    @seq.setter
    def seq(self, seq: str) -> None:
        if isinstance(seq, str):
            self._seq = seq
        else:
            raise ValueError('"seq" must be a str')

    @property
    def id(self) -> str:
        """Sequence id."""
        return self._id

    @id.setter
    def id(self, id: str) -> None:
        self._id = id

    @property
    def predId(self) -> str:
        """Sequence predicted id."""
        return self._predId

    @predId.setter
    def predId(self, predId: str) -> None:
        self._predId = predId

    @property
    def onehot(self) -> str:
        """Sequence one-hot encoding."""
        return self._onehot

    @onehot.setter
    def onehot(self, onehot: str) -> None:
        self._onehot = onehot

    def getFeatures(self) -> list[str]:
        """Return Sequence features."""
        features: list[str] = list(self.onehot)
        features.append(self.id)
        return features

    def encode(self) -> None:
        """Encode Sequence using one-hot."""
        for base in self.seq:
            match base:
                case "A":
                    self.onehot += "1000"
                case "G":
                    self.onehot += "0100"
                case "C":
                    self.onehot += "0010"
                case "T":
                    self.onehot += "0001"

    def getBase(self, pos: int) -> str:
        """Return base pair in Sequence."""
        length: int = self.getLength()
        if pos >= length:
            raise ValueError(f"{pos} greater than length {length}")
        base: str = self.seq[pos]
        return base

    def getLength(self) -> int:
        """Return Sequence length."""
        length: int = len(self.seq)
        return length

    def print(self) -> None:
        """Print Sequence."""
        print(self.seq)
