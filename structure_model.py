"""
================================================================================
Module      : section.py
Purpose     : Section class storing cross-sectional properties.
Attributes  : A – cross-sectional area [m²]
              I – second moment of area [m⁴]
Units       : m
================================================================================
"""


class Section:
    """Stores cross-sectional geometric properties."""

    def __init__(self, name: str, A: float, I: float = 0.0):
        """
        Inputs:
            name – section label (str)
            A    – cross-sectional area [m²]
            I    – second moment of area [m⁴] (0 for truss)
        """
        self.name = name
        self.A    = float(A)
        self.I    = float(I)

    def __repr__(self):
        return f"Section({self.name}: A={self.A:.4f}, I={self.I:.4e})"
