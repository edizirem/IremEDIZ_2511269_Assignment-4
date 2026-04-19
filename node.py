"""
================================================================================
Module      : material.py
Purpose     : Material class storing elastic properties.
Attributes  : E – Young's modulus [kN/m²]
              G – Shear modulus [kN/m²]
Units       : kN, m
================================================================================
"""


class Material:
    """Stores elastic material properties."""

    def __init__(self, name: str, E: float, G: float = 0.0):
        """
        Inputs:
            name – material label (str)
            E    – Young's modulus [kN/m²]
            G    – Shear modulus [kN/m²] (optional)
        """
        self.name = name
        self.E    = float(E)
        self.G    = float(G)

    def __repr__(self):
        return f"Material({self.name}: E={self.E:.2e}, G={self.G:.2e})"
