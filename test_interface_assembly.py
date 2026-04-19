"""
================================================================================
Module      : support.py
Purpose     : Abstract Support class and concrete subtypes.
              Restraint pattern: [ux, uy, rz] → True = restrained, False = free
Units       : dimensionless (restraint flags)
================================================================================
"""


class Support:
    """Abstract base class for all supports."""

    def __init__(self, name: str):
        self.name = name
        self.restraints = [False, False, False]   # [ux, uy, rz]

    def isA(self, support_type) -> bool:
        """Check if this support is an instance of support_type."""
        return isinstance(self, support_type)

    def restrained_dofs(self) -> list:
        """Return list of restrained local DOF indices [0,1,2]."""
        return [i for i, r in enumerate(self.restraints) if r]

    def __repr__(self):
        labels  = ['ux', 'uy', 'rz']
        fixed   = [labels[i] for i in self.restrained_dofs()]
        return f"{self.__class__.__name__}({self.name}: fixed={fixed})"


class Fixed(Support):
    """Fixed support — ux, uy, rz all restrained."""
    def __init__(self, name: str = "Fixed"):
        super().__init__(name)
        self.restraints = [True, True, True]


class Pin(Support):
    """Pin support — ux, uy restrained; rz free."""
    def __init__(self, name: str = "Pin"):
        super().__init__(name)
        self.restraints = [True, True, False]


class Roller(Support):
    """Roller support — uy restrained only."""
    def __init__(self, name: str = "Roller"):
        super().__init__(name)
        self.restraints = [False, True, False]


class FixedRoller(Support):
    """Fixed-Roller — uy and rz restrained, ux free."""
    def __init__(self, name: str = "FixedRoller"):
        super().__init__(name)
        self.restraints = [False, True, True]


class XRoller(Support):
    """Horizontal roller — ux restrained only."""
    def __init__(self, name: str = "XRoller"):
        super().__init__(name)
        self.restraints = [True, False, False]
