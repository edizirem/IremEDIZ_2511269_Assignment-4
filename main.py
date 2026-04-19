"""
================================================================================
Module      : loads.py
Purpose     : Load class hierarchy.
              NodalLoad   – concentrated force/moment at a node
              MemberLoad  – abstract base for loads on members
              PointLoad   – transverse point load at distance 'a' from start
              UniformLoad – uniformly distributed load over full member length
Units       : kN, m, kN·m
================================================================================
"""

import numpy as np


class NodalLoad:
    """Concentrated nodal load: Fx [kN], Fy [kN], Mz [kN·m]."""

    def __init__(self, name: str, Fx: float = 0.0,
                 Fy: float = 0.0, Mz: float = 0.0):
        self.name = name
        self.Fx   = float(Fx)
        self.Fy   = float(Fy)
        self.Mz   = float(Mz)

    def vector(self) -> np.ndarray:
        """Return [Fx, Fy, Mz] as numpy array."""
        return np.array([self.Fx, self.Fy, self.Mz])

    def __repr__(self):
        return f"NodalLoad({self.name}: Fx={self.Fx}, Fy={self.Fy}, Mz={self.Mz})"


class MemberLoad:
    """
    Abstract base for member loads.
    Subclasses must implement fixed_end_moments(L).
    """

    def __init__(self, name: str):
        self.name = name

    def isA(self, load_type) -> bool:
        return isinstance(self, load_type)

    def fixed_end_moments(self, L: float) -> np.ndarray:
        """
        Purpose  : Fixed-end force vector in LOCAL coordinates.
        Inputs   : L – member length [m]
        Outputs  : numpy array [N1, V1, M1, N2, V2, M2]
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class PointLoad(MemberLoad):
    """
    Transverse point load P at distance 'a' from start node.
    Positive P = local +y direction.

    Fixed-end moments (standard beam theory):
        Ra = P*b²*(3a+b) / L³
        Rb = P*a²*(a+3b) / L³
        Ma = P*a*b²       / L²   (clockwise at start = positive)
        Mb = -P*a²*b      / L²   (counter-clockwise at end)
    where b = L - a
    """

    def __init__(self, name: str, P: float, a: float):
        """
        Inputs:
            P – load magnitude [kN] (positive = local +y)
            a – distance from start node to load [m]
        """
        super().__init__(name)
        self.P = float(P)
        self.a = float(a)

    def fixed_end_moments(self, L: float) -> np.ndarray:
        P, a  = self.P, self.a
        b     = L - a
        Ra    =  P * b**2 * (3*a + b) / L**3
        Rb    =  P * a**2 * (a + 3*b) / L**3
        Ma    =  P * a * b**2          / L**2
        Mb    = -P * a**2 * b          / L**2
        return np.array([0.0, Ra, Ma, 0.0, Rb, Mb])


class UniformLoad(MemberLoad):
    """
    Uniformly distributed transverse load w [kN/m] over full member.
    Positive w = local +y direction.

    Fixed-end moments:
        Ra = Rb = wL/2
        Ma = wL²/12,  Mb = -wL²/12
    """

    def __init__(self, name: str, w: float):
        """
        Inputs:
            w – load intensity [kN/m] (positive = local +y)
        """
        super().__init__(name)
        self.w = float(w)

    def fixed_end_moments(self, L: float) -> np.ndarray:
        w  = self.w
        R  = w * L / 2.0
        M  = w * L**2 / 12.0
        return np.array([0.0, R, M, 0.0, R, -M])


# ==============================================================================
# Thermal Loads
# ==============================================================================

class ThermalLoad(MemberLoad):
    """
    Abstract base class for thermal loads.
    Temperature change causes element expansion/contraction,
    producing fixed-end forces even without external loads.

    Subclasses:
        AxialThermalLoad  – uniform temperature change (Truss or Frame)
                            produces axial force only: N = E*A*alpha*deltaT
        BeamThermalLoad   – temperature gradient across section height (Frame)
                            produces axial + bending moment:
                            N = E*A*alpha*deltaT_avg
                            M = E*I*alpha*deltaT_grad/h

    Units: deltaT [°C], alpha [1/°C], h [m]
    """

    def __init__(self, name: str, alpha: float, deltaT: float):
        """
        Inputs:
            alpha  – thermal expansion coefficient [1/°C]
            deltaT – uniform temperature change [°C]
                     positive = expansion, negative = contraction
        """
        super().__init__(name)
        self.alpha  = float(alpha)
        self.deltaT = float(deltaT)

    def fixed_end_moments(self, L: float) -> np.ndarray:
        raise NotImplementedError


class AxialThermalLoad(ThermalLoad):
    """
    Uniform temperature change applied to a member.
    Valid for BOTH Truss and Frame elements.

    Physics:
        Free thermal elongation: delta_T = alpha * deltaT * L
        If both ends are fixed, axial force builds up:
            N_thermal = E * A * alpha * deltaT   [kN]

    Fixed-end force vector (local coordinates):
        [N1, V1, M1, N2, V2, M2] = [-N, 0, 0, +N, 0, 0]
        where N = E * A * alpha * deltaT

    Note:
        For a Truss element (4 DOFs), only N1 and N2 are used.
        V1, M1, V2, M2 are zero — truss carries no moment.
        The Frame element uses all 6 components.
    """

    def __init__(self, name: str, alpha: float, deltaT: float):
        """
        Inputs:
            alpha  – thermal expansion coefficient [1/°C]  e.g. 12e-6 for steel
            deltaT – temperature change [°C]
                     positive = warming = expansion = compression in fixed member
        """
        super().__init__(name, alpha, deltaT)

    def fixed_end_moments(self, L: float,
                          E: float = None, A: float = None) -> np.ndarray:
        """
        Purpose  : Compute fixed-end axial force from uniform temperature change.
        Inputs   : L – member length [m]
                   E – Young's modulus [kN/m²]  (passed by element)
                   A – cross-sectional area [m²] (passed by element)
        Outputs  : numpy array [N1, V1, M1, N2, V2, M2] in local coords
        Note     : E and A are passed at assembly time from the element.
                   If not provided here (e.g. called without context),
                   N is returned as alpha*deltaT*L (unit stiffness form).
        """
        if E is not None and A is not None:
            N = E * A * self.alpha * self.deltaT
        else:
            N = self.alpha * self.deltaT * L   # placeholder
        # Compression at start (+N pushes inward), tension at end
        return np.array([-N, 0.0, 0.0, N, 0.0, 0.0])

    def __repr__(self):
        return (f"AxialThermalLoad({self.name}: "
                f"alpha={self.alpha:.2e}, deltaT={self.deltaT}°C)")


class BeamThermalLoad(ThermalLoad):
    """
    Temperature gradient across the section height (Frame elements only).
    Top face temperature differs from bottom face temperature.

    Physics:
        deltaT_top    – temperature at top face [°C]
        deltaT_bottom – temperature at bottom face [°C]
        deltaT_avg    = (deltaT_top + deltaT_bottom) / 2  → axial effect
        deltaT_grad   = (deltaT_top - deltaT_bottom) / h  → curvature effect

    Fixed-end forces (local coordinates, fixed-fixed beam):
        N_thermal = E * A * alpha * deltaT_avg
        M_thermal = E * I * alpha * deltaT_grad   (= E*I*alpha*(dT_top-dT_bot)/h)

        [N1, V1,  M1,         N2,  V2,  M2       ]
        [-N,  0, -M_thermal,  +N,   0, +M_thermal ]

    Note:
        For a uniform temperature change (deltaT_top == deltaT_bottom):
            deltaT_grad = 0 → no bending, only axial (same as AxialThermalLoad)
        For opposite temperatures (deltaT_top = -deltaT_bottom):
            deltaT_avg  = 0 → no axial, only bending
    """

    def __init__(self, name: str, alpha: float,
                 deltaT_top: float, deltaT_bottom: float, h: float):
        """
        Inputs:
            alpha         – thermal expansion coefficient [1/°C]
            deltaT_top    – temperature change at top face [°C]
            deltaT_bottom – temperature change at bottom face [°C]
            h             – section height [m]
        """
        deltaT_avg = (deltaT_top + deltaT_bottom) / 2.0
        super().__init__(name, alpha, deltaT_avg)
        self.deltaT_top    = float(deltaT_top)
        self.deltaT_bottom = float(deltaT_bottom)
        self.h             = float(h)
        self.deltaT_grad   = (deltaT_top - deltaT_bottom) / h

    def fixed_end_moments(self, L: float,
                          E: float = None, A: float = None,
                          I: float = None) -> np.ndarray:
        """
        Purpose  : Compute fixed-end forces from temperature gradient.
        Inputs   : L – member length [m]
                   E – Young's modulus [kN/m²]
                   A – cross-sectional area [m²]
                   I – second moment of area [m⁴]
        Outputs  : numpy array [N1, V1, M1, N2, V2, M2]
        """
        if E is not None and A is not None and I is not None:
            N = E * A * self.alpha * self.deltaT          # axial
            M = E * I * self.alpha * self.deltaT_grad     # bending
        else:
            N = self.alpha * self.deltaT * L
            M = self.alpha * self.deltaT_grad * L**2
        return np.array([-N, 0.0, -M, N, 0.0, M])

    def __repr__(self):
        return (f"BeamThermalLoad({self.name}: "
                f"alpha={self.alpha:.2e}, "
                f"dT_top={self.deltaT_top}°C, "
                f"dT_bot={self.deltaT_bottom}°C, "
                f"h={self.h}m)")
