"""
================================================================================
Module      : node.py
Purpose     : Node class — Id/Name, XYZ(3), DofNumbers(3), Support,
              NodalLoad(), Members(), Spring()
Units       : m [coordinates], kN [forces]
================================================================================
"""

import numpy as np


class Node:
    """2D structural node with 3 DOFs: [ux, uy, rz]."""

    def __init__(self, name, x: float, y: float):
        """
        Inputs:
            name – node ID or label
            x    – X coordinate [m]
            y    – Y coordinate [m]
        """
        self.name        = name
        self.xyz         = np.array([float(x), float(y), 0.0])
        self.dof_numbers = [-1, -1, -1]    # set by numberDof()
        self.support     = None
        self.nodal_load  = None
        self.members     = []
        self.spring      = np.array([0.0, 0.0, 0.0])   # [kx, ky, kr]

    @property
    def x(self): return self.xyz[0]

    @property
    def y(self): return self.xyz[1]

    def Spring(self, kx: float = 0.0, ky: float = 0.0, kr: float = 0.0):
        """Assign spring stiffnesses [kN/m, kN/m, kN·m/rad]."""
        self.spring = np.array([float(kx), float(ky), float(kr)])

    def NodalLoad(self, load):
        """Assign a NodalLoad object."""
        self.nodal_load = load

    def Members(self) -> list:
        """Return list of connected members."""
        return self.members

    def is_restrained(self) -> bool:
        return self.support is not None

    def restrained_local_dofs(self) -> list:
        if self.support is None:
            return []
        return self.support.restrained_dofs()

    def assemble_load(self, F: np.ndarray):
        """Add nodal load into global F."""
        if self.nodal_load is None:
            return
        load_vec = self.nodal_load.vector()
        for local_dof in range(3):
            gdof = self.dof_numbers[local_dof]
            if gdof >= 0:
                F[gdof] += load_vec[local_dof]

    def assemble_spring(self, K: np.ndarray):
        """Add spring stiffness into global K diagonal."""
        for local_dof in range(3):
            gdof = self.dof_numbers[local_dof]
            if gdof >= 0:
                K[gdof, gdof] += self.spring[local_dof]

    def __repr__(self):
        return (f"Node({self.name}: x={self.x:.3f}, y={self.y:.3f}, "
                f"dofs={self.dof_numbers})")
