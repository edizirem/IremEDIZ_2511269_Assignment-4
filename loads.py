"""
================================================================================
Module      : elements.py
Purpose     : Element class hierarchy for Assignment 3.
              Element (Abstract) → Members (Abstract) → Frame, Truss

              KEY NEW FEATURE — Moment Releases (Frame only):
                A moment release at a node means M=0 at that end.
                Implementation: static condensation of the released rz DOF
                from the 6x6 local stiffness matrix.

              Static condensation procedure:
                If release at end i (rz_i = DOF index 2 in local):
                  Partition k into kept (kk) and released (rr) DOFs.
                  k_condensed = k_kk - k_kr * (1/k_rr) * k_rk
                  f_condensed = f_kk - k_kr * (1/k_rr) * f_rr
                The condensed matrix is 5x5 (one DOF eliminated per release).
                For two releases: 4x4.

Units       : kN, m, kN·m
================================================================================
"""

import numpy as np
import math


# ==============================================================================
# Abstract Element
# ==============================================================================

class Element:
    """Abstract base: Id/Name, isA(), Assemble()"""

    def __init__(self, name):
        self.name = name

    def isA(self, element_type) -> bool:
        """Check element type."""
        return isinstance(self, element_type)

    def Assemble(self, K: np.ndarray, F: np.ndarray):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


# ==============================================================================
# Abstract Members
# ==============================================================================

class Members(Element):
    """Abstract: Nodes(2), MemberLoads, Assemble()"""

    def __init__(self, name, node_i, node_j, material, section):
        super().__init__(name)
        self.nodes        = [node_i, node_j]
        self.member_loads = []
        self.material     = material
        self.section      = section
        self._compute_geometry()
        node_i.members.append(self)
        node_j.members.append(self)

    def _compute_geometry(self):
        """Compute element length and direction cosines."""
        dx    = self.nodes[1].x - self.nodes[0].x
        dy    = self.nodes[1].y - self.nodes[0].y
        self.L  = math.sqrt(dx*dx + dy*dy)
        if self.L < 1e-12:
            raise ValueError(f"Element '{self.name}' has zero length.")
        self.cx = dx / self.L
        self.cy = dy / self.L

    def add_member_load(self, load):
        """Attach a MemberLoad."""
        self.member_loads.append(load)

    def G_vector(self) -> list:
        raise NotImplementedError

    def fixed_end_vector_global(self) -> np.ndarray:
        raise NotImplementedError


# ==============================================================================
# Frame Element  (with moment releases)
# ==============================================================================

class Frame(Members):
    """
    2D Euler-Bernoulli frame element with optional moment releases.

    Parameters:
        release_start (bool) – moment release at start node (M1=0)
        release_end   (bool) – moment release at end node   (M2=0)

    When a release is active, the corresponding rz DOF is condensed out
    of the local stiffness matrix using static condensation.
    This means the released moment = 0 is enforced internally.

    DOF order (local): [u1, v1, rz1, u2, v2, rz2]
    DOF indices:          0   1    2   3   4    5
    Release flags map to indices 2 (start) and 5 (end).
    """

    NDOF = 6

    def __init__(self, name, node_i, node_j, material, section,
                 release_start: bool = False, release_end: bool = False):
        """
        Inputs:
            release_start – True = moment release at start node
            release_end   – True = moment release at end node
        """
        super().__init__(name, node_i, node_j, material, section)
        self.release_start = release_start
        self.release_end   = release_end

    # ── Local stiffness (full 6x6) ──────────────────────────────────────────

    def _local_stiffness_full(self) -> np.ndarray:
        """
        Purpose : Full 6x6 local stiffness matrix (no releases).
        Outputs : numpy array (6,6)
        """
        E, A, I, L = (self.material.E, self.section.A,
                      self.section.I, self.L)
        ea   = E * A / L
        ei2  = 2  * E * I / L
        ei4  = 4  * E * I / L
        ei6  = 6  * E * I / L**2
        ei12 = 12 * E * I / L**3

        return np.array([
            [ ea,    0,    0,  -ea,    0,    0],
            [  0, ei12,  ei6,    0,-ei12,  ei6],
            [  0,  ei6,  ei4,    0,  -ei6, ei2],
            [-ea,    0,    0,   ea,    0,    0],
            [  0,-ei12, -ei6,    0, ei12, -ei6],
            [  0,  ei6,  ei2,    0,  -ei6, ei4],
        ])

    def _condense(self, k: np.ndarray, f: np.ndarray,
                  release_dofs: list) -> tuple:
        """
        Purpose  : Apply static condensation to eliminate released DOFs.
                   Released DOF has zero moment → M=0 constraint.
        Inputs   : k            – 6x6 local stiffness
                   f            – 6x1 fixed-end force vector
                   release_dofs – list of local DOF indices to condense (2, 5)
        Outputs  : (k_c, f_c, keep_dofs) where k_c and f_c are condensed,
                   keep_dofs is the list of remaining DOF indices
        Algorithm:
                   Partition: keep = all DOFs not in release_dofs
                   k_kk = k[keep,:][:,keep]
                   k_kr = k[keep,:][:,release_dofs]
                   k_rr = k[release_dofs,:][:,release_dofs]
                   k_c  = k_kk - k_kr @ inv(k_rr) @ k_rk
                   f_c  = f[keep] - k_kr @ inv(k_rr) @ f[release_dofs]
        """
        all_dofs  = list(range(6))
        keep_dofs = [d for d in all_dofs if d not in release_dofs]

        k_kk = k[np.ix_(keep_dofs, keep_dofs)]
        k_kr = k[np.ix_(keep_dofs, release_dofs)]
        k_rr = k[np.ix_(release_dofs, release_dofs)]
        k_rk = k[np.ix_(release_dofs, keep_dofs)]

        f_k  = f[keep_dofs]
        f_r  = f[release_dofs]

        # Condensation
        k_rr_inv = np.linalg.inv(k_rr)
        k_c      = k_kk - k_kr @ k_rr_inv @ k_rk
        f_c      = f_k  - k_kr @ k_rr_inv @ f_r

        return k_c, f_c, keep_dofs

    def local_stiffness(self) -> np.ndarray:
        """
        Purpose  : Local stiffness matrix accounting for releases.
                   Returns full 6x6 if no releases.
                   Returns condensed matrix (5x5 or 4x4) embedded back
                   into a 6x6 with zeros at released DOF rows/columns
                   if releases are active.
        Outputs  : numpy array (6,6)
        """
        k = self._local_stiffness_full()

        release_dofs = []
        if self.release_start:
            release_dofs.append(2)   # rz at start (local DOF index 2)
        if self.release_end:
            release_dofs.append(5)   # rz at end   (local DOF index 5)

        if not release_dofs:
            return k

        # Dummy zero FEF for condensation of stiffness only
        f_dummy = np.zeros(6)
        k_c, _, keep_dofs = self._condense(k, f_dummy, release_dofs)

        # Embed condensed k back into 6x6 (zeros at released rows/cols)
        k_full = np.zeros((6, 6))
        for i, ki in enumerate(keep_dofs):
            for j, kj in enumerate(keep_dofs):
                k_full[ki, kj] = k_c[i, j]

        return k_full

    def fixed_end_vector_local(self) -> np.ndarray:
        """
        Purpose  : Fixed-end force vector in local coordinates,
                   accounting for moment releases.
                   Thermal loads receive E, A, I for force computation.
        Outputs  : numpy array (6,)
        """
        from loads import AxialThermalLoad, BeamThermalLoad
        f = np.zeros(6)
        E = self.material.E
        A = self.section.A
        I = self.section.I
        for load in self.member_loads:
            if isinstance(load, BeamThermalLoad):
                f += load.fixed_end_moments(self.L, E=E, A=A, I=I)
            elif isinstance(load, AxialThermalLoad):
                f += load.fixed_end_moments(self.L, E=E, A=A)
            else:
                f += load.fixed_end_moments(self.L)

        release_dofs = []
        if self.release_start:
            release_dofs.append(2)
        if self.release_end:
            release_dofs.append(5)

        if not release_dofs:
            return f

        k = self._local_stiffness_full()
        _, f_c, keep_dofs = self._condense(k, f, release_dofs)

        # Embed condensed f back into 6-vector
        f_full = np.zeros(6)
        for i, ki in enumerate(keep_dofs):
            f_full[ki] = f_c[i]

        return f_full

    # ── Rotation matrix ──────────────────────────────────────────────────────

    def rotation_matrix(self) -> np.ndarray:
        """
        Purpose  : 6x6 rotation matrix (local → global).
        Outputs  : numpy array (6,6)
        """
        cx, cy = self.cx, self.cy
        sx, sy = -cy, cx
        T = np.zeros((6, 6))
        for i in [0, 1]:
            base         = i * 3
            T[base,   base]   =  cx;  T[base,   base+1] = cy
            T[base+1, base]   =  sx;  T[base+1, base+1] = sy
            T[base+2, base+2] = 1.0
        return T

    # ── Global stiffness and fixed-end vector ────────────────────────────────

    def global_stiffness(self) -> np.ndarray:
        """
        Purpose  : 6x6 global stiffness: k_g = T^T * k_local * T
        Outputs  : numpy array (6,6)
        """
        T  = self.rotation_matrix()
        kl = self.local_stiffness()
        return T.T @ kl @ T

    def fixed_end_vector_global(self) -> np.ndarray:
        """
        Purpose  : Fixed-end force vector in global coordinates.
        Outputs  : numpy array (6,)
        """
        f_local = self.fixed_end_vector_local()
        T       = self.rotation_matrix()
        return T.T @ f_local

    # ── Assembly ─────────────────────────────────────────────────────────────

    def G_vector(self) -> list:
        """Return 6 global DOF indices for this element."""
        return self.nodes[0].dof_numbers + self.nodes[1].dof_numbers

    def Assemble(self, K: np.ndarray, F: np.ndarray):
        """
        Purpose  : Assemble element contributions into global K and F.
        Inputs   : K – global stiffness matrix (modified in-place)
                   F – global force vector    (modified in-place)
        """
        k_g  = self.global_stiffness()
        f_eq = self.fixed_end_vector_global()
        G    = self.G_vector()

        for p in range(self.NDOF):
            gp = G[p]
            if gp < 0:
                continue
            F[gp] += f_eq[p]
            for q in range(self.NDOF):
                gq = G[q]
                if gq < 0:
                    continue
                K[gp, gq] += k_g[p, q]

    # ── Member end forces ────────────────────────────────────────────────────

    def member_end_forces(self, U: np.ndarray) -> np.ndarray:
        """
        Purpose  : Compute local member end forces from global displacements.
        Inputs   : U – full global displacement vector (active DOFs only)
        Outputs  : numpy array [N1, V1, M1, N2, V2, M2] in local coords
        Notes    : Released moments are returned as 0 (enforced by condensation)
        """
        G        = self.G_vector()
        d_global = np.array([U[g] if g >= 0 else 0.0 for g in G])
        T        = self.rotation_matrix()
        d_local  = T @ d_global
        k_full   = self._local_stiffness_full()
        f_local  = k_full @ d_local

        # Subtract fixed-end forces (before release correction)
        from loads import AxialThermalLoad, BeamThermalLoad
        E = self.material.E; A = self.section.A; I = self.section.I
        for load in self.member_loads:
            if isinstance(load, BeamThermalLoad):
                f_local -= load.fixed_end_moments(self.L, E=E, A=A, I=I)
            elif isinstance(load, AxialThermalLoad):
                f_local -= load.fixed_end_moments(self.L, E=E, A=A)
            else:
                f_local -= load.fixed_end_moments(self.L)

        # Enforce zero moment at released ends
        if self.release_start:
            f_local[2] = 0.0
        if self.release_end:
            f_local[5] = 0.0

        return f_local


# ==============================================================================
# Truss Element
# ==============================================================================

class Truss(Members):
    """
    2D truss (bar) element — axial deformation only.
    4 DOFs: [ux1, uy1, ux2, uy2]
    No rotational DOF, no bending, no moment releases.
    """

    NDOF = 4

    def G_vector(self) -> list:
        """Return 4 global DOF indices (no rz)."""
        ni, nj = self.nodes
        return [ni.dof_numbers[0], ni.dof_numbers[1],
                nj.dof_numbers[0], nj.dof_numbers[1]]

    def local_stiffness(self) -> np.ndarray:
        """2x2 axial stiffness matrix."""
        ea_l = self.material.E * self.section.A / self.L
        return np.array([[ ea_l, -ea_l],
                         [-ea_l,  ea_l]])

    def rotation_matrix(self) -> np.ndarray:
        """2x4 transformation matrix."""
        cx, cy = self.cx, self.cy
        return np.array([[cx, cy,  0,  0],
                         [ 0,  0, cx, cy]])

    def global_stiffness(self) -> np.ndarray:
        """4x4 global stiffness: T^T * k_local * T."""
        T  = self.rotation_matrix()
        kl = self.local_stiffness()
        return T.T @ kl @ T

    def fixed_end_vector_global(self) -> np.ndarray:
        """Truss carries no transverse member loads."""
        return np.zeros(4)

    def Assemble(self, K: np.ndarray, F: np.ndarray):
        """
        Assemble truss stiffness into K.
        For thermal loads: compute equivalent nodal forces and add to F.
        """
        from loads import AxialThermalLoad
        k_g = self.global_stiffness()
        G   = self.G_vector()
        for p in range(self.NDOF):
            gp = G[p]
            if gp < 0:
                continue
            for q in range(self.NDOF):
                gq = G[q]
                if gq < 0:
                    continue
                K[gp, gq] += k_g[p, q]

        # Thermal load equivalent nodal forces for truss
        E = self.material.E; A = self.section.A
        T = self.rotation_matrix()
        for load in self.member_loads:
            if isinstance(load, AxialThermalLoad):
                fef_local = load.fixed_end_moments(self.L, E=E, A=A)
                # Only axial: local [N1, N2] → map to global 4-DOF
                f_local_4 = np.array([fef_local[0], 0.0, fef_local[3], 0.0])
                f_global  = T.T @ np.array([fef_local[0], fef_local[3]])
                # Distribute to 4 global DOFs via transformation
                f_eq = k_g @ np.zeros(4)  # placeholder
                # Direct approach: transform local axial to global
                cx, cy = self.cx, self.cy
                f_eq = np.array([fef_local[0]*cx, fef_local[0]*cy,
                                 fef_local[3]*cx, fef_local[3]*cy])
                for p in range(self.NDOF):
                    gp = G[p]
                    if gp >= 0:
                        F[gp] += f_eq[p]

    def member_end_forces(self, U: np.ndarray) -> np.ndarray:
        """
        Axial force vector [N1, N2]. Positive = tension.
        Thermal loads are subtracted from mechanical response.
        """
        from loads import AxialThermalLoad
        G        = self.G_vector()
        d_global = np.array([U[g] if g >= 0 else 0.0 for g in G])
        T        = self.rotation_matrix()
        d_local  = T @ d_global
        f_local  = self.local_stiffness() @ d_local
        E = self.material.E; A = self.section.A
        for load in self.member_loads:
            if isinstance(load, AxialThermalLoad):
                fef = load.fixed_end_moments(self.L, E=E, A=A)
                f_local -= np.array([fef[0], fef[3]])  # only N1, N2
        return f_local
