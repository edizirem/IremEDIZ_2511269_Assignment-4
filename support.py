"""
================================================================================
Module      : structure_model.py
Purpose     : StructureModel — top-level analysis class.

              Implements the professor's partitioned stiffness approach:
                [Kff | Kfr] {Uf}   {Ff}
                [Krf | Krr] {Ur} = {Fr}

              Normal case (Ur = 0):
                Kff * Uf = Ff       → solve for displacements
                Fr = Krf * Uf - Fr_eq_restrained  → support reactions

              Settlement case (Ur != 0):
                Kff * Uf = Ff - Kfr*Ur
                Fr = Krf*Uf + Krr*Ur - Fr_eq_restrained

              Key distinction:
                The equivalent nodal load vector F contains both direct
                nodal loads AND fixed-end forces from member loads.
                When computing reactions, the fixed-end force contribution
                at restrained DOFs must be subtracted to get true reactions.

Units       : kN, m, kN·m
================================================================================
"""

import numpy as np


class StructureModel:
    """Top-level structural model using partitioned stiffness method."""

    def __init__(self, name: str = "Structure"):
        self.name             = name
        self.nodes            = []
        self.elements         = []
        self._nf              = 0
        self._nr              = 0
        self._free_dofs       = []
        self._restrained_dofs = []
        self._n_total         = 0

    # ── Model building ───────────────────────────────────────────────────────

    def add_node(self, node):
        self.nodes.append(node)

    def add_element(self, element):
        self.elements.append(element)

    # ── DOF numbering ────────────────────────────────────────────────────────

    def numberDof(self) -> tuple:
        """
        Purpose  : Assign global DOF numbers.
                   Free DOFs numbered first (0..nf-1),
                   restrained DOFs after (nf..nf+nr-1).
                   This makes Kff the top-left block of K.
        Outputs  : (nf, nr)
        """
        self._free_dofs       = []
        self._restrained_dofs = []
        free_list             = []
        restrained_list       = []

        for node in self.nodes:
            restrained = node.restrained_local_dofs()
            for local_dof in range(3):
                if local_dof in restrained:
                    restrained_list.append((node, local_dof))
                else:
                    free_list.append((node, local_dof))

        counter = 0
        for node, local_dof in free_list:
            node.dof_numbers[local_dof] = counter
            self._free_dofs.append(counter)
            counter += 1

        self._nf = counter

        for node, local_dof in restrained_list:
            node.dof_numbers[local_dof] = counter
            self._restrained_dofs.append(counter)
            counter += 1

        self._nr      = len(restrained_list)
        self._n_total = counter
        return self._nf, self._nr

    # ── Assemble full K and F ────────────────────────────────────────────────

    def _assemble_full(self):
        """
        Purpose  : Build the full (nf+nr)×(nf+nr) stiffness matrix K
                   and equivalent load vector F (nodal loads + member FEF).
        Outputs  : (K_full, F_full)
        """
        n = self._n_total
        K = np.zeros((n, n))
        F = np.zeros(n)

        for elem in self.elements:
            elem.Assemble(K, F)

        for node in self.nodes:
            node.assemble_load(F)
            node.assemble_spring(K)

        return K, F

    # ── Partition ────────────────────────────────────────────────────────────

    def _partition(self, K_full: np.ndarray, F_full: np.ndarray):
        """
        Purpose  : Partition K and F into free/restrained blocks.
                   Following professor's notation:
                     Kff = free×free,    Kfr = free×restrained
                     Krf = restrained×free, Krr = restrained×restrained
                     Ff  = free force vector
                     Fr_eq = restrained force vector (FEF at restrained DOFs)
        Outputs  : Kff, Kfr, Krf, Krr, Ff, Fr_eq
        """
        f = self._free_dofs
        r = self._restrained_dofs

        Kff   = K_full[np.ix_(f, f)]
        Kfr   = K_full[np.ix_(f, r)]
        Krf   = K_full[np.ix_(r, f)]
        Krr   = K_full[np.ix_(r, r)]
        Ff    = F_full[f]
        Fr_eq = F_full[r]

        return Kff, Kfr, Krf, Krr, Ff, Fr_eq

    # ── Stability check ──────────────────────────────────────────────────────

    def stability_check(self, Kff: np.ndarray) -> tuple:
        """
        Purpose  : Check if Kff is non-singular.
                   Singular Kff → mechanism or insufficient supports.
        Inputs   : Kff
        Outputs  : (is_stable: bool, message: str)
        """
        if Kff.shape[0] == 0:
            return False, "No free DOFs — structure is fully restrained."

        rank     = np.linalg.matrix_rank(Kff, tol=1e-8)
        expected = Kff.shape[0]

        if rank < expected:
            n_mech = expected - rank
            cond   = np.linalg.cond(Kff)
            msg = (
                f"SINGULAR STIFFNESS MATRIX — Structure is a mechanism.\n"
                f"  Free DOFs : {expected}   Rank of Kff : {rank}\n"
                f"  Independent mechanisms : {n_mech}\n"
                f"  Condition number       : {cond:.2e}\n"
                f"  Likely cause: insufficient supports, too many moment\n"
                f"  releases forming a hinge chain, or truss node with\n"
                f"  unrestrained rotational DOF."
            )
            return False, msg

        cond = np.linalg.cond(Kff)
        if cond > 1e12:
            return True, (
                f"WARNING — Kff nearly singular (cond={cond:.2e}).\n"
                f"  Results may be unreliable. Check for near-mechanisms."
            )

        return True, f"Stable. Condition number: {cond:.2e}"

    # ── Solve ────────────────────────────────────────────────────────────────

    def solve(self, Kff: np.ndarray, Ff: np.ndarray,
              Ur: np.ndarray = None,
              Kfr: np.ndarray = None) -> np.ndarray:
        """
        Purpose  : Solve Kff*Uf = Ff  (or Ff - Kfr*Ur for settlements).
        Outputs  : Uf — free DOF displacements
        """
        rhs = Ff.copy()
        if Ur is not None and np.any(Ur != 0) and Kfr is not None:
            rhs -= Kfr @ Ur
        return np.linalg.solve(Kff, rhs)

    # ── Reactions ────────────────────────────────────────────────────────────

    def compute_reactions(self, Krf: np.ndarray, Uf: np.ndarray,
                          Fr_eq: np.ndarray,
                          Krr: np.ndarray = None,
                          Ur: np.ndarray = None) -> np.ndarray:
        """
        Purpose  : Compute support reactions.

        Formula (normal case, Ur=0):
            Fr = Krf * Uf - Fr_eq_restrained

        Formula (settlement, Ur!=0):
            Fr = Krf*Uf + Krr*Ur - Fr_eq_restrained

        The Fr_eq_restrained term accounts for fixed-end forces from
        member loads that act at restrained DOFs.

        Inputs   : Krf    – restrained×free stiffness block
                   Uf     – solved free DOF displacements
                   Fr_eq  – equivalent load at restrained DOFs (from F_full)
                   Krr    – restrained×restrained block (for settlements)
                   Ur     – restrained DOF displacements (None = zero)
        Outputs  : Fr – reaction vector (one value per restrained DOF)
        """
        Fr = Krf @ Uf - Fr_eq
        if Ur is not None and Krr is not None and np.any(Ur != 0):
            Fr += Krr @ Ur
        return Fr

    # ── Full displacement vector ─────────────────────────────────────────────

    def _build_U_full(self, Uf: np.ndarray,
                      Ur: np.ndarray = None) -> np.ndarray:
        """Reconstruct full displacement vector from free + restrained parts."""
        U_full = np.zeros(self._n_total)
        for i, dof in enumerate(self._free_dofs):
            U_full[dof] = Uf[i]
        if Ur is not None:
            for i, dof in enumerate(self._restrained_dofs):
                U_full[dof] = Ur[i]
        return U_full

    # ── Member end forces ────────────────────────────────────────────────────

    def member_end_forces(self, U_full: np.ndarray) -> dict:
        """Compute local end forces for all elements."""
        results = {}
        for elem in self.elements:
            if hasattr(elem, 'member_end_forces'):
                results[elem.name] = elem.member_end_forces(U_full)
        return results

    # ── Full analysis pipeline ────────────────────────────────────────────────

    def run(self, Ur: np.ndarray = None, silent: bool = False):
        """
        Purpose  : Execute the full analysis pipeline.
        Inputs   : Ur     – prescribed support settlements (optional)
                   silent – suppress console output
        Outputs  : result dict with Kff, Krf, Ff, Uf, Fr, U_full,
                             end_forces, reactions, is_stable
        """
        def log(*args, **kwargs):
            if not silent:
                print(*args, **kwargs)

        log(f"\n{'='*65}")
        log(f"  STRUCTURAL ANALYSIS — {self.name}")
        log(f"{'='*65}")

        # Step 1: DOF numbering
        nf, nr = self.numberDof()
        log(f"\n[1] DOF Numbering — {nf} free DOFs, {nr} restrained DOFs")
        log(f"  {'Node':<10} {'ux':>5} {'uy':>5} {'rz':>5}  {'Support'}")
        for node in self.nodes:
            sup = str(node.support) if node.support else "free"
            log(f"  {str(node.name):<10} "
                f"{node.dof_numbers[0]:>5}  "
                f"{node.dof_numbers[1]:>5}  "
                f"{node.dof_numbers[2]:>5}  {sup}")

        # Step 2: Assemble
        log(f"\n[2] Assembling K ({nf+nr}x{nf+nr}) and F...")
        K_full, F_full = self._assemble_full()

        # Step 3: Partition
        log(f"\n[3] Partitioning: Kff({nf}x{nf}), Krf({nr}x{nf})")
        Kff, Kfr, Krf, Krr, Ff, Fr_eq = self._partition(K_full, F_full)

        # Step 4: Stability check
        log(f"\n[4] Stability check...")
        is_stable, stab_msg = self.stability_check(Kff)
        for line in stab_msg.splitlines():
            log(f"  {line}")

        if not is_stable:
            log(f"\n{'='*65}")
            log(f"  ANALYSIS ABORTED — {self.name}")
            log(f"{'='*65}")
            return {
                'is_stable' : False,
                'message'   : stab_msg,
                'Kff'       : Kff,
                'Ff'        : Ff,
                'Uf'        : None,
                'Fr'        : None,
                'U_full'    : None,
                'end_forces': None,
                'reactions' : None,
            }

        # Step 5: Solve
        log(f"\n[5] Solving Kff * Uf = Ff...")
        Uf = self.solve(Kff, Ff, Ur, Kfr)

        # Step 6: Reactions  Fr = Krf*Uf - Fr_eq
        log(f"\n[6] Reactions: Fr = Krf*Uf - Fr_eq")
        Fr = self.compute_reactions(Krf, Uf, Fr_eq, Krr, Ur)

        # Build full displacement vector
        U_full = self._build_U_full(Uf, Ur)

        # Map reactions to node/DOF labels
        dof_labels   = ['UX', 'UY', 'RZ']
        reaction_map = {}
        log(f"\n  {'Node':<10} {'DOF':<5} {'Reaction [kN or kNm]':>22}")
        for node in self.nodes:
            for local_dof in range(3):
                gdof = node.dof_numbers[local_dof]
                if gdof in self._restrained_dofs:
                    idx = self._restrained_dofs.index(gdof)
                    rxn = Fr[idx]
                    key = f"{node.name}_{dof_labels[local_dof]}"
                    reaction_map[key] = rxn
                    log(f"  {str(node.name):<10} "
                        f"{dof_labels[local_dof]:<5} {rxn:>22.4f}")

        # Step 7: Nodal displacements
        log(f"\n[7] Nodal Displacements")
        log(f"  {'Node':<10} {'UX [m]':>14} {'UY [m]':>14} {'RZ [rad]':>14}")
        for node in self.nodes:
            vals = [U_full[node.dof_numbers[d]] for d in range(3)]
            log(f"  {str(node.name):<10} "
                f"{vals[0]:>14.6e}  {vals[1]:>14.6e}  {vals[2]:>14.6e}")

        # Step 8: Member end forces
        log(f"\n[8] Member End Forces (local coordinates)")
        log(f"  {'Member':<10} {'N1[kN]':>10} {'V1[kN]':>10} "
            f"{'M1[kNm]':>10} {'N2[kN]':>10} {'V2[kN]':>10} {'M2[kNm]':>10}")
        end_forces = self.member_end_forces(U_full)
        for name, f_loc in end_forces.items():
            if len(f_loc) == 6:
                log(f"  {str(name):<10} " +
                    "  ".join(f"{v:>10.4f}" for v in f_loc))
            else:
                log(f"  {str(name):<10}  N={f_loc[0]:>10.4f} kN (axial)")

        log(f"\n{'='*65}")

        return {
            'is_stable'  : True,
            'message'    : stab_msg,
            'K_full'     : K_full,
            'Kff'        : Kff,
            'Kfr'        : Kfr,
            'Krf'        : Krf,
            'Krr'        : Krr,
            'Ff'         : Ff,
            'Fr_eq'      : Fr_eq,
            'Uf'         : Uf,
            'Fr'         : Fr,
            'U_full'     : U_full,
            'end_forces' : end_forces,
            'reactions'  : reaction_map,
        }
