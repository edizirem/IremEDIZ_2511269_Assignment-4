"""
================================================================================
Module      : test_interface_assembly.py
Purpose     : Interface tests for global matrix assembly (Kff/Krf partitioning).
              Tests that StructureModel correctly assembles and partitions K,F
              using the professor's formulation.
Tolerance   : 1e-6
Run         : python tests/test_interface_assembly.py
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from material        import Material
from section         import Section
from support         import Fixed, Pin, Roller
from loads           import NodalLoad, UniformLoad
from node            import Node
from elements        import Frame, Truss
from structure_model import StructureModel

TOL = 1e-6

def make_cantilever():
    """Single fixed-free cantilever, L=4m, P=10kN horizontal at tip."""
    mat=Material("S",E=200000,G=80000); sec=Section("S",A=0.02,I=8e-4)
    n1=Node("N1",0,0); n1.support=Fixed("F")
    n2=Node("N2",4,0); n2.NodalLoad(NodalLoad("P",Fx=10,Fy=0))
    col=Frame("C",n1,n2,mat,sec)
    m=StructureModel("Cantilever")
    m.add_node(n1); m.add_node(n2); m.add_element(col)
    return m, n1, n2, col

def make_two_span():
    """Two-span continuous beam with central support."""
    mat=Material("S",E=200000,G=80000); sec=Section("S",A=0.02,I=8e-4)
    n1=Node("N1",0,0); n1.support=Pin("P1")
    n2=Node("N2",3,0)
    n3=Node("N3",6,0); n3.support=Roller("R")
    n2.NodalLoad(NodalLoad("V",Fy=-50))
    b1=Frame("B1",n1,n2,mat,sec); b2=Frame("B2",n2,n3,mat,sec)
    m=StructureModel("TwoSpan")
    for nd in [n1,n2,n3]: m.add_node(nd)
    for el in [b1,b2]: m.add_element(el)
    return m, n1, n2, n3

# ── Interface Test 1: DOF numbering ──────────────────────────────────────────

def test_interface_dof_numbering_free_first():
    """
    Free DOFs must be numbered before restrained DOFs (0..nf-1 first).
    This ensures Kff occupies the top-left block of K.
    """
    m, n1, n2, _ = make_cantilever()
    nf, nr = m.numberDof()
    # n2 is free → DOFs 0,1,2; n1 is fixed → DOFs 3,4,5
    assert nf == 3, f"Expected 3 free DOFs, got {nf}"
    assert nr == 3, f"Expected 3 restrained DOFs, got {nr}"
    # All free DOFs must be numbered < nf
    for dof in m._free_dofs:
        assert dof < nf, f"Free DOF {dof} >= nf={nf}"
    # All restrained DOFs must be numbered >= nf
    for dof in m._restrained_dofs:
        assert dof >= nf, f"Restrained DOF {dof} < nf={nf}"
    print("PASS  test_interface_dof_numbering_free_first")


def test_interface_dof_count_correct():
    """Total DOFs = 3 * num_nodes for frame structures."""
    m, *_ = make_two_span()
    nf, nr = m.numberDof()
    assert nf + nr == 3 * 3, f"Total DOFs {nf+nr} != 9"
    print("PASS  test_interface_dof_count_correct")


def test_interface_dof_restrained_match_support():
    """Restrained DOF count must match total restrained directions in supports."""
    m, n1, n2, n3 = make_two_span()
    nf, nr = m.numberDof()
    # n1=Pin(ux,uy)=2, n3=Roller(uy)=1 → nr=3
    assert nr == 3, f"Expected 3 restrained DOFs, got {nr}"
    assert n1.dof_numbers[0] >= nf, "N1.ux should be restrained"
    assert n1.dof_numbers[1] >= nf, "N1.uy should be restrained"
    assert n3.dof_numbers[1] >= nf, "N3.uy should be restrained"
    print("PASS  test_interface_dof_restrained_match_support")


# ── Interface Test 2: Kff assembly ───────────────────────────────────────────

def test_interface_Kff_shape():
    """Kff must be (nf x nf)."""
    m, *_ = make_cantilever()
    nf, nr = m.numberDof()
    K, F = m._assemble_full()
    Kff, Kfr, Krf, Krr, Ff, Fr_eq = m._partition(K, F)
    assert Kff.shape == (nf, nf), f"Kff shape {Kff.shape} != ({nf},{nf})"
    print("PASS  test_interface_Kff_shape")


def test_interface_Kff_symmetric():
    """Kff must be symmetric (K is symmetric → subblock is too)."""
    m, *_ = make_cantilever()
    m.numberDof()
    K, F = m._assemble_full()
    Kff, *_ = m._partition(K, F)
    assert np.allclose(Kff, Kff.T, atol=TOL), "Kff not symmetric"
    print("PASS  test_interface_Kff_symmetric")


def test_interface_Kff_positive_definite():
    """Kff must be positive definite for a stable structure."""
    m, *_ = make_cantilever()
    m.numberDof()
    K, F = m._assemble_full()
    Kff, *_ = m._partition(K, F)
    eigvals = np.linalg.eigvalsh(Kff)
    assert np.all(eigvals > 0), \
        f"Kff not positive definite, min eigenvalue = {eigvals.min():.4e}"
    print("PASS  test_interface_Kff_positive_definite")


def test_interface_Krf_shape():
    """Krf must be (nr x nf)."""
    m, *_ = make_cantilever()
    nf, nr = m.numberDof()
    K, F = m._assemble_full()
    _, _, Krf, *_ = m._partition(K, F)
    assert Krf.shape == (nr, nf), f"Krf shape {Krf.shape} != ({nr},{nf})"
    print("PASS  test_interface_Krf_shape")


def test_interface_Krf_equals_Kff_transpose_block():
    """Krf must equal Kfr.T (K is symmetric → Krf = Kfr^T)."""
    m, *_ = make_cantilever()
    m.numberDof()
    K, F = m._assemble_full()
    Kff, Kfr, Krf, *_ = m._partition(K, F)
    assert np.allclose(Krf, Kfr.T, atol=TOL), "Krf != Kfr.T"
    print("PASS  test_interface_Krf_equals_Kff_transpose_block")


# ── Interface Test 3: Force vector assembly ───────────────────────────────────

def test_interface_Ff_nodal_load_correct():
    """Nodal load Fx=10 at free node must appear correctly in Ff."""
    m, n1, n2, _ = make_cantilever()
    m.numberDof()
    K, F = m._assemble_full()
    _, _, _, _, Ff, _ = m._partition(K, F)
    # n2.ux is a free DOF → its entry in Ff should be 10
    ux_dof = n2.dof_numbers[0]
    idx_in_ff = m._free_dofs.index(ux_dof)
    assert abs(Ff[idx_in_ff] - 10.0) < TOL, \
        f"Ff[{idx_in_ff}]={Ff[idx_in_ff]:.4f}, expected 10.0"
    print("PASS  test_interface_Ff_nodal_load_correct")


def test_interface_member_load_added_to_F():
    """UDL member load must appear in assembled F (not zero)."""
    mat=Material("S",E=200000,G=80000); sec=Section("S",A=0.02,I=8e-4)
    n1=Node("N1",0,0); n1.support=Fixed("F")
    n2=Node("N2",4,0); n2.support=Roller("R")
    beam=Frame("B",n1,n2,mat,sec)
    beam.add_member_load(UniformLoad("W",w=-10))
    m=StructureModel("UDL test")
    m.add_node(n1); m.add_node(n2); m.add_element(beam)
    m.numberDof()
    K, F = m._assemble_full()
    _, _, _, _, Ff, Fr_eq = m._partition(K, F)
    # There should be non-zero entries from the UDL
    assert np.any(np.abs(F) > TOL), "F is all zero despite member load"
    print("PASS  test_interface_member_load_added_to_F")


# ── Interface Test 4: Full K assembly ─────────────────────────────────────────

def test_interface_full_K_symmetric():
    """Full assembled K must be symmetric."""
    m, *_ = make_two_span()
    m.numberDof()
    K, _ = m._assemble_full()
    assert np.allclose(K, K.T, atol=TOL), "Full K not symmetric"
    print("PASS  test_interface_full_K_symmetric")


def test_interface_full_K_size():
    """Full K must be (nf+nr) x (nf+nr)."""
    m, *_ = make_two_span()
    nf, nr = m.numberDof()
    K, _ = m._assemble_full()
    n = nf + nr
    assert K.shape == (n, n), f"K shape {K.shape} != ({n},{n})"
    print("PASS  test_interface_full_K_size")


def test_interface_mixed_element_assembly():
    """Mixed frame+truss model: K assembles without error."""
    mat=Material("S",E=200000,G=80000)
    sec=Section("S",A=0.02,I=8e-4); tr=Section("T",A=0.01)
    n1=Node("N1",0,0); n1.support=Pin("P1")
    n2=Node("N2",4,0); n2.support=Pin("P2")
    n3=Node("N3",2,3)
    from support import Support
    class TrussNode(Support):
        def __init__(self):
            super().__init__("TN"); self.restraints=[False,False,True]
    n3.support=TrussNode()
    n3.NodalLoad(NodalLoad("V",Fy=-10))
    t1=Truss("T1",n1,n3,mat,tr); t2=Truss("T2",n3,n2,mat,tr)
    m=StructureModel("Mixed")
    for nd in [n1,n2,n3]: m.add_node(nd)
    for el in [t1,t2]: m.add_element(el)
    nf,nr=m.numberDof()
    K,F=m._assemble_full()
    assert K.shape==(nf+nr,nf+nr), f"K shape wrong: {K.shape}"
    assert np.allclose(K,K.T,atol=TOL), "K not symmetric for mixed model"
    print("PASS  test_interface_mixed_element_assembly")


# ── Interface Test 5: Stability check ────────────────────────────────────────

def test_interface_stability_stable():
    """Stable structure: stability_check returns True."""
    m, *_ = make_cantilever()
    m.numberDof()
    K, F = m._assemble_full()
    Kff, *_ = m._partition(K, F)
    is_stable, msg = m.stability_check(Kff)
    assert is_stable, f"Stable structure flagged as unstable: {msg}"
    print("PASS  test_interface_stability_stable")


def test_interface_stability_mechanism_detected():
    """Unstable structure (no supports): stability_check returns False."""
    mat=Material("S",E=200000,G=80000); sec=Section("S",A=0.02,I=8e-4)
    n1=Node("N1",0,0); n2=Node("N2",4,0)
    beam=Frame("B",n1,n2,mat,sec)
    m=StructureModel("Unstable")
    m.add_node(n1); m.add_node(n2); m.add_element(beam)
    m.numberDof()
    K,F=m._assemble_full()
    Kff,*_=m._partition(K,F)
    is_stable,msg=m.stability_check(Kff)
    assert not is_stable, "Unsupported beam should be detected as mechanism"
    print("PASS  test_interface_stability_mechanism_detected")


# ── Interface Test 6: Reaction computation ────────────────────────────────────

def test_interface_reactions_equilibrium():
    """
    Sum of reactions + applied loads = 0 (global equilibrium).
    Cantilever: Fx=10 applied, so Rx_fixed = -10.
    """
    m, n1, n2, _ = make_cantilever()
    r = m.run(silent=True)
    assert r['is_stable']
    # ΣFx = 0: reaction UX + applied Fx = 0
    Rx = r['reactions'].get('N1_UX', 0)
    assert abs(Rx + 10.0) < TOL, \
        f"ΣFx = {Rx + 10.0:.4e}, expected 0 (Rx={Rx:.4f})"
    print("PASS  test_interface_reactions_equilibrium")


def test_interface_reactions_formula_Krf_Uf():
    """
    Verify: Fr = Krf*Uf - Fr_eq  matches run() reactions.
    """
    m, *_ = make_two_span()
    r = m.run(silent=True)
    assert r['is_stable']
    Fr_direct = r['Fr']
    # Recompute manually
    Krf = r['Krf']; Uf = r['Uf']; Fr_eq = r['Fr_eq']
    Fr_manual = Krf @ Uf - Fr_eq
    assert np.allclose(Fr_direct, Fr_manual, atol=TOL), \
        f"Fr mismatch:\n  run={Fr_direct}\n  manual={Fr_manual}"
    print("PASS  test_interface_reactions_formula_Krf_Uf")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    tests=[
        test_interface_dof_numbering_free_first,
        test_interface_dof_count_correct,
        test_interface_dof_restrained_match_support,
        test_interface_Kff_shape,
        test_interface_Kff_symmetric,
        test_interface_Kff_positive_definite,
        test_interface_Krf_shape,
        test_interface_Krf_equals_Kff_transpose_block,
        test_interface_Ff_nodal_load_correct,
        test_interface_member_load_added_to_F,
        test_interface_full_K_symmetric,
        test_interface_full_K_size,
        test_interface_mixed_element_assembly,
        test_interface_stability_stable,
        test_interface_stability_mechanism_detected,
        test_interface_reactions_equilibrium,
        test_interface_reactions_formula_Krf_Uf,
    ]
    passed=failed=0
    print(f"\n{'='*60}\n  INTERFACE TESTS — Global Assembly  (tol={TOL})\n{'='*60}\n")
    for t in tests:
        try: t(); passed+=1
        except AssertionError as e: print(f"FAIL  {t.__name__}\n      {e}"); failed+=1
        except Exception as e: print(f"ERROR {t.__name__}\n      {type(e).__name__}: {e}"); failed+=1
    print(f"\n{'='*60}\n  {passed} passed, {failed} failed / {len(tests)} total\n{'='*60}\n")
    return failed==0

if __name__=="__main__":
    sys.exit(0 if run_all() else 1)
