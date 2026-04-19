"""
================================================================================
Module      : main.py
Purpose     : Assignment 4 driver — Q2 structures (a) and (b) + Q3 structures.
              Q2(a): Mixed frame+truss with support settlement
              Q2(b): Mixed frame+truss with thermal loading
              Q3:    Five structures from Assignment 3 (corrected)
Units       : kN, m, kN·m
================================================================================
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from material        import Material
from section         import Section
from support         import Fixed, Pin, Roller, Support
from loads           import NodalLoad, UniformLoad, AxialThermalLoad, BeamThermalLoad
from node            import Node
from elements        import Frame, Truss
from structure_model import StructureModel


# ── Custom support types ──────────────────────────────────────────────────────

class TrussRoller(Support):
    """Roller for truss nodes: uy and rz restrained, ux free."""
    def __init__(self, name="TrussRoller"):
        super().__init__(name)
        self.restraints = [False, True, True]

class TrussPin(Support):
    """Pin for truss nodes: ux, uy and rz all restrained."""
    def __init__(self, name="TrussPin"):
        super().__init__(name)
        self.restraints = [True, True, True]

class TrussHinge(Support):
    """Free truss node: only rz restrained."""
    def __init__(self, name="TrussHinge"):
        super().__init__(name)
        self.restraints = [False, False, True]


# ==============================================================================
# Q2(a) — Mixed Frame + Truss with Support Settlement
# ==============================================================================

def q2a_settlement():
    print("\n" + "="*65)
    print("  Q2(a): Frame+Truss — Support Settlement at E = 2mm")
    print("="*65)

    mat_conc  = Material("Concrete", E=30000e3, G=12500e3)
    mat_steel = Material("Steel",    E=200e6,   G=80e6)

    b, h = 0.5, 0.5
    sec_frame = Section("50x50cm", A=b*h, I=b*h**3/12)
    d_out=0.03; t_w=0.001; d_in=d_out-2*t_w
    A_pipe = math.pi/4*(d_out**2-d_in**2)
    sec_pipe = Section("Pipe_3cm_t1mm", A=A_pipe, I=0.0)

    nA = Node("A", 0.0,  4.0);  nA.support = Pin("Pin@A")
    nB = Node("B", 4.0,  4.0)
    nC = Node("C", 12.0, 4.0)
    nD = Node("D", 15.0, 4.0)
    nE = Node("E", 4.0,  0.0);  nE.support = Pin("Pin@E")
    nF = Node("F", 9.0,  0.0);  nF.support = TrussRoller("TrussRoller@F")

    mAB = Frame("AB", nA, nB, mat_conc, sec_frame)
    mBC = Frame("BC", nB, nC, mat_conc, sec_frame)
    mCD = Frame("CD", nC, nD, mat_conc, sec_frame)
    mBE = Frame("BE", nB, nE, mat_conc, sec_frame)
    mCF = Truss("CF", nF, nC, mat_steel, sec_pipe)

    model = StructureModel("Q2(a) Frame+Truss — Settlement")
    for nd in [nA, nB, nC, nD, nE, nF]:       model.add_node(nd)
    for el in [mAB, mBC, mCD, mBE, mCF]:       model.add_element(el)

    nf, nr = model.numberDof()
    Ur = np.zeros(nr)
    Ur[model._restrained_dofs.index(nE.dof_numbers[1])] = -0.002

    return model.run(Ur=Ur)


# ==============================================================================
# Q2(b) — Mixed Frame + Truss with Thermal Loading
# ==============================================================================

def q2b_thermal():
    print("\n" + "="*65)
    print("  Q2(b): Frame+Truss — Thermal Loading (Tb=+50°C)")
    print("="*65)

    mat_conc  = Material("Concrete", E=30000e3, G=12500e3)
    mat_steel = Material("Steel",    E=200e6,   G=80e6)

    b_c, h_c = 0.40, 0.80
    sec_frame = Section("40x80cm", A=b_c*h_c, I=b_c*h_c**3/12)
    d_out=0.03; t_w=0.001; d_in=d_out-2*t_w
    A_pipe = math.pi/4*(d_out**2-d_in**2)
    sec_pipe = Section("Pipe_3cm_t1mm", A=A_pipe, I=0.0)

    nA = Node("A", 0.0,  8.0);  nA.support = Fixed("Fixed@A")
    nB = Node("B", 7.0,  8.0)
    nC = Node("C", 16.0, 8.0);  nC.support = Pin("Pin@C")
    nD = Node("D", 3.0,  0.0);  nD.support = TrussPin("TrussPin@D")
    nE = Node("E", 13.0, 0.0);  nE.support = TrussPin("TrussPin@E")

    mAB = Frame("AB", nA, nB, mat_conc, sec_frame)
    mBC = Frame("BC", nB, nC, mat_conc, sec_frame)
    mBD = Truss("BD", nB, nD, mat_steel, sec_pipe)
    mBE = Truss("BE", nB, nE, mat_steel, sec_pipe)

    dT_bot = 50.0
    mAB.add_member_load(BeamThermalLoad("T_AB", alpha=8e-6,
                        deltaT_top=0.0, deltaT_bottom=dT_bot, h=h_c))
    mBC.add_member_load(BeamThermalLoad("T_BC", alpha=8e-6,
                        deltaT_top=0.0, deltaT_bottom=dT_bot, h=h_c))
    mBD.add_member_load(AxialThermalLoad("T_BD", alpha=1.2e-5, deltaT=dT_bot))
    mBE.add_member_load(AxialThermalLoad("T_BE", alpha=1.2e-5, deltaT=dT_bot))

    model = StructureModel("Q2(b) Frame+Truss — Thermal")
    for nd in [nA, nB, nC, nD, nE]:       model.add_node(nd)
    for el in [mAB, mBC, mBD, mBE]:       model.add_element(el)

    return model.run()


# ==============================================================================
# Q3 STRUCTURES (from Assignment 3, corrected)
# ==============================================================================

def std_mat():  return Material("Steel", E=200000.0, G=80000.0)
def std_sec():  return Section("S1", A=0.02, I=8e-4)
def tr_sec():   return Section("Tr", A=0.01, I=0.0)


# Q3(a) — Portal frame, 2 rollers → MECHANISM
def q3a_portal_frame():
    print("\n" + "="*65)
    print("  Q3(a): Portal Frame — 2 Roller Supports → MECHANISM")
    print("="*65)
    mat = std_mat(); sec = std_sec()
    n1=Node("N1",0,0); n1.support=Roller("Roller@N1")
    n2=Node("N2",0,4)
    n3=Node("N3",4,4)
    n4=Node("N4",4,0); n4.support=Roller("Roller@N4")
    n2.NodalLoad(NodalLoad("H", Fx=10.0, Fy=0.0))
    model = StructureModel("Q3(a) Portal Frame")
    for nd in [n1,n2,n3,n4]: model.add_node(nd)
    for el in [Frame("C1",n1,n2,mat,sec),
               Frame("B",n2,n3,mat,sec),
               Frame("C2",n4,n3,mat,sec)]: model.add_element(el)
    res = model.run()
    if not res['is_stable']:
        print("  REASON: Rollers resist only uy → no horizontal restraint → sway mode.")
    return res


# Q3(b1) — Fixed column → STABLE
def q3b1_fixed_column():
    print("\n" + "="*65)
    print("  Q3(b1): Cantilever Column — Fixed Base → STABLE")
    print("="*65)
    mat = std_mat(); sec = std_sec()
    n1=Node("N1",0,0); n1.support=Fixed("Fixed@N1")
    n2=Node("N2",0,5)
    n2.NodalLoad(NodalLoad("H", Fx=10.0, Fy=0.0))
    model = StructureModel("Q3(b1)")
    model.add_node(n1); model.add_node(n2)
    model.add_element(Frame("Col",n1,n2,mat,sec))
    res = model.run()
    if res['is_stable']:
        P,L,E,I = 10,5,200000,8e-4
        print(f"  Hand: delta=PL^3/3EI={P*L**3/(3*E*I):.4f}m  M_base=PL={P*L:.4f}kN.m")
    return res


# Q3(b2) — Free column, no support → MECHANISM
def q3b2_free_column():
    print("\n" + "="*65)
    print("  Q3(b2): Free Column — No Support → MECHANISM")
    print("="*65)
    mat = std_mat(); sec = std_sec()
    n1=Node("N1",0,0); n2=Node("N2",0,5)
    n2.NodalLoad(NodalLoad("H", Fx=10.0, Fy=0.0))
    model = StructureModel("Q3(b2)")
    model.add_node(n1); model.add_node(n2)
    model.add_element(Frame("Col",n1,n2,mat,sec))
    res = model.run()
    if not res['is_stable']:
        print("  REASON: No supports → 3 rigid body modes → singular Kff.")
    return res


# Q3(c) — TWO DISCONNECTED SUBSTRUCTURES (CORRECTED)
def q3c_two_substructures():
    print("\n" + "="*65)
    print("  Q3(c): TWO Disconnected Substructures (CORRECTED)")
    print("="*65)

    mat = std_mat(); sec = std_sec()

    # Left: L-frame (fixed + roller)
    print("\n  [Left substructure] L-Frame — Fixed base + Roller")
    n1=Node("N1",0,0); n1.support=Fixed("Fixed@N1")
    n2=Node("N2",0,4)
    n3=Node("N3",4,4); n3.support=Roller("Roller@N3")
    n2.NodalLoad(NodalLoad("H", Fx=10.0, Fy=0.0))
    m_left = StructureModel("Q3(c) Left — L-Frame")
    for nd in [n1,n2,n3]: m_left.add_node(nd)
    for el in [Frame("Col",n1,n2,mat,sec),
               Frame("Beam",n2,n3,mat,sec)]: m_left.add_element(el)
    res_left = m_left.run()

    # Right: free-standing column (no load, no connection to left)
    print("\n  [Right substructure] Free-Standing Column — No load")
    mat2=std_mat(); sec2=std_sec()
    n4=Node("N4",6,0); n4.support=Fixed("Fixed@N4")
    n5=Node("N5",6,4)
    m_right = StructureModel("Q3(c) Right — Free Column")
    m_right.add_node(n4); m_right.add_node(n5)
    m_right.add_element(Frame("Col2",n4,n5,mat2,sec2))
    res_right = m_right.run()
    if res_right['is_stable']:
        print("  Result: STABLE — all forces = 0 (no load applied, trivial solution)")

    return res_left, res_right


# Q3(d) — 2-bar truss → STABLE
def q3d_two_bar_truss():
    print("\n" + "="*65)
    print("  Q3(d): 2-Bar Truss — 2 Pins + Hinge at Top → STABLE")
    print("="*65)
    mat = std_mat(); sec = tr_sec()
    L=5; COS45=math.cos(math.radians(45)); SIN45=math.sin(math.radians(45))
    n1=Node("N1",-L*COS45,0); n1.support=TrussPin()
    n2=Node("N2", L*COS45,0); n2.support=TrussPin()
    n3=Node("N3",0,L*SIN45);  n3.support=TrussHinge()
    n3.NodalLoad(NodalLoad("H", Fx=10.0, Fy=0.0))
    model = StructureModel("Q3(d) 2-Bar Truss")
    for nd in [n1,n2,n3]: model.add_node(nd)
    for el in [Truss("T1",n1,n3,mat,sec),
               Truss("T2",n3,n2,mat,sec)]: model.add_element(el)
    res = model.run()
    if res['is_stable']:
        print(f"  Hand: N=P/(2cos45)={10/(2*COS45):.4f}kN")
    return res


# Q3(e) — Beam with internal hinge — DUAL-NODE APPROACH (CORRECTED)
def q3e_beam_internal_hinge():
    print("\n" + "="*65)
    print("  Q3(e): Beam with Internal Hinge — Dual-Node Approach (CORRECTED)")
    print("  N1(pin)--M1--N2(pin)--M2--[HINGE]--M3--N4(pin)")
    print("  Expected: STABLE (Gerber beam)")
    print("="*65)

    mat = std_mat(); sec = std_sec()

    n1  = Node("N1",  0.0,    0.0); n1.support  = Pin("Pin@N1")
    n2  = Node("N2",  4.0,    0.0); n2.support  = Pin("Pin@N2")
    n3L = Node("N3L", 8.0,    0.0)   # left side of hinge
    n3R = Node("N3R", 8.0001, 0.0)   # right side of hinge (tiny offset)
    n4  = Node("N4", 12.0,    0.0); n4.support  = Pin("Pin@N4")

    n3L.NodalLoad(NodalLoad("V", Fx=0.0, Fy=-10.0))

    # Regular frame members — NO releases on M2 and M3
    m1 = Frame("M1", n1,  n2,  mat, sec)
    m2 = Frame("M2", n2,  n3L, mat, sec)
    m3 = Frame("M3", n3R, n4,  mat, sec)

    # Rigid link: very stiff beam with releases at BOTH ends
    # Transmits N and V only — enforces M=0 at hinge ✓
    mat_link = Material("RigidLink", E=200000e3, G=80e6)
    sec_link  = Section("Link",      A=100.0,    I=100.0)
    link = Frame("Link", n3L, n3R, mat_link, sec_link,
                 release_start=True, release_end=True)

    model = StructureModel("Q3(e) Beam with Internal Hinge")
    for nd in [n1, n2, n3L, n3R, n4]:  model.add_node(nd)
    for el in [m1, m2, link, m3]:       model.add_element(el)

    res = model.run()

    if res['is_stable']:
        ef   = res['end_forces']
        rxns = res['reactions']
        R1 = rxns.get('N1_UY', 0)
        R2 = rxns.get('N2_UY', 0)
        R4 = rxns.get('N4_UY', 0)
        print(f"\n  Hinge verification:")
        print(f"    Link M_start = {ef['Link'][2]:.6f} kN.m  (should be 0)")
        print(f"    Link M_end   = {ef['Link'][5]:.6f} kN.m  (should be 0)")
        print(f"\n  Hand: R1=-10.0, R2=+20.0, R4=0.0 kN")
        print(f"  FEM:  R1={R1:.4f}, R2={R2:.4f}, R4={R4:.4f} kN")
        print(f"  SFy = {R1+R2+R4-10:.6f} kN  (should be 0) OK")

    return res


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  ASSIGNMENT 4 — FULL ANALYSIS")
    print("="*65)

    # Q2
    res_2a = q2a_settlement()
    res_2b = q2b_thermal()

    # Q3
    res_3a        = q3a_portal_frame()
    res_3b1       = q3b1_fixed_column()
    res_3b2       = q3b2_free_column()
    res_3c_l, res_3c_r = q3c_two_substructures()
    res_3d        = q3d_two_bar_truss()
    res_3e        = q3e_beam_internal_hinge()

    print("\n\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print("  Q2:")
    for key,res,lbl in [('2a',res_2a,'Frame+Truss — 2mm settlement'),
                         ('2b',res_2b,'Frame+Truss — thermal Tb=+50C')]:
        st = "STABLE   OK" if res.get('is_stable') else "MECHANISM"
        print(f"    Q{key}: {st}  — {lbl}")
    print("  Q3:")
    q3_results = [
        ('3a',  res_3a,   'Portal frame — 2 rollers'),
        ('3b1', res_3b1,  'Fixed column — free top'),
        ('3b2', res_3b2,  'Free column — no support'),
        ('3c_L',res_3c_l, 'L-frame (left substructure)'),
        ('3c_R',res_3c_r, 'Free column (right substructure)'),
        ('3d',  res_3d,   '2-bar truss — 2 pins + hinge'),
        ('3e',  res_3e,   'Beam with internal hinge (dual-node)'),
    ]
    for key,res,lbl in q3_results:
        st = "STABLE   OK" if res.get('is_stable') else "MECHANISM"
        print(f"    Q{key}: {st}  — {lbl}")
