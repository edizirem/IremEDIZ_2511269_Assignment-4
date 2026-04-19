# CE 4011 — Assignment 4: Thermal Loads and Support Settlements

Extension of the Assignment 3 OOP structural analysis program with two new capabilities:
- **Thermal loading** — uniform temperature change and thermal gradient
- **Support settlements** — prescribed displacements at restrained DOFs

---

## New Features

### Thermal Loads (`loads.py`)

```
MemberLoad (abstract)
├── PointLoad
├── UniformLoad
└── ThermalLoad (abstract)         ← NEW
        alpha [1/°C]
        deltaT [°C]
        ├── AxialThermalLoad        ← NEW — uniform dT, Frame and Truss
        │       N = E·A·alpha·deltaT
        │       FEV = [-N, 0, 0, +N, 0, 0]
        │
        └── BeamThermalLoad         ← NEW — gradient dT, Frame only
                deltaT_top, deltaT_bottom, h [m]
                dT_avg  = (top+bot)/2  → N = E·A·alpha·dT_avg
                dT_grad = (top-bot)/h  → M = E·I·alpha·dT_grad
                FEV = [-N, 0, -M, +N, 0, +M]
```

### Support Settlements (`structure_model.py`)

Following the professor's partitioned stiffness formula:

```
Normal:     Kff * Uf = Ff
Settlement: Kff * Uf = Ff - Kfr * Ur
Reactions:  Fr = Krf * Uf + Krr * Ur - Fr_applied
```

Pass `Ur` vector to `model.run(Ur=Ur)`.

---

## File Structure

```
assignment4/
├── material.py              # Material (E, G)
├── section.py               # Section (A, I)
├── support.py               # Fixed, Pin, Roller, FixedRoller, XRoller
├── loads.py                 # NodalLoad, PointLoad, UniformLoad,
│                            # AxialThermalLoad, BeamThermalLoad  ← NEW
├── node.py                  # Node (XYZ, DOFs, spring, load)
├── elements.py              # Frame (with releases), Truss
├── structure_model.py       # StructureModel (Kff/Krf, settlements) ← UPDATED
├── main.py                  # Q2(a) settlement + Q2(b) thermal
└── tests/
    ├── test_unit_frame.py              # Frame element unit tests
    ├── test_interface_assembly.py      # Assembly interface tests
    ├── test_regression.py              # Regression tests
    └── test_thermal_settlement.py      # NEW: thermal + settlement tests
```

---

## How to Run

### Requirements
```
Python >= 3.9
numpy
```

### Run Q2 analyses
```bash
cd assignment4
python main.py
```

### Run all tests
```bash
python tests/test_unit_frame.py
python tests/test_interface_assembly.py
python tests/test_regression.py
python tests/test_thermal_settlement.py
```

---

## Usage Examples

### Thermal load — uniform temperature change
```python
# Works for both Frame and Truss elements
member.add_member_load(AxialThermalLoad("T", alpha=12e-6, deltaT=50.0))
```

### Thermal load — temperature gradient (Frame only)
```python
# Top face hotter than bottom (or vice versa)
beam.add_member_load(BeamThermalLoad("TG",
    alpha=8e-6,
    deltaT_top=0.0,       # top face temperature
    deltaT_bottom=50.0,   # bottom face temperature
    h=0.80))              # section depth [m]
```

### Support settlement
```python
model = StructureModel("My Structure")
# ... add nodes, elements ...

nf, nr = model.numberDof()
Ur = np.zeros(nr)

# Find the restrained DOF index for uy at the settling node
uy_gdof = settling_node.dof_numbers[1]
idx = model._restrained_dofs.index(uy_gdof)
Ur[idx] = -0.002   # 2mm downward settlement

results = model.run(Ur=Ur)
```

---

## Q2 Results Summary

### Q2(a) — Frame+Truss with Support Settlement (Δ=2mm at E)

| Node | UX [mm] | UY [mm] | RZ [rad] |
|------|---------|---------|----------|
| A    | 0.0000  | 0.0000  | -6.24e-4 |
| B    | 0.0039  | -1.9961 | -2.50e-4 |
| C    | 0.0039  | -3.9961 | -2.50e-4 |
| D    | 0.0039  | -4.7461 | -2.50e-4 |
| E    | 0.0000  | -2.0000 | +1.24e-4 |
| F    | -5.3243 | 0.0000  | 0.000    |

Key results: M_B = 29.18 kN·m, CF truss force ≈ 0 (physically correct)

### Q2(b) — Frame+Truss with Thermal Load (Tb=+50°C)

| Node | UX [mm] | UY [mm]  | RZ [rad]  |
|------|---------|----------|-----------|
| A    | 0.0000  | 0.0000   | 0.000     |
| B    | -0.0006 | +3.9211  | +6.46e-4  |
| C    | 0.0000  | 0.0000   | -2.10e-3  |

Key results: N_AB ≈ 1921 kN (axial thermal), M_A = 407 kN·m, B deflects 3.92mm upward

---

## Units

| Quantity | Unit |
|----------|------|
| Length   | m    |
| Force    | kN   |
| Moment   | kN·m |
| Stress   | kN/m²|
| Temperature | °C |
| Alpha    | 1/°C |
