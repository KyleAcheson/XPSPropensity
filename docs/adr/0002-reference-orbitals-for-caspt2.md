---
status: accepted
---

# CASPT2 calculations use the reference SA-RASSCF orbitals

For a CASPT2 calculation, the orbitals, orbital energies and populations are those of the reference SA-RASSCF calculation (`mcscf_file`), and the states' dominant configurations come from the MS-mixed CI vectors, which Molcas expresses in those same reference orbitals. We never transform orbitals or CI vectors to another orbital basis. Molcas works the same way: CASPT2 solves its equations in quasi-canonical orbitals but transforms back, and orbital-level analysis (CI printout, RASSI) uses the reference orbitals.

## Considered Options

- **PT2 natural orbitals**: rejected. They diagonalise a different density, so they can reorder or mix relative to the reference orbitals, and the CI vectors (CSFs) would have to be re-expressed in them, which Molcas does not do.

## Consequences

- RASSCF and CASPT2 runs on the same data share orbitals, orbital energies and propensity values; only the state energies and dominant configurations differ, so the energy order of the states, and with it the set of shake-downs, can differ between the two.
