---
status: accepted
---

# SA orbitals only; state-specific natural orbitals deferred

All orbital-level quantities (overlaps, orbital roles, orbital order, CHP energies) use the SA orbitals of each calculation, even though MS-CASPT2 states can be strongly mixed and SA orbitals represent them only approximately. In the bromothiophene data (weights |U_aI|² of the reference roots in the eigenvectors of the symmetrised H_EFF) the valence states are mostly one root (largest weight 0.81–1.00 for the singlets, 0.77–1.00 for the triplets), but many core-ionised states are not: doublet states 5 and 6 have 0.56/0.18, and quartet states 12, 14, 17 and 18 are close to 50/50 between two roots. We keep SA orbitals for now because state-specific natural orbitals raise questions that are not yet settled: how to label near-degenerate natural orbitals consistently between states, which orbital energies CHP should use, which orbital order the shake-down checks should use, and how to get the final-state densities.

## Considered Options

- **SS-RASSCF orbitals** (one RASSCF optimisation per state): rejected, not just deferred. Each state then has its own orbitals and CI space, which rules out MS-CASPT2 (only SS-CASPT2 is possible).
- **State-specific natural orbitals from the SA calculation** (Pegamoid/Molcas style): the route to use when this is picked up again. For MS-CASPT2 the state densities are D_I = Σ_ab U_aI U_bI D_ab, with U the eigenvectors of the symmetrised H_EFF and D_ab the RASSCF (transition) densities; RASSI HDF5 files provide these (checked for the bromothiophene singlet and triplet valence states). Their natural orbitals stay within the SA active space.
