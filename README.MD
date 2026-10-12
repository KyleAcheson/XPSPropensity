# XPSPropensity

A module that computes approximate Dyson norms for shake-down XPS spectroscopy based on orbital overlaps only.

### Theoretical Background

The dyson norm between an $N$ electron intial state wavefunction and an $(N-1)$ final state wavefunction is,

$$
|\langle \Phi_F | \hat{a}_{\bar{c}} | \Psi_I \rangle|^2,
$$

where $\hat{a}_{\bar{c}}$ is the annahiliation operator acting on $|\Psi_I \rangle = |\psi_c \bar{\psi}_c \psi_i \bar{\psi_{k}}\rangle$
to give $\hat{a}_{\bar{c}} |\Psi_I \rangle = |\psi_c \psi_i \bar{\psi_{k}}\rangle$ and $|\Phi_F \rangle = | \phi_c \psi_i \bar{\psi}_{j} \rangle$.
Here $k > j$ and this ammounts to one example of a shake-down transition between initial and final states.

It can be shown that a handful of transition propensity rules between different configuration state functions (CSFs) can be derived. These
simplistic rules are based solely on the overlap of orbitals between the intial and final states (thus accounting for orbital relaxation).
One such rule between an intial singlet excited CSF $\Psi_I^{0,0}$ and a final doublet CSF $\Phi_F^{D1,1/2}$ is defined as,

$$
|\langle \Phi_F^{D1, 1/2} | \hat{a}_{\bar{c}} | \Psi_I^{0,0} \rangle|^2 \approx |S_{jk} + S_{ik}S_{ji}|^2.
$$

This program extracts the basis set, ci vectors, and converged orbitals from an electronic structure calculation and
compute the approximate Dyson norm based on these propensity rules for a variety of transition types.

### Initial implementation

The `src/xpsoverlaps` package starts with program-neutral records in
`model.py` and format-specific readers under `readers/openmolcas`. The
RASSCF HDF5 reader normalizes geometry, contracted basis shells, AO labels,
MO coefficients/energies/occupations, RAS partitions, and RASSCF root
energies. The CASPT2 log reader preserves every printed mixed-state
configuration and its coefficient and weight. The supplied core and valence
fixtures include CASPT2 mixed-state tables; each state stores only its
dominant printed configuration and its coefficient and weight.
The supplied TZ HDF5 files contain 239 basis functions.

`RasscfLogReader` reads RASSCF root energies and selects each root's
highest-weight configuration from the printed CI-coefficient table in the
RASSCF log.

`Caspt2Hdf5Reader` is a separate reader for CASPT2 HDF5 results. It normalizes
the same geometry, basis, orbital, and active-space fields as the RASSCF HDF5
reader, while taking state energies from `STATE_PT2_ENERGIES`.

Use the program-neutral constructor to combine an OpenMolcas HDF5 file with
its matching state/CI log:

```python
from xpsoverlaps.readers import get_calculation

calculation = get_calculation(
    program="OpenMolcas",
    calculation_type="RASSCF",
    hdf5_file="calculation.rasscf.h5",
    log_file="calculation.log",
)
```

Currently, `RASSCF` and `CASPT2` are supported for OpenMolcas. Both input
files are required; the HDF5 data supplies geometry, orbitals, basis, and
energies, while the log supplies spin/symmetry metadata and the dominant
configuration.

The records are mutable dataclasses. Molecular coordinates and orbital
coefficients are NumPy arrays; ordered orbitals and occupations use lists so
they can be reordered in place. Keep their corresponding arrays and IDs in
sync when changing an ordering.

`get_calculation` dispatches to the requested program's readers and combines
their results with the internal `attach_states` helper (both live in
`readers/loader.py`). `attach_states` associates dominant-configuration
occupation positions with the HDF5 calculation's ordered active-orbital IDs,
validates their lengths, and checks that HDF5 and log state energies agree.
HDF5 `CI_VECTORS` are not yet decoded into occupation records by the RASSCF
reader. No overlap calculations or transition analysis are performed by these
readers.

Run the focused reader tests with `pytest`. The package's current structure is:

```text
src/xpsoverlaps/
├── __init__.py
├── main.py
├── model.py
└── readers/
    ├── __init__.py
    ├── loader.py
    └── openmolcas/
        ├── __init__.py
        ├── basis.py
        ├── caspt2_log.py
        ├── hdf5_reader.py
        └── rasscf_log.py
```
