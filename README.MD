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

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

Requires Python ≥ 3.9 with `numpy`, `h5py`, `pyyaml` and `jsonschema`.

### Running

The program is driven by a YAML input file (see `inputs/input.yaml`):

```yaml
Calculator:
  program: molcas        # electronic-structure code
  method: overlaps       # 'overlaps' (needs Initial and Final) or 'chp' (Initial only)

Initial:
  type: caspt2           # or rasscf; must match Final
  wf_file: ../data/molcas/brthph/s2p/mb/valence_caspt2.caspt2.h5   # orbitals, basis, energies
  state_file: ../data/molcas/brthph/s2p/mb/valence_caspt2.log      # CI configurations (Molcas only)
  homo: 34               # optional, given together with lumo: 1-based MO indices (Initial only)
  lumo: 40

Final:
  type: caspt2
  wf_file: ../data/molcas/brthph/s2p/mb/core_caspt2.caspt2.h5
  state_file: ../data/molcas/brthph/s2p/mb/core_caspt2.log
```

Relative paths are resolved from the input file's directory. Choice values are
case-insensitive. `homo`/`lumo` must be two different active orbitals; choosing
the right ones is up to the user. The final-state HOMO/LUMO are the orbitals
they map onto. Run with:

```bash
python src/xpsoverlaps/main.py -i inputs/input.yaml -o output.log
# or, once installed:
xpsoverlaps -i inputs/input.yaml -o output.log
```

The propensity rules themselves are not implemented yet; `method: chp` is a
placeholder.

### Using it from Python

```python
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.overlap import mo_overlap

initial = get_calculation("molcas", "caspt2", wf_file="valence.caspt2.h5", state_file="valence.log")
final = get_calculation("molcas", "caspt2", wf_file="core.caspt2.h5", state_file="core.log")

S_MO = mo_overlap(initial, final)   # rows: initial MOs, columns: final MOs
```

Supported: OpenMolcas (`molcas`/`openmolcas`) `RASSCF` and `CASPT2`, each read
from an HDF5 file (`wf_file`) plus its log (`state_file`). Molecular symmetry
is not supported yet.

### Package structure

```text
src/xpsoverlaps/
├── main.py              # command-line entry point
├── inputs.py            # YAML input loading and validation
├── input_schema.yaml    # input schema (JSON Schema written in YAML)
├── model.py             # program-neutral data model (Calculation and its parts)
├── composition.py       # AO composition of MOs (Mulliken partition)
├── mapping.py           # initial -> final active-orbital mapping
├── frontier.py          # HOMO/LUMO from the input, carried to the final state by the mapping
├── shakedown.py         # shake-down transitions between initial states and their i, j, k roles
├── propensity.py        # propensity workflows (overlaps, chp)
├── overlap/             # AO overlap providers and MO overlaps
│   ├── base.py          #   AOOverlapProvider interface
│   ├── gaussian.py      #   integrals for GaussianBasis
│   ├── engine.py        #   integrals from a program's own engine (EngineBasis)
│   ├── providers.py     #   provider selection by basis type
│   └── mo.py            #   mo_overlap
└── readers/
    ├── loader.py        # CalculationLoader base class and get_calculation
    └── openmolcas/
        ├── hdf5_file.py     # HDF5 access layer (all Molcas format details)
        ├── hdf5_reader.py   # one reader per record: molecule, basis, orbitals, active space
        ├── rasscf_log.py    # RASSCF states from the log
        ├── caspt2_log.py    # MS-CASPT2 states from the log
        ├── basis.py         # Molcas basis-set reconstruction
        └── loader.py        # OpenMolcasLoader: reads each record and builds the Calculation
tests/                   # pytest suite (uses the data in data/molcas)
```

Run the tests with `pytest`.

### Adding a new electronic-structure program

1. **Record readers** under `readers/<program>/`: one function per model record
   (molecule, basis, orbitals, active space, states), each returning that
   record. Records that depend on others take them as arguments, e.g.
   `read_basis(file, molecule)`. Keep file-format details in one access layer
   that the readers share (see `openmolcas/hdf5_file.py`).
2. **A loader**: a subclass of `CalculationLoader`. Defining it registers it;
   `get_calculation` does not need to change. Template for a program that
   writes everything to one file (`readers/myprogram/loader.py`):

   ```python
   """MyProgram loader: every record is read from one output file."""

   from pathlib import Path
   from typing import List, Optional

   from xpsoverlaps.model import (
       ActiveSpace, AOBasis, Calculation, ElectronicState, Molecule, OrbitalSet, Provenance,
   )
   from xpsoverlaps.readers.loader import CalculationLoader


   # Record readers: one per model record; records they depend on are arguments.
   def read_molecule(path: Path) -> Molecule: ...
   def read_basis(path: Path, molecule: Molecule) -> AOBasis: ...   # GaussianBasis or EngineBasis
   def read_orbitals(path: Path, basis: AOBasis) -> OrbitalSet: ...
   def read_active_space(path: Path, orbitals: OrbitalSet) -> ActiveSpace: ...
   def read_states(path: Path) -> List[ElectronicState]: ...


   class MyProgramLoader(CalculationLoader):
       programs = ("myprogram",)          # names accepted by get_calculation (any case)
       display_name = "MyProgram"         # used in error messages
       calculation_types = {"CASSCF"}     # upper case

       def _load(self, kind: str, wf_file: Path, state_file: Optional[Path]) -> Calculation:
           molecule = read_molecule(wf_file)
           basis = read_basis(wf_file, molecule)
           orbitals = read_orbitals(wf_file, basis)
           active_space = read_active_space(wf_file, orbitals)
           return Calculation(
               provenance=Provenance("MyProgram", kind, [str(wf_file)]),
               molecule=molecule,
               basis=basis,
               orbitals=orbitals,
               active_space=active_space,
               states=read_states(wf_file),
           )
   ```

   - If states are in a second file, read them from `state_file` and check
     they match the wavefunction file (`OpenMolcasLoader` compares energies).
   - If several readers parse the same file, open it once in an access object
     and pass that instead of the path (see `MolcasHdf5`).
   - Check the new readers with `Cᵀ S C = I` (orthonormal orbitals), as the
     Molcas tests do.
3. **Import its package** in `readers/__init__.py`, so the loader is registered
   (`readers/myprogram/__init__.py` imports the loader module).
4. **Add the program name** to `Calculator.program` in `input_schema.yaml`
   (and any rules for its file keys).

### Data model: what a loader must produce

Everything downstream works only on `Calculation` (`model.py`). Constructing
one validates it, so a loader that returns a `Calculation` is consistent. The
records are immutable (frozen dataclasses): to change one, derive a copy with
`dataclasses.replace`, which is validated again. The overview at the top of
`model.py` shows how the parts refer to each other.

| Part | Contents and conventions |
|---|---|
| `Provenance` | program, method, source files |
| `Molecule` | atom labels and coordinates (N × 3 array, **bohr**) |
| `basis` | An `AOBasis` subclass with one entry per AO, **in the row order of the MO coefficients**: `function_labels` (display only), `function_centers` (index into `Molecule.atoms`), `function_angular_momenta` (l). Either a `GaussianBasis` (contracted Gaussian shells; normalization conventions in the `GaussianShell` docstring) or an `EngineBasis` (integrals from a registered engine backend) |
| `OrbitalSet` | `coefficients` (n_AO × n_MO array, AOs as rows), one `orbital_ids` entry per MO (stable IDs such as `mo:34`), `energies`, `occupations`, and the program's orbital type labels. `basis_id` must equal the basis' `basis_id` |
| `ActiveSpace` | active `orbital_ids` in the order used by the CI occupation strings (RAS1, then RAS2, then RAS3), a partition label per orbital, and the number of active electrons |
| `ElectronicState` | `state_id`, `energy` (hartree), `spin` (S), `symmetry`, and `dominant_configuration` |
| `Configuration` | `occupations` as a list of `0`, `u`, `d`, `2` per active orbital, `coefficient` and `weight`; position k refers to `active_space.orbital_ids[k]`. `u`/`d` are spin-coupling labels of a CSF (as printed by Molcas), not α/β electrons |

Things to check in a new reader: orbitals must be orthonormal with the AO
overlap (`Cᵀ S C = I`; the tests check this for Molcas), and only the dominant
configuration of each state is kept.
