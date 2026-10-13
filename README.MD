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

Final:
  type: caspt2
  wf_file: ../data/molcas/brthph/s2p/mb/core_caspt2.caspt2.h5
  state_file: ../data/molcas/brthph/s2p/mb/core_caspt2.log
```

Relative paths are resolved from the input file's directory. Choice values are
case-insensitive. Run with:

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
├── propensity.py        # propensity workflows (overlaps, chp)
├── overlap/             # AO overlap providers and MO overlaps
│   ├── base.py          #   AOOverlapProvider interface
│   ├── gaussian.py      #   integrals for GaussianBasis
│   ├── engine.py        #   integrals from a program's own engine (EngineBasis)
│   ├── providers.py     #   provider selection by basis type
│   └── mo.py            #   mo_overlap
└── readers/
    ├── loader.py        # CalculationLoader base class, get_calculation, with_states
    └── openmolcas/      # OpenMolcas readers and OpenMolcasLoader
tests/                   # pytest suite (uses the data in data/molcas)
```

Run the tests with `pytest`.

### Adding a new electronic-structure program

1. **Readers** under `readers/<program>/` that turn the program's files into
   the model types described below.
2. **A loader**: a subclass of `CalculationLoader`. Defining it registers it;
   `get_calculation` does not need to change.

   ```python
   class MyProgramLoader(CalculationLoader):
       programs = ("myprogram",)          # names accepted by get_calculation
       display_name = "MyProgram"
       calculation_types = {"CASSCF"}     # upper case

       def _load(self, kind, wf_file, state_file):
           ...                            # read the files and return a Calculation
   ```

   If states come from a separate file, build the `Calculation` from the
   wavefunction file and combine them with `with_states(calculation, states)`
   (as `OpenMolcasLoader` does).
3. **Import its package** in `readers/__init__.py`, so the loader is registered.
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
| `Configuration` | `occupations` as a list of `0`, `u`, `d`, `2` per active orbital, `coefficient`, `weight`, and `orbital_ids` binding each position to an orbital. `u`/`d` are spin-coupling labels of a CSF (as printed by Molcas), not α/β electrons |

Things to check in a new reader: orbitals must be orthonormal with the AO
overlap (`Cᵀ S C = I`; the tests check this for Molcas), and only the dominant
configuration of each state is kept.
