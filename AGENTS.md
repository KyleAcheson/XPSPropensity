XPSoverlaps approximates XPS shake-down Dyson norms with propensity rules built
on orbital overlaps S_ab = <phi_a|psi_b> (phi final/relaxed, psi initial).
Usage and input format are in `README.MD`; each module's docstring describes
its own design in detail.

## Theory

Read the LaTeX sources, not the PDFs:

- `docs/shakedown_cases.tex`: shake-down cases (3e/5e, GS/ES), orbital roles
  i, j, k, and the propensity rules and channels.
- `docs/chp.tex`: the core-hole potential (CHP), S_pq ~ -(pq|cc) / (e_p - e_q).

## Code map

Pipeline (`main.py` -> `methods.py`):

```text
YAML input (inputs.py, input_schema.yaml)
  -> CalculationLoader -> Calculation          (readers/, model.py)
  -> overlap source:
       overlaps: AO/MO overlaps -> orbital mapping   (overlap/, mapping.py, composition.py)
       chp:      (pq|cc) + orbital energies          (integrals/, readers/openmolcas/fcidump.py)
  -> orbital order, HOMO/LUMO                  (frontier.py)
  -> shake-downs and orbital roles i, j, k     (shakedown_transitions.py)
  -> propensity channels -> report            (propensity_rules.py)
```

| Module | Responsibility |
|---|---|
| `model.py` | Program-neutral records: `Calculation` and its parts, `ActiveIntegrals` |
| `inputs.py`, `input_schema.yaml` | Load and validate the YAML input (JSON Schema plus checks it cannot express) |
| `readers/loader.py` | `CalculationLoader` ABC, `get_calculation`, `get_active_integrals` |
| `readers/openmolcas/` | `MolcasHdf5` file-access layer, per-record HDF5 readers, RASSCF/CASPT2 log readers, FCIDUMP reader, `OpenMolcasLoader` |
| `overlap/` | `AOOverlapProvider` ABC (Gaussian and engine providers), `mo_overlap` |
| `composition.py` | Mulliken AO composition of MOs (mapping tie-breaks, orbital descriptions) |
| `mapping.py` | Initial -> final active-orbital mapping (`OrbitalMapping`), mapped CI report |
| `integrals/` | `CoreHoleIntegrals` ABC, `CoulombMatrix` (pq|cc), `StoredIntegrals` |
| `overlap_sources.py` | `OrbitalOverlaps` ABC: `MappedOverlaps` (overlaps method), `CoreHoleOverlaps` (CHP) |
| `frontier.py` | Orbital order (by populations), reference determinant, HOMO/LUMO |
| `shakedown_transitions.py` | Find and classify shake-downs, `ShakeDown.overlaps(overlap_source)` |
| `propensity_rules.py` | `Channel` and the `CHANNELS` table, evaluation and reports |
| `methods.py` | `propensity_overlaps` and `propensity_chp`: run the stages, log the sections |

## Extension points

- **New program:** subclass `CalculationLoader` (it registers itself via
  `__init_subclass__`), import its package in `readers/__init__.py`, and add
  the program name to `input_schema.yaml`. Never edit `get_calculation`.
- **New basis representation / integral engine:** an `AOOverlapProvider`, or
  `register_engine` for an `EngineBasis`.
- **New way to get S_ab:** an `OrbitalOverlaps` subclass.
- **New source of (pq|cc):** a `CoreHoleIntegrals` subclass.
- **New propensity rule or channel:** an entry in `CHANNELS`.

## Design principles

- **Data, algorithms and reports are separate.** Loaders build a `Calculation`;
  analysis works only on the model, never on program files; report functions
  return lines and `methods.py` logs them.
- **Program details stay in the readers.** File layouts, log syntax and
  program conventions (ordering, phases, normalization) are absorbed by the
  reader; nothing else imports from `readers/openmolcas`.
- **Identities and index spaces are explicit.** Orbitals are referred to by ID
  (`mo:34`), never by array position; CI occupations are in active-space order.
- **Parsed data is never mutated.** Orbital correspondence is a mapping
  (data), not a reordering of a calculation; records are built once and
  validated on construction.
- **Approximations are explicit and reported.** E.g. the dominant CSF stands
  for a state, diagonal overlaps are set to 1, CHP is first order; warnings are
  printed when they look unreliable.
- **Tests check known identities.** E.g. every overlap provider must give
  `C.T @ S_AO @ C = 1` for each calculation (`tests/test_overlap.py`).

## Coding standards

- Interfaces are abstract base classes with a template method (public method
  validates, subclass implements `_compute`/`_load`), not Protocols or bare
  type-hinted callables.
- Behaviour belongs to the class that owns the data: a method, not a module
  function taking the object (`Channel.evaluate`, `ShakeDown.overlaps`). One
  class holds everything one computation needs.
- Dataclasses only for records: the model is frozen dataclasses with
  `eq=False`. Small results are `NamedTuple`s, the YAML input `TypedDict`s. A
  class with behaviour or validation in its constructor is a plain class with
  `__init__`, not a dataclass with `__post_init__`.
- No type aliases that exist only for hinting, no container types duplicating
  existing ones, no partly built objects.
- Names say what they hold: `orbital_roles` (never bare "roles"),
  `overlap_source` (never bare "source").
- Notation: phi = final (relaxed), psi = initial; S_ab = <phi_a|psi_b>, the
  first index is the relaxed orbital. Printed formulas are generated from the
  same text that is evaluated, so output and computation cannot disagree.
- Keep `README.MD` simple.

## Agent skills

### Issue tracker

Issues and specs are GitHub issues on KyleAcheson/XPSPropensity, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
