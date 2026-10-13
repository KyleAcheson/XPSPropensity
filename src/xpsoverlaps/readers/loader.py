"""Program-neutral loading of calculations from program-specific files.

Each electronic-structure program provides one ``CalculationLoader`` subclass.
Defining the subclass registers it under its program names, so
``get_calculation`` never needs editing to support a new program; the
subclass's module only has to be imported (see ``xpsoverlaps.readers``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
import inspect
from pathlib import Path
from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple, Type

from xpsoverlaps.model import Calculation, ElectronicState

# Lower-case program name -> loader class; filled by CalculationLoader.__init_subclass__.
_LOADERS: Dict[str, Type["CalculationLoader"]] = {}


class CalculationLoader(ABC):
    """Builds a program-neutral ``Calculation`` from one program's output files.

    Subclasses set ``programs`` (names accepted by ``get_calculation``),
    ``display_name`` and ``calculation_types``, and implement ``_load``.
    """

    programs: ClassVar[Tuple[str, ...]]
    display_name: ClassVar[str]
    calculation_types: ClassVar[Set[str]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        for name in cls.programs:
            key = name.strip().casefold()
            if key in _LOADERS and _LOADERS[key] is not cls:
                raise ValueError(
                    f"Program name {name!r} is already registered by {_LOADERS[key].__name__}."
                )
            _LOADERS[key] = cls

    def load(
        self,
        calculation_type: str,
        wf_file: str | Path,
        state_file: str | Path | None = None,
    ) -> Calculation:
        kind = calculation_type.strip().upper()
        if kind not in self.calculation_types:
            supported = ", ".join(sorted(self.calculation_types))
            raise ValueError(
                f"Unsupported {self.display_name} calculation type {calculation_type!r}; "
                f"supported types are: {supported}."
            )
        return self._load(kind, Path(wf_file), Path(state_file) if state_file is not None else None)

    @abstractmethod
    def _load(self, kind: str, wf_file: Path, state_file: Optional[Path]) -> Calculation:
        """Read the files for an upper-case, supported ``kind`` of calculation."""


def get_calculation(
    program: str,
    calculation_type: str,
    wf_file: str | Path,
    state_file: str | Path | None = None,
) -> Calculation:
    """Read and assemble a calculation using the requested program and method.

    ``wf_file`` holds the wavefunction (geometry, basis, orbitals, energies and,
    for some programs, the CI vectors). ``state_file`` is only needed by
    programs that write state CI data to a separate file (e.g. OpenMolcas).
    """
    loader = _LOADERS.get(program.strip().casefold())
    if loader is None:
        supported = ", ".join(sorted(_LOADERS))
        raise NotImplementedError(
            f"Calculation construction for program {program!r} is not implemented; "
            f"supported programs are: {supported}."
        )
    return loader().load(calculation_type, wf_file, state_file)


def with_states(
    calculation: Calculation,
    states: Sequence[ElectronicState],
    energy_tolerance: float = 1e-6,
) -> Calculation:
    """Return a new calculation with state metadata from a separate source.

    For programs that write orbitals and states to different files. States are
    matched by position and keep the calculation's energies; when both sources
    supply an energy they must agree within ``energy_tolerance`` (hartree).
    Spin, symmetry and the dominant configuration (bound to the active
    orbital IDs) come from ``states``. Neither input is modified.
    """
    if len(calculation.states) != len(states):
        raise ValueError(
            f"Calculation contains {len(calculation.states)} states but the state file "
            f"contains {len(states)} states."
        )

    active_ids = calculation.active_space.orbital_ids
    merged = []
    for calculation_state, source_state in zip(calculation.states, states):
        if (
            calculation_state.energy is not None
            and source_state.energy is not None
            and abs(calculation_state.energy - source_state.energy) > energy_tolerance
        ):
            raise ValueError(
                f"State {calculation_state.state_id} energy {calculation_state.energy:.8f} "
                f"does not match state {source_state.state_id} energy "
                f"{source_state.energy:.8f} from the state file; the files may not "
                "belong to the same calculation."
            )
        configuration = source_state.dominant_configuration
        if configuration is not None:
            if len(configuration.occupations) != len(active_ids):
                raise ValueError(
                    f"State {source_state.state_id} dominant configuration has "
                    f"{len(configuration.occupations)} occupations; "
                    f"the active space has {len(active_ids)} orbitals."
                )
            configuration = dataclasses.replace(configuration, orbital_ids=list(active_ids))
        merged.append(
            dataclasses.replace(
                calculation_state,
                spin=source_state.spin,
                symmetry=source_state.symmetry,
                dominant_configuration=configuration,
            )
        )
    return dataclasses.replace(calculation, states=merged)
