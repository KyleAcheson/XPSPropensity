"""Program-neutral loading of calculations from program-specific files.

Each electronic-structure program provides one ``CalculationLoader`` subclass.
Defining the subclass registers it under its program names, so
``get_calculation`` never needs editing to support a new program; the
subclass's module only has to be imported (see ``xpsoverlaps.readers``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from pathlib import Path
from typing import ClassVar, Dict, Optional, Set, Tuple, Type

from xpsoverlaps.model import Calculation

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

