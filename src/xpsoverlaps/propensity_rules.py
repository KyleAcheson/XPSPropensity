"""Propensity rules for shake-down transitions (csf_rules notes).

Overlaps follow S_ab = <phi_a|psi_b>: the FIRST index is initial orbital a
relaxed in the final state, the second the initial-state orbital b. The
numbers come from an overlap source (``overlap_sources``) chosen by the
propensity method; ``ShakeDown.overlaps`` pairs it with a shake-down's orbital
roles i, j, k as a ``ShakeDownOverlaps``. The overlaps method uses the final
orbital the mapping matches to a, with absolute values. Diagonal overlaps
S_ii, S_jj, S_kk are kept in the formulas but set to 1 (weak relaxation);
their actual values are reported and flagged below ``DIAGONAL_WARNING``.

Every bright channel is one ``Channel`` in the ``CHANNELS`` table, keyed by
shake-down case and initial spin. A channel holds everything needed for its
intensity, weight * |amplitude|^2: the equations defining its quantities
(M1..M4, Pi, Omega), written as in the notes and evaluated in order, and the
amplitude in terms of them. The same text is evaluated and printed, so the
printed rule and the number cannot disagree.

Which channels exist depends only on the initial spin (averaged over initial
M_S, summed over final M_S): a singlet reaches the closed-shell doublet (GS) or
D1 |Pi + Omega|^2 (ES); a triplet reaches D2 1/3 |Pi - Omega|^2 and Q
2/3 |Pi - Omega|^2 (ES) and nothing in ground-state shake-down. Each channel is
evaluated with the overlaps of the final calculation whose states have the
channel's final spin, so a triplet's D2 and Q come from different final
orbitals when the doublet and quartet states are separate calculations.
"""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from xpsoverlaps.mapping import MappedFinal
from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap_sources import CoreHoleOverlaps, MappedOverlaps, OrbitalOverlaps
from xpsoverlaps.shakedown_transitions import ShakeDown, ShakeDownOverlaps

DIAGONAL_WARNING = 0.9

Term = Tuple[int, Tuple[str, ...]]  # sign, factors ("S_ij" or a quantity name such as "M1")

_ORBITAL_ROLES = "ijk"


class ChannelResult(NamedTuple):
    channel: "Channel"
    values: Dict[str, float]  # the channel's quantities: M1.., Pi, Omega
    amplitude: float
    intensity: float  # weight * amplitude^2

    @property
    def name(self) -> str:
        return self.channel.name


class Channel:
    """One bright shake-down channel and everything needed to compute its intensity.

    The intensity is ``weight * |amplitude|^2``, where the amplitude is written
    in terms of quantities (M1, Pi, Omega, A, ...) defined by ``equations``.
    Formulas are sums of signed products of factors, each factor an overlap
    ``S_ab`` (a, b one of the roles i, j, k; first index final, second initial)
    or a quantity defined by an earlier equation. They are parsed and checked
    once, when the channel is created.

    Parameters
    ----------
    name : str
        Channel label: "D1", "D2", "Q" or "closed-shell doublet".
    final_spin : float
        Spin S of the final states the channel reaches (0.5 or 1.5). The
        channel is evaluated with the overlaps of the final calculation of
        this spin.
    weight : Fraction
        Spin weight multiplying ``|amplitude|^2``: 1 (D1), 1/3 (D2), 2/3 (Q),
        or 2 for the closed-shell doublet.
    equations : sequence of str
        Equations ``"name = formula"`` defining the quantities, in evaluation
        order, e.g. ``("M1 = S_ii S_jj - S_ij S_ji", ..., "Pi = M1 M2")``.
        Each formula may use overlaps and the quantities defined before it.
        Empty if the amplitude uses overlaps only.
    amplitude : str
        Formula inside ``|...|^2``, in terms of overlaps and the quantities,
        e.g. ``"Pi - Omega"``.

    Attributes
    ----------
    name, final_spin : as given.
    weight : Fraction
        The weight as a ``Fraction``.
    equations : list of (str, str, list of Term)
        Quantity name, formula text and parsed terms of each equation.
    amplitude_text : str
        The amplitude formula as given, for printing.
    amplitude : list of Term
        The parsed amplitude.

    Raises
    ------
    ValueError
        If an equation is not ``"name = formula"``, a factor is neither an
        overlap ``S_ab`` of the roles i, j, k nor a quantity name, a formula
        is empty, or a formula uses a quantity not defined by an earlier
        equation. The message names the channel.

    Examples
    --------
    >>> d1 = Channel("D1", 1/2, Fraction(1), ("Pi = S_ii S_jk", "Omega = S_ik S_ji"), "Pi + Omega")
    >>> S = shake_down.overlaps(MappedOverlaps(mapping))  # S("jk") = S_jk of this shake-down
    >>> d1.evaluate(S).intensity  # |S_jk + S_ik S_ji|^2
    """

    def __init__(self, name: str, final_spin: float, weight: Fraction,
                 equations: Sequence[str], amplitude: str) -> None:
        self.name = name  # "D1", "D2", "Q" or "closed-shell doublet"
        self.final_spin = final_spin  # spin of the final states it reaches
        self.weight = Fraction(weight)  # 1, 1/3, 2/3, or 2 for the closed shell
        # (quantity name, formula text, parsed terms), in evaluation order
        self.equations = [self._parse_equation(text) for text in equations]
        self.amplitude_text = amplitude
        self.amplitude = self._parse(amplitude)
        self._check_names()

    def __repr__(self) -> str:
        return f"Channel({self.name!r}, final S = {self.final_spin:g}: {self.formula})"

    @property
    def formula(self) -> str:
        """The intensity formula as text, e.g. ``"1/3 |Pi - Omega|^2"``."""
        return self._weighted(self.amplitude_text)

    @property
    def overlaps(self) -> List[str]:
        """Role pairs (``"ii"``, ``"ij"``, ...) the channel uses, in order of first appearance."""
        all_terms = [terms for _, _, terms in self.equations] + [self.amplitude]
        return list(dict.fromkeys(
            factor[2:] for terms in all_terms for _, factors in terms for factor in factors if factor.startswith("S_")
        ))

    def evaluate(self, S: ShakeDownOverlaps) -> ChannelResult:
        """Evaluate the equations in order, then the amplitude and the intensity.

        Parameters
        ----------
        S : ShakeDownOverlaps
            Overlap of a role pair, ``S("ik")`` = S_ik, for one shake-down and
            one overlap source (e.g. one final calculation).

        Returns
        -------
        ChannelResult
            This channel, the value of each quantity, the amplitude, and the
            intensity ``weight * amplitude^2``.
        """
        values: Dict[str, float] = {}
        for name, _, terms in self.equations:
            values[name] = self._evaluate_terms(terms, S, values)
        amplitude = self._evaluate_terms(self.amplitude, S, values)
        return ChannelResult(self, values, amplitude, float(self.weight) * amplitude ** 2)

    def equation_lines(self, S: ShakeDownOverlaps, result: ChannelResult) -> List[str]:
        """Each equation written out with its overlaps and quantities substituted.

        Parameters
        ----------
        S : ShakeDownOverlaps
            The overlaps ``result`` was evaluated with.
        result : ChannelResult
            The result of ``evaluate(S)``.

        Returns
        -------
        list of str
            One line per equation, e.g.
            ``"  M1 = S_ii S_jj - S_ij S_ji = (1)(1) - (0.1263)(0.1267) = +0.98399"``.
        """
        return [f"  {name} = {text} = {self._substitute(terms, S, result.values)} = {result.values[name]:+.5f}"
                for name, text, terms in self.equations]

    def intensity_line(self, S: ShakeDownOverlaps, result: ChannelResult) -> str:
        """The intensity formula, expanded where possible, with values and result.

        Parameters
        ----------
        S : ShakeDownOverlaps
            The overlaps ``result`` was evaluated with.
        result : ChannelResult
            The result of ``evaluate(S)``.

        Returns
        -------
        str
            e.g. ``"  D1: |Pi + Omega|^2 = |M1 M2 + M3 M4|^2 = |(+0.12305) + (-0.00017)|^2 = 0.015099"``.
        """
        formula = self.formula
        expanded = self._expand()
        if expanded != self.amplitude_text:
            formula += f" = {self._weighted(expanded)}"
        numbers = self._weighted(self._substitute(self.amplitude, S, result.values))
        return f"  {self.name}: {formula} = {numbers} = {result.intensity:.6f}"

    # --- formulas --------------------------------------------------------------

    def _weighted(self, amplitude: str) -> str:
        return (f"{self.weight} " if self.weight != 1 else "") + f"|{amplitude}|^2"

    def _parse_equation(self, text: str) -> Tuple[str, str, List[Term]]:
        """"M1 = S_ii S_jj - S_ij S_ji" -> ("M1", "S_ii S_jj - S_ij S_ji", terms)."""
        name, separator, formula = (part.strip() for part in text.partition("="))
        if not separator or not name.isidentifier() or name.startswith("S_"):
            raise ValueError(f"Channel {self.name}: {text!r} is not an equation 'name = formula'.")
        return name, formula, self._parse(formula)

    def _parse(self, formula: str) -> List[Term]:
        """'S_ii S_jj - S_ij S_ji' -> [(1, ('S_ii', 'S_jj')), (-1, ('S_ij', 'S_ji'))]."""
        terms: List[Tuple[int, List[str]]] = []
        sign, factors = 1, None
        for token in formula.split():
            if token in ("+", "-"):
                sign, factors = (1 if token == "+" else -1), None
                continue
            if token.startswith("S_") and not (len(token) == 4 and set(token[2:]) <= set(_ORBITAL_ROLES)):
                raise ValueError(f"Channel {self.name}: {token!r} in {formula!r} is not an overlap S_ab of roles i, j, k.")
            if not token.startswith("S_") and not token.isidentifier():
                raise ValueError(f"Channel {self.name}: {token!r} in {formula!r} is not an overlap or a quantity name.")
            if factors is None:
                factors = []
                terms.append((sign, factors))
            factors.append(token)
        if not terms:
            raise ValueError(f"Channel {self.name}: empty formula.")
        return [(sign, tuple(factors)) for sign, factors in terms]

    def _check_names(self) -> None:
        """Every quantity a formula uses is defined by an earlier equation."""
        defined: List[str] = []
        for name, text, terms in self.equations + [("amplitude", self.amplitude_text, self.amplitude)]:
            undefined = sorted({f for _, factors in terms for f in factors
                                if not f.startswith("S_") and f not in defined})
            if undefined:
                raise ValueError(
                    f"Channel {self.name}: {name} = {text} uses {', '.join(undefined)}, "
                    "not defined by an earlier equation."
                )
            defined.append(name)

    @staticmethod
    def _evaluate_terms(terms: List[Term], S: ShakeDownOverlaps, values: Dict[str, float]) -> float:
        return sum(
            sign * prod(S(f[2:]) if f.startswith("S_") else values[f] for f in factors) for sign, factors in terms
        )

    @staticmethod
    def _substitute(terms: List[Term], S: ShakeDownOverlaps, values: Dict[str, float]) -> str:
        """The formula with every factor replaced by its value: '(1)(1) - (0.1263)(0.1267)'."""
        def number(factor: str) -> str:
            if factor.startswith("S_"):
                pair = factor[2:]
                return "1" if pair[0] == pair[1] else f"{S(pair):.4f}"
            return f"{values[factor]:+.5f}"

        text = ""
        for n, (sign, factors) in enumerate(terms):
            if n or sign < 0:
                text += (" + " if sign > 0 else " - ") if n else "-"
            text += "".join(f"({number(f)})" for f in factors)
        return text

    def _expand(self) -> str:
        """The amplitude with each quantity defined by a single product (Pi = M1 M2)
        replaced by that product: "Pi + Omega" -> "M1 M2 + M3 M4"."""
        products = {name: text for name, text, terms in self.equations if len(terms) == 1 and terms[0][0] == 1}
        return " ".join(products.get(token, token) for token in self.amplitude_text.split())


# --- The channels ------------------------------------------------------------

# The equations of each case, passed to its channels below. The 2x2 minors of the
# valence block (notes eq. 20) are shared by the 5e cases.
M1 = "M1 = S_ii S_jj - S_ij S_ji"
M2 = "M2 = S_ij S_kk - S_ik S_kj"
M3 = "M3 = S_ij S_jk - S_ik S_jj"
M4 = "M4 = S_ii S_kj - S_ij S_ki"
# Excited-state shake-down: direct (Pi) and exchange (Omega) pathway amplitudes.
ES_3E = ("Pi = S_ii S_jk", "Omega = S_ik S_ji")  # notes eq. 30
ES_5E = (M1, M2, M3, M4, "Pi = M1 M2", "Omega = M3 M4")  # notes eq. 21
# Ground-state shake-down: the closed-shell amplitude A.
GS_3E = ("A = S_ii S_ik",)  # i = HOMO; notes eq. 37
GS_5E = (M1, M3, "A = M1 M3")  # i = HOMO-n, j = HOMO; notes eq. 34

# Bright channels by (shake-down case, initial spin); an empty tuple means dark
# (notes Secs. 5-7). Channel(name, final spin, weight, equations, amplitude).
CHANNELS: Dict[Tuple[str, float], Tuple[Channel, ...]] = {
    ("3e ES", 0): (Channel("D1", 1/2, Fraction(1), ES_3E, "Pi + Omega"),),
    ("3e ES", 1): (Channel("D2", 1/2, Fraction(1, 3), ES_3E, "Pi - Omega"),
                   Channel("Q", 3/2, Fraction(2, 3), ES_3E, "Pi - Omega")),
    ("5e ES", 0): (Channel("D1", 1/2, Fraction(1), ES_5E, "Pi + Omega"),),
    ("5e ES", 1): (Channel("D2", 1/2, Fraction(1, 3), ES_5E, "Pi - Omega"),
                   Channel("Q", 3/2, Fraction(2, 3), ES_5E, "Pi - Omega")),
    ("3e GS", 0): (Channel("closed-shell doublet", 1/2, Fraction(2), GS_3E, "A"),),
    ("3e GS", 1): (),
    ("5e GS", 0): (Channel("closed-shell doublet", 1/2, Fraction(2), GS_5E, "A"),),
    ("5e GS", 1): (),
}


# --- Evaluation on a shake-down ---------------------------------------------


class PropensityResult(NamedTuple):
    shake_down: ShakeDown
    initial_spin: Optional[float]
    overlaps: ShakeDownOverlaps  # S_ab of the shake-down's orbitals from one overlap source
    overlap_values: Dict[str, float]  # role pair -> S used by the evaluated channels (diagonals = 1)
    diagonals: Dict[str, float]  # role pair ("ii", ...) -> actual S
    channels: List[ChannelResult]  # evaluated channels
    note: Optional[str] = None  # why no channel is bright

    @property
    def values(self) -> Dict[str, float]:
        """The quantities (M1.., Pi, Omega) of the evaluated channels."""
        return {name: value for result in self.channels for name, value in result.values.items()}


def _initial_spin(shake_down: ShakeDown, initial: Calculation) -> Optional[float]:
    return next(s.spin for s in initial.states if s.state_id == shake_down.initial_state)


def bright_channels(shake_down: ShakeDown, initial: Calculation) -> Tuple[Tuple[Channel, ...], Optional[str]]:
    """The channels of a computable shake-down for its initial spin, or a note why there are none."""
    spin = _initial_spin(shake_down, initial)
    key = (shake_down.case, spin)
    if key not in CHANNELS:
        return (), f"no rule for initial spin S = {spin}"
    if not CHANNELS[key]:
        return (), "dark: a triplet has no ground-state shake-down channel"
    return CHANNELS[key], None


def compute_propensity(
    shake_down: ShakeDown, initial: Calculation, overlap_source: OrbitalOverlaps,
    final_spin: Optional[float] = None,
) -> PropensityResult:
    """Evaluate the bright channels of a computable shake-down with the overlaps of ``overlap_source``.

    With ``final_spin`` only the channels to final states of that spin are
    evaluated: ``overlap_source`` then belongs to the final calculation of that spin.
    """
    S = shake_down.overlaps(overlap_source)
    channels, note = bright_channels(shake_down, initial)
    selected = [c for c in channels if final_spin is None or c.final_spin == final_spin]
    pairs = dict.fromkeys(pair for channel in selected for pair in channel.overlaps)
    return PropensityResult(
        shake_down,
        _initial_spin(shake_down, initial),
        S,
        {pair: S(pair) for pair in pairs},
        {role * 2: S.actual(role * 2) for role in S.orbital_roles},
        [channel.evaluate(S) for channel in selected],
        note,
    )


def evaluate_shake_downs(
    shake_downs: Sequence[ShakeDown], initial: Calculation, overlap_source: OrbitalOverlaps,
    final_spin: Optional[float] = None,
) -> List[PropensityResult]:
    """The computable shake-downs with a bright channel (to ``final_spin``, if given),
    evaluated with the overlaps of ``overlap_source``."""
    return [
        compute_propensity(sd, initial, overlap_source, final_spin)
        for sd in shake_downs
        if sd.computable and any(final_spin is None or c.final_spin == final_spin
                                 for c in bright_channels(sd, initial)[0])
    ]


def evaluate_final(
    shake_downs: Sequence[ShakeDown], initial: Calculation, final: MappedFinal
) -> List[PropensityResult]:
    """The computable shake-downs with a channel to ``final``'s spin, evaluated with its overlaps."""
    return evaluate_shake_downs(shake_downs, initial, MappedOverlaps(final.mapping), final.spin)


# --- Report ------------------------------------------------------------------


def _number(state_id: str) -> str:
    return state_id.split(":")[-1]


def _result_lines(result: PropensityResult) -> List[str]:
    """One shake-down: orbital roles, overlaps by orbital, diagonal checks, each channel's rule in full."""
    sd = result.shake_down
    S = result.overlaps
    lines = [
        "",
        f"{_number(sd.initial_state)} -> {_number(sd.lower_state)}   {sd.case}   "
        f"initial S = {result.initial_spin:g}" + (f"   (d = {sd.d} omitted)" if sd.d else ""),
        "  " + "   ".join(S.describe_role(role) for role in S.orbital_roles),
    ]
    lines += [f"  S_{pair} = {S.describe(pair)} = {value:.4f}" for pair, value in result.overlap_values.items()
              if pair[0] != pair[1]]
    for pair, value in result.diagonals.items():
        flag = f"   below {DIAGONAL_WARNING}: weak-relaxation approximation poor" if value < DIAGONAL_WARNING else ""
        lines.append(f"  S_{pair} = {S.describe(pair)} = {value:.4f} (set to 1){flag}")
    # Channels evaluated together share their equations: print each quantity once.
    for channel_result in result.channels:
        lines += [line for line in channel_result.channel.equation_lines(S, channel_result) if line not in lines]
    lines += [r.channel.intensity_line(S, r) for r in result.channels]
    return lines


class CoreHoleRoute(NamedTuple):
    """The core orbitals of one irrep and the final irrep they reach: Γ_F = Γ_I × Γ_c (ADR-0005)."""

    core_irrep: str
    cores: Tuple[str, ...]
    final_irrep: str

    @property
    def heading(self) -> str:
        """e.g. "Core hole a' (mo:21_i1, mo:22_i1) -> final A'"."""
        return f"Core hole {self.core_irrep} ({', '.join(self.cores)}) -> final {self.final_irrep.upper()}"


def core_hole_routes(initial: Calculation, cores: Sequence[str]) -> List[CoreHoleRoute]:
    """One route per irrep among the core orbitals, in their order: the final irrep their holes reach."""
    initial_irrep = initial.states[0].symmetry
    irrep_of = dict(zip(initial.orbitals.orbital_ids, initial.orbitals.irreps))
    return [
        CoreHoleRoute(irrep, tuple(c for c in cores if irrep_of[c] == irrep),
                      initial.point_group.product(initial_irrep, irrep))
        for irrep in dict.fromkeys(irrep_of[c] for c in cores)
    ]


class SummaryRow(NamedTuple):
    """One channel of one shake-down in the summary, or the reason it has none."""

    shake_down: ShakeDown
    channel: Optional[Channel]  # None when the shake-down has no bright channel
    route: Optional[CoreHoleRoute]  # None without symmetry
    final_label: Optional[str]  # None when no final calculation serves the channel
    intensity: Optional[float]
    note: Optional[str] = None

    def columns(self, symmetric: bool) -> str:
        sd = self.shake_down
        pair = f"{_number(sd.initial_state)} -> {_number(sd.lower_state)}"
        irreps = f"{self.route.core_irrep:<4}  {self.route.final_irrep.upper():<4}  " if symmetric else ""
        intensity = f"{self.intensity:10.6f}" if self.intensity is not None else f"{'-':>10}"
        name = self.channel.name if self.channel is not None else "-"
        return (f"{pair:>8}  {sd.case:<6}  {irreps}{self.final_label or '-':<8}  {name:<21}  {intensity}"
                + (f"   {self.note}" if self.note else ""))


def summary_header(symmetric: bool) -> str:
    irreps = f"{'Core':<4}  {'To':<4}  " if symmetric else ""  # core-hole irrep, final irrep
    return f"{'a -> b':>8}  {'Case':<6}  {irreps}{'Final':<8}  {'Channel':<21}  {'Intensity':>10}"


def summary_rows(
    shake_downs: Sequence[ShakeDown],
    initial: Calculation,
    evaluations: Sequence[Tuple[MappedFinal, Sequence[PropensityResult]]],
    routes: Optional[Sequence[CoreHoleRoute]] = None,
) -> List[SummaryRow]:
    """Every computable shake-down's channels, each with the final calculation of its spin
    (and, with ``routes``, of each core-hole route's final irrep)."""
    results_by_final = {
        final.label: {result.shake_down: result for result in results} for final, results in evaluations
    }
    rows = []
    for sd in shake_downs:
        if not sd.computable:
            continue
        channels, note = bright_channels(sd, initial)
        for route in routes or [None]:
            if not channels:
                rows.append(SummaryRow(sd, None, route, None, None, note))
            for channel in channels:
                final = next((f for f, _ in evaluations if f.spin == channel.final_spin
                              and (route is None or f.irrep == route.final_irrep)), None)
                if final is None:
                    irrep = "" if route is None else f", {route.final_irrep.upper()}"
                    rows.append(SummaryRow(sd, channel, route, None, None,
                                           f"no S = {channel.final_spin:g}{irrep} final calculation"))
                    continue
                result = next(r for r in results_by_final[final.label][sd].channels if r.channel is channel)
                rows.append(SummaryRow(sd, channel, route, final.label, result.intensity))
    return rows


def propensity_report(
    shake_downs: Sequence[ShakeDown],
    initial: Calculation,
    evaluations: Sequence[Tuple[MappedFinal, Sequence[PropensityResult]]],
    routes: Optional[Sequence[CoreHoleRoute]] = None,
    forbidden: Sequence[str] = (),
) -> List[str]:
    """For each final calculation the rules evaluated with its overlaps, then one
    summary of every computable shake-down's channels across the final calculations.

    With symmetry (``routes``), the final calculations are grouped by the
    core-hole route that reaches their irrep, and ``forbidden`` lists the
    final calculations no core hole of this initial irrep reaches.
    """
    lines = [
        "Propensity rules",
        "",
        "S_ab = |<phi_a|psi_b>|: a is the final-state orbital matched to initial orbital a,",
        "b is the initial-state orbital. Diagonal S_ii, S_jj, S_kk are set to 1.",
        "Non-zero channels: singlet -> GS closed-shell doublet or ES D1;"
        " triplet -> ES D2 and Q (GS dark).",
        "Each channel uses the final calculation of its spin: closed-shell doublet, D1, D2 -> S = 0.5;"
        " Q -> S = 1.5.",
    ]
    if routes is not None:
        lines.append("A core hole of irrep c reaches final states of irrep (initial irrep) x c only;"
                     " its rules use the final calculations of that irrep.")
        lines += [f"{title}: symmetry-forbidden for this initial irrep (no core hole reaches it)."
                  for title in forbidden]

    def final_blocks(group: Sequence[Tuple[MappedFinal, Sequence[PropensityResult]]]) -> List[str]:
        block = []
        for final, results in group:
            block += ["", final.title, "-" * len(final.title)]
            if not results:
                block.append(f"No shake-down has a channel to S = {final.spin:g} final states.")
            for result in results:
                block += _result_lines(result)
        return block

    if routes is None:
        lines += final_blocks(evaluations)
    for route in routes or []:
        lines += ["", route.heading, "=" * len(route.heading)]
        group = [(final, results) for final, results in evaluations if final.irrep == route.final_irrep]
        lines += final_blocks(group) if group else ["", f"No {route.final_irrep.upper()} final calculation."]

    lines += ["", "Summary", summary_header(routes is not None)]
    lines += [row.columns(routes is not None) for row in summary_rows(shake_downs, initial, evaluations, routes)]
    return lines


def closing_summary(blocks: Sequence[Tuple[str, Sequence[SummaryRow]]], failures: Sequence[str] = ()) -> List[str]:
    """The summaries of several initial calculations in one table; ``blocks`` holds each one's irrep and
    rows, ``failures`` the errors of the blocks that were skipped."""
    lines = ["Summary over all initial calculations", "", f"{'Init':<4}  " + summary_header(True)]
    for irrep, rows in blocks:
        lines += [f"{irrep.upper():<4}  " + row.columns(True) for row in rows]
    if failures:
        lines += ["", "Not evaluated:"] + [f"  {failure.splitlines()[0]}" for failure in failures]
    return lines


def core_hole_report(
    shake_downs: Sequence[ShakeDown],
    initial: Calculation,
    evaluations: Sequence[Tuple[CoreHoleOverlaps, Sequence[PropensityResult]]],
    routes: Optional[Sequence[CoreHoleRoute]] = None,
) -> List[str]:
    """For each core orbital the rules evaluated with its core-hole overlaps, then one
    summary of every computable shake-down's channels: the intensity per core orbital and their mean.

    With symmetry (``routes``) each core orbital is labelled with its irrep and
    the final irrep its hole reaches (ADR-0005).
    """
    cores = [overlap_source.core for overlap_source, _ in evaluations]
    titles = core_titles(cores, routes)
    lines = [
        "Propensity rules (core-hole potential)",
        "",
        *evaluations[0][0].propensity_notes,
        "Non-zero channels: singlet -> GS closed-shell doublet or ES D1;"
        " triplet -> ES D2 and Q (GS dark).",
        "Every channel uses the same overlaps (Coulomb only): Q = 2 D2.",
    ]
    if routes is not None:
        lines.append("A core hole of irrep c reaches final states of irrep (initial irrep) x c only.")
    for core, (_, results) in zip(cores, evaluations):
        title = titles.get(core, f"Core hole in {core}")
        lines += ["", title, "-" * len(title)]
        if not results:
            lines.append("No shake-down has a bright channel.")
        for result in results:
            lines += _result_lines(result)

    columns = "".join(f"  {core:>10}" for core in cores)
    lines += ["", "Summary"]
    if routes is not None:
        lines.append("Final irreps: " + ", ".join(f"{core} ({route.core_irrep}) -> {route.final_irrep.upper()}"
                                                  for route in routes for core in route.cores))
    lines.append(f"{'a -> b':>8}  {'Case':<6}  {'Channel':<21}{columns}  {'Mean':>10}")
    for sd, channel, intensities, note in core_hole_rows(shake_downs, initial, evaluations):
        pair = f"{_number(sd.initial_state)} -> {_number(sd.lower_state)}"
        if channel is None:
            blanks = "".join(f"  {'-':>10}" for _ in cores)
            lines.append(f"{pair:>8}  {sd.case:<6}  {'-':<21}{blanks}  {'-':>10}   {note}")
            continue
        values = "".join(f"  {intensities[core]:10.6f}" for core in cores)
        lines.append(f"{pair:>8}  {sd.case:<6}  {channel.name:<21}{values}  "
                     f"{sum(intensities.values()) / len(intensities):10.6f}")
    return lines


def core_titles(cores: Sequence[str], routes: Optional[Sequence[CoreHoleRoute]]) -> Dict[str, str]:
    """Heading of each core orbital's results: "Core hole in mo:1_i2 (a") -> final A'" with symmetry."""
    if routes is None:
        return {core: f"Core hole in {core}" for core in cores}
    return {core: f"Core hole in {core} ({route.core_irrep}) -> final {route.final_irrep.upper()}"
            for route in routes for core in route.cores}


def core_hole_rows(
    shake_downs: Sequence[ShakeDown],
    initial: Calculation,
    evaluations: Sequence[Tuple[CoreHoleOverlaps, Sequence[PropensityResult]]],
) -> List[Tuple[ShakeDown, Optional[Channel], Dict[str, float], Optional[str]]]:
    """Every computable shake-down's channels with the intensity per core orbital, or a note why it has none."""
    results_by_core = {overlap_source.core: {result.shake_down: result for result in results}
                       for overlap_source, results in evaluations}
    rows = []
    for sd in shake_downs:
        if not sd.computable:
            continue
        channels, note = bright_channels(sd, initial)
        if not channels:
            rows.append((sd, None, {}, note))
        for channel in channels:
            rows.append((sd, channel, {
                core: next(r for r in by_sd[sd].channels if r.channel is channel).intensity
                for core, by_sd in results_by_core.items()
            }, None))
    return rows


def core_hole_closing_summary(blocks: Sequence[Tuple[str, Sequence[CoreHoleRoute], Sequence[Tuple]]]) -> List[str]:
    """The core-hole summaries of several initial calculations in one table, one row per core orbital.

    ``blocks`` holds each initial calculation's irrep, core-hole routes and ``core_hole_rows``.
    """
    lines = ["Summary over all initial calculations (core-hole potential)", "",
             f"{'Init':<4}  {'a -> b':>8}  {'Case':<6}  {'Channel':<21}  {'Core':<9} {'Irr':<4}  {'To':<4}  "
             f"{'Intensity':>10}"]
    for irrep, routes, rows in blocks:
        route_of = {core: route for route in routes for core in route.cores}
        for sd, channel, intensities, note in rows:
            pair = f"{_number(sd.initial_state)} -> {_number(sd.lower_state)}"
            if channel is None:
                lines.append(f"{irrep.upper():<4}  {pair:>8}  {sd.case:<6}  {'-':<21}  {'-':<9} {'':<4}  {'':<4}  "
                             f"{'-':>10}   {note}")
            for core, intensity in intensities.items():
                route = route_of[core]
                lines.append(f"{irrep.upper():<4}  {pair:>8}  {sd.case:<6}  {channel.name:<21}  {core:<9} "
                             f"{route.core_irrep:<4}  {route.final_irrep.upper():<4}  {intensity:10.6f}")
    return lines
