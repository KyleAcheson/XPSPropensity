"""Propensity rules for shake-down transitions (csf_rules notes).

Overlaps follow S_ab = |<phi_a|psi_b>|: the FIRST index is a final-state
orbital, the final orbital the mapping matches to initial orbital a; the
second is the initial-state orbital b. Absolute values are used throughout.
Diagonal overlaps S_ii, S_jj, S_kk are kept in the formulas but set to 1
(weak relaxation); their actual values are reported and flagged below
``DIAGONAL_WARNING``.

Each rule is written once as text, in the notation of the notes, and that
text is both evaluated and printed:

    M1 = S_ii S_jj - S_ij S_ji        M2 = S_ij S_kk - S_ik S_kj
    M3 = S_ij S_jk - S_ik S_jj        M4 = S_ii S_kj - S_ij S_ki

    3e ES:  Pi = S_ii S_jk,  Omega = S_ik S_ji
    5e ES:  Pi = M1 M2,      Omega = M3 M4
    3e GS:  closed-shell doublet 2 |S_ii S_ik|^2   (i = HOMO)
    5e GS:  closed-shell doublet 2 |M1 M3|^2       (i = HOMO-n, j = HOMO)

Which channels are non-zero depends only on the initial spin (averaged over
initial M_S, summed over final M_S): a singlet reaches the closed-shell
doublet (GS) or D1 |Pi + Omega|^2 (ES); a triplet reaches D2 1/3 |Pi - Omega|^2
and Q 2/3 |Pi - Omega|^2 (ES) and nothing in ground-state shake-down.

Each channel is evaluated with the overlaps of the final calculation whose
states have the channel's spin (doublet channels: S = 1/2, Q: S = 3/2), so a
triplet's D2 and Q come from different final orbitals when the doublet and
quartet states are separate calculations.
"""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from xpsoverlaps.mapping import MappedFinal, OrbitalMapping
from xpsoverlaps.model import Calculation
from xpsoverlaps.shakedown import ShakeDown

DIAGONAL_WARNING = 0.9

# Final-state spin of each channel: the final calculation it is evaluated with.
CHANNEL_SPIN = {"closed-shell doublet": 0.5, "D1": 0.5, "D2": 0.5, "Q": 1.5}

Overlap = Callable[[str], float]  # role pair such as "ik" -> S_ik

# --- Formulas: sums of signed products of factors S_ab or named quantities ---

Term = Tuple[int, List[str]]  # sign, factors


def _terms(formula: str) -> List[Term]:
    """'S_ii S_jj - S_ij S_ji' -> [(1, ['S_ii', 'S_jj']), (-1, ['S_ij', 'S_ji'])]."""
    terms: List[Term] = []
    sign, factors = 1, None
    for token in formula.split():
        if token in ("+", "-"):
            sign, factors = (1 if token == "+" else -1), None
        else:
            if factors is None:
                factors = []
                terms.append((sign, factors))
            factors.append(token)
    return terms


def _pairs(formula: str) -> List[str]:
    """Role pairs of the S_ab factors in a formula."""
    return [factor[2:] for _, factors in _terms(formula) for factor in factors if factor.startswith("S_")]


def _value(factor: str, S: Overlap, values: Dict[str, float]) -> float:
    return S(factor[2:]) if factor.startswith("S_") else values[factor]


def _evaluate(formula: str, S: Overlap, values: Dict[str, float]) -> float:
    return sum(sign * prod(_value(f, S, values) for f in factors) for sign, factors in _terms(formula))


def _substitute(formula: str, S: Overlap, values: Dict[str, float]) -> str:
    """The formula with every factor replaced by its value: '(1)(1) - (0.1263)(0.1267)'."""
    def number(factor: str) -> str:
        if factor.startswith("S_"):
            pair = factor[2:]
            return "1" if pair[0] == pair[1] else f"{S(pair):.4f}"
        return f"{values[factor]:+.5f}"

    text = ""
    for n, (sign, factors) in enumerate(_terms(formula)):
        if n or sign < 0:
            text += (" + " if sign > 0 else " - ") if n else "-"
        text += "".join(f"({number(f)})" for f in factors)
    return text


def _expand(formula: str, definitions: Dict[str, str]) -> str:
    """Replace names defined by a single product (Pi = M1 M2) by that product."""
    return " ".join(
        definitions[token] if token in definitions and len(_terms(definitions[token])) == 1 else token
        for token in formula.split()
    )


# --- Rules -------------------------------------------------------------------


class ChannelRule(NamedTuple):
    name: str  # "closed-shell doublet", "D1", "D2" or "Q"
    weight: str  # "", "2", "1/3", "2/3"
    amplitude: str  # formula inside |...|^2

    def formula(self, amplitude: Optional[str] = None) -> str:
        return (f"{self.weight} " if self.weight else "") + f"|{amplitude or self.amplitude}|^2"


class Rule(NamedTuple):
    """Named quantities, evaluated in order, and the bright channels per initial spin."""

    case: str
    definitions: Tuple[Tuple[str, str], ...]  # (name, formula)
    channels: Dict[float, Tuple[ChannelRule, ...]]  # initial spin -> channels (empty: dark)

    @property
    def overlaps(self) -> List[str]:
        """Role pairs used, in order of first appearance (diagonals included)."""
        formulas = [formula for _, formula in self.definitions]
        formulas += [c.amplitude for channels in self.channels.values() for c in channels]
        return list(dict.fromkeys(pair for formula in formulas for pair in _pairs(formula)))

    def evaluate(self, S: Overlap) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, formula in self.definitions:
            values[name] = _evaluate(formula, S, values)
        return values


_M = {
    "M1": "S_ii S_jj - S_ij S_ji",
    "M2": "S_ij S_kk - S_ik S_kj",
    "M3": "S_ij S_jk - S_ik S_jj",
    "M4": "S_ii S_kj - S_ij S_ki",
}
_ES_CHANNELS = {
    0.0: (ChannelRule("D1", "", "Pi + Omega"),),
    1.0: (ChannelRule("D2", "1/3", "Pi - Omega"), ChannelRule("Q", "2/3", "Pi - Omega")),
}

RULES: Dict[str, Rule] = {
    "3e GS": Rule("3e GS", (), {0.0: (ChannelRule("closed-shell doublet", "2", "S_ii S_ik"),), 1.0: ()}),
    "5e GS": Rule("5e GS", (("M1", _M["M1"]), ("M3", _M["M3"])),
                  {0.0: (ChannelRule("closed-shell doublet", "2", "M1 M3"),), 1.0: ()}),
    "3e ES": Rule("3e ES", (("Pi", "S_ii S_jk"), ("Omega", "S_ik S_ji")), _ES_CHANNELS),
    "5e ES": Rule("5e ES", tuple(_M.items()) + (("Pi", "M1 M2"), ("Omega", "M3 M4")), _ES_CHANNELS),
}


# --- Evaluation on a shake-down ---------------------------------------------


class Channel(NamedTuple):
    name: str  # "closed-shell doublet", "D1", "D2" or "Q"
    formula: str  # e.g. "|Pi + Omega|^2"
    intensity: float


class PropensityResult(NamedTuple):
    shake_down: ShakeDown
    initial_spin: Optional[float]
    roles: Dict[str, str]  # role -> initial orbital ID
    overlaps: Dict[str, float]  # role pair -> S used (diagonals = 1)
    diagonals: Dict[str, float]  # role pair ("ii", ...) -> actual |S|
    amplitudes: Dict[str, float]  # named quantities of the rule: M1.., Pi, Omega
    channels: List[Channel]  # non-zero channels for this initial spin
    note: Optional[str] = None  # why no channel is bright


class RoleOverlaps:
    """S_ab = |<phi_final(a)|psi_b>| for the orbital roles of one shake-down."""

    def __init__(self, roles: Dict[str, str], mapping: OrbitalMapping) -> None:
        self.roles = roles
        self.mapping = mapping

    def actual(self, pair: str) -> float:
        a, b = pair
        return abs(self.mapping.overlap(self.roles[b], self.mapping.final_id(self.roles[a])))

    def __call__(self, pair: str) -> float:
        return 1.0 if pair[0] == pair[1] else self.actual(pair)

    def describe(self, pair: str) -> str:
        a, b = pair
        final = self.mapping.final_id(self.roles[a])
        return f"<final {final} ({a}) | initial {self.roles[b]} ({b})>"


def _roles(shake_down: ShakeDown) -> Dict[str, str]:
    return {role: orbital for role, orbital in
            (("i", shake_down.i), ("j", shake_down.j), ("k", shake_down.k)) if orbital is not None}


def _initial_spin(shake_down: ShakeDown, initial: Calculation) -> Optional[float]:
    return next(s.spin for s in initial.states if s.state_id == shake_down.initial_state)


def channel_rules(shake_down: ShakeDown, initial: Calculation) -> Tuple[Tuple[ChannelRule, ...], Optional[str]]:
    """The bright channels of a computable shake-down for its initial spin, or a note why there are none."""
    rule = RULES[shake_down.case]
    spin = _initial_spin(shake_down, initial)
    if spin not in rule.channels:
        return (), f"no rule for initial spin S = {spin}"
    if not rule.channels[spin]:
        return (), "dark: a triplet has no ground-state shake-down channel"
    return rule.channels[spin], None


def compute_propensity(
    shake_down: ShakeDown, initial: Calculation, mapping: OrbitalMapping, final_spin: Optional[float] = None
) -> PropensityResult:
    """Evaluate the rule of a computable shake-down and its non-zero channels.

    With ``final_spin`` only the channels to final states of that spin are
    evaluated: ``mapping`` then belongs to the final calculation of that spin.
    """
    rule = RULES[shake_down.case]
    roles = _roles(shake_down)
    S = RoleOverlaps(roles, mapping)
    spin = _initial_spin(shake_down, initial)
    diagonals = {role * 2: S.actual(role * 2) for role in roles}
    overlaps = {pair: S(pair) for pair in rule.overlaps}
    values = rule.evaluate(S)

    channels: List[Channel] = []
    bright, note = channel_rules(shake_down, initial)
    for channel in bright:
        if final_spin is not None and CHANNEL_SPIN[channel.name] != final_spin:
            continue
        weight = Fraction(channel.weight) if channel.weight else 1
        amplitude = _evaluate(channel.amplitude, S, values)
        channels.append(Channel(channel.name, channel.formula(), float(weight) * amplitude ** 2))
    return PropensityResult(shake_down, spin, roles, overlaps, diagonals, values, channels, note)


def evaluate_final(
    shake_downs: Sequence[ShakeDown], initial: Calculation, final: MappedFinal
) -> List[PropensityResult]:
    """The computable shake-downs with a channel to ``final``'s spin, evaluated with its overlaps."""
    return [
        compute_propensity(sd, initial, final.mapping, final.spin)
        for sd in shake_downs
        if sd.computable and any(CHANNEL_SPIN[c.name] == final.spin for c in channel_rules(sd, initial)[0])
    ]


def rule_lines(result: PropensityResult, mapping: OrbitalMapping) -> List[str]:
    """The rule of one shake-down written out in full, with the overlaps substituted."""
    rule = RULES[result.shake_down.case]
    S = RoleOverlaps(result.roles, mapping)
    definitions = dict(rule.definitions)
    lines = [f"  {name} = {formula} = {_substitute(formula, S, result.amplitudes)} = {result.amplitudes[name]:+.5f}"
             for name, formula in rule.definitions]
    by_name = {channel_rule.name: channel_rule for channel_rule in rule.channels.get(result.initial_spin, ())}
    for channel in result.channels:
        channel_rule = by_name[channel.name]
        expanded = _expand(channel_rule.amplitude, definitions)
        formula = channel_rule.formula()
        if expanded != channel_rule.amplitude:
            formula += f" = {channel_rule.formula(expanded)}"
        numbers = channel_rule.formula(_substitute(channel_rule.amplitude, S, result.amplitudes))
        lines.append(f"  {channel.name}: {formula} = {numbers} = {channel.intensity:.6f}")
    return lines


def _number(state_id: str) -> str:
    return state_id.split(":")[-1]


def _result_lines(result: PropensityResult, mapping: OrbitalMapping) -> List[str]:
    """One shake-down: roles, overlaps by orbital, diagonal checks, the rule in full."""
    sd = result.shake_down
    S = RoleOverlaps(result.roles, mapping)
    roles = "   ".join(
        f"{role} = {orbital} -> final {mapping.final_id(orbital)}" for role, orbital in result.roles.items()
    )
    lines = [
        "",
        f"{_number(sd.initial_state)} -> {_number(sd.lower_state)}   {sd.case}   "
        f"initial S = {result.initial_spin:g}" + (f"   (d = {sd.d} omitted)" if sd.d else ""),
        f"  {roles}",
    ]
    lines += [f"  S_{pair} = |{S.describe(pair)}| = {value:.4f}" for pair, value in result.overlaps.items()
              if pair[0] != pair[1]]
    for pair, value in result.diagonals.items():
        flag = f"   below {DIAGONAL_WARNING}: weak-relaxation approximation poor" if value < DIAGONAL_WARNING else ""
        lines.append(f"  S_{pair} = |{S.describe(pair)}| = {value:.4f} (set to 1){flag}")
    return lines + rule_lines(result, mapping)


def propensity_report(
    shake_downs: Sequence[ShakeDown],
    initial: Calculation,
    evaluations: Sequence[Tuple[MappedFinal, Sequence[PropensityResult]]],
) -> List[str]:
    """For each final calculation the rules evaluated with its overlaps, then one
    summary of every computable shake-down's channels across the final calculations."""
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
    for final, results in evaluations:
        lines += ["", final.title, "-" * len(final.title)]
        if not results:
            lines.append(f"No shake-down has a channel to S = {final.spin:g} final states.")
        for result in results:
            lines += _result_lines(result, final.mapping)

    results_by_final = {
        final.label: {result.shake_down: result for result in results} for final, results in evaluations
    }
    finals_by_spin = {final.spin: final for final, _ in evaluations}
    lines += ["", "Summary", f"{'a -> b':>8}  {'Case':<6}  {'Final':<8}  {'Channel':<21}  {'Intensity':>10}"]
    for sd in shake_downs:
        if not sd.computable:
            continue
        pair = f"{_number(sd.initial_state)} -> {_number(sd.lower_state)}"
        bright, note = channel_rules(sd, initial)
        if not bright:
            lines.append(f"{pair:>8}  {sd.case:<6}  {'-':<8}  {'-':<21}  {'-':>10}   {note}")
        for channel_rule in bright:
            spin = CHANNEL_SPIN[channel_rule.name]
            final = finals_by_spin.get(spin)
            if final is None:
                lines.append(f"{pair:>8}  {sd.case:<6}  {'-':<8}  {channel_rule.name:<21}  {'-':>10}"
                             f"   no S = {spin:g} final calculation")
                continue
            channel = next(c for c in results_by_final[final.label][sd].channels if c.name == channel_rule.name)
            lines.append(f"{pair:>8}  {sd.case:<6}  {final.label:<8}  {channel.name:<21}  {channel.intensity:10.6f}")
    return lines
