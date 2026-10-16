"""Propensity rules for shake-down transitions (docs/shakedown_cases.pdf).

Overlaps follow S_ab = |<phi_a|psi_b>|: the FIRST index is a final-state
orbital, the final orbital the mapping matches to initial orbital a; the
second is the initial-state orbital b. Absolute values are used throughout.
Diagonal overlaps S_ii, S_jj, S_kk are set to 1 (weak relaxation); their
actual values are reported and flagged below ``DIAGONAL_WARNING``.

Which channels are non-zero depends only on the initial spin (averaged over
initial M_S, summed over final M_S):

    singlet:  GS -> closed-shell doublet 2|S_ik|^2;  ES -> D1 |Pi + Omega|^2
    triplet:  GS -> dark;  ES -> D2 1/3 |Pi - Omega|^2,  Q 2/3 |Pi - Omega|^2
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from xpsoverlaps.mapping import OrbitalMapping
from xpsoverlaps.model import Calculation
from xpsoverlaps.shakedown import ShakeDown

DIAGONAL_WARNING = 0.9

# Final-state spin of each channel, to check it exists in the final calculation.
_CHANNEL_SPIN = {"closed-shell doublet": 0.5, "D1": 0.5, "D2": 0.5, "Q": 1.5}

Overlap = Callable[[str], float]  # role pair such as "ik" -> S_ik


class Rule(NamedTuple):
    """Pathway amplitudes Pi (direct) and Omega (exchange) of one case."""

    case: str
    overlaps: Tuple[str, ...]  # role pairs used, e.g. ("jk", "ik", "ji")
    direct: str  # formula of Pi
    exchange: str  # formula of Omega
    amplitudes: Callable[[Overlap], Dict[str, float]]  # S -> {"Pi": .., "Omega": .., extras}


def _three_electron(S: Overlap) -> Dict[str, float]:
    return {"Pi": S("jk"), "Omega": S("ik") * S("ji")}


def _five_electron(S: Overlap) -> Dict[str, float]:
    m1 = 1 - S("ij") * S("ji")
    m2 = S("ij") - S("ik") * S("kj")
    m3 = S("ij") * S("jk") - S("ik")
    m4 = S("kj") - S("ij") * S("ki")
    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "Pi": m1 * m2, "Omega": m3 * m4}


RULES: Dict[str, Rule] = {
    "3e ES": Rule("3e ES", ("jk", "ik", "ji"), "S_jk", "S_ik S_ji", _three_electron),
    "5e ES": Rule("5e ES", ("ij", "ji", "ik", "ki", "jk", "kj"), "M1 M2", "M3 M4", _five_electron),
}


class Channel(NamedTuple):
    name: str  # "closed-shell doublet", "D1", "D2" or "Q"
    formula: str
    intensity: float


class PropensityResult(NamedTuple):
    shake_down: ShakeDown
    initial_spin: Optional[float]
    roles: Dict[str, str]  # role -> initial orbital ID
    overlaps: Dict[str, float]  # role pair -> S used (diagonals = 1)
    diagonals: Dict[str, float]  # role pair ("ii", ...) -> actual |S|
    amplitudes: Dict[str, float]  # Pi, Omega (and M1..M4 for 5e)
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


def compute_propensity(
    shake_down: ShakeDown, initial: Calculation, mapping: OrbitalMapping
) -> PropensityResult:
    """Dispatch the rule for a computable shake-down and evaluate its non-zero channels."""
    roles = {role: orbital for role, orbital in
             (("i", shake_down.i), ("j", shake_down.j), ("k", shake_down.k)) if orbital is not None}
    S = RoleOverlaps(roles, mapping)
    spin = next(s.spin for s in initial.states if s.state_id == shake_down.initial_state)
    diagonals = {role * 2: S.actual(role * 2) for role in roles}

    if shake_down.case == "3e GS":
        overlaps = {"ik": S("ik")}
        amplitudes: Dict[str, float] = {}
        bright = {0.0: [Channel("closed-shell doublet", "2|S_ik|^2", 2 * overlaps["ik"] ** 2)],
                  1.0: []}
    else:
        rule = RULES[shake_down.case]
        overlaps = {pair: S(pair) for pair in rule.overlaps}
        amplitudes = rule.amplitudes(S)
        plus = (amplitudes["Pi"] + amplitudes["Omega"]) ** 2
        minus = (amplitudes["Pi"] - amplitudes["Omega"]) ** 2
        bright = {0.0: [Channel("D1", "|Pi + Omega|^2", plus)],
                  1.0: [Channel("D2", "1/3 |Pi - Omega|^2", minus / 3),
                        Channel("Q", "2/3 |Pi - Omega|^2", 2 * minus / 3)]}

    note = None
    if spin not in bright:
        channels, note = [], f"no rule for initial spin S = {spin}"
    else:
        channels = bright[spin]
        if not channels:
            note = "dark: a triplet has no ground-state shake-down channel"
    return PropensityResult(shake_down, spin, roles, overlaps, diagonals, amplitudes, channels, note)


def propensity_report(
    results: Sequence[PropensityResult], mapping: OrbitalMapping, final: Calculation
) -> List[str]:
    """Overlaps by orbital label, diagonal checks, amplitudes and channel intensities."""
    final_spins = sorted({s.spin for s in final.states if s.spin is not None})

    def number(state_id: str) -> str:
        return state_id.split(":")[-1]

    lines = [
        "Propensity rules",
        "",
        "S_ab = |<phi_a|psi_b>|: a is the final-state orbital matched to initial orbital a,",
        "b is the initial-state orbital. Diagonal S_ii, S_jj, S_kk are set to 1.",
        f"Final calculation spin: S = {', '.join(f'{s:g}' for s in final_spins)}.",
        "Non-zero channels: singlet -> GS closed-shell doublet or ES D1;"
        " triplet -> ES D2 and Q (GS dark).",
    ]
    for result in results:
        sd = result.shake_down
        S = RoleOverlaps(result.roles, mapping)
        roles = "   ".join(
            f"{role} = {orbital} -> final {mapping.final_id(orbital)}" for role, orbital in result.roles.items()
        )
        lines += [
            "",
            f"{number(sd.initial_state)} -> {number(sd.lower_state)}   {sd.case}   "
            f"initial S = {result.initial_spin:g}" + (f"   (d = {sd.d} omitted)" if sd.d else ""),
            f"  {roles}",
        ]
        lines += [f"  S_{pair} = |{S.describe(pair)}| = {value:.4f}" for pair, value in result.overlaps.items()
                  if pair[0] != pair[1]]
        for pair, value in result.diagonals.items():
            flag = f"   below {DIAGONAL_WARNING}: weak-relaxation approximation poor" if value < DIAGONAL_WARNING else ""
            lines.append(f"  S_{pair} = |{S.describe(pair)}| = {value:.4f} (set to 1){flag}")
        if result.amplitudes:
            lines.append("  " + "   ".join(f"{name} = {value:+.5f}" for name, value in result.amplitudes.items()))
        for channel in result.channels:
            missing = "" if _CHANNEL_SPIN[channel.name] in final_spins else (
                f"   (no S = {_CHANNEL_SPIN[channel.name]:g} states in the final calculation)")
            lines.append(f"  {channel.name}: {channel.formula} = {channel.intensity:.6f}{missing}")
        if result.note:
            lines.append(f"  {result.note}")

    lines += ["", "Summary", f"{'a -> b':>8}  {'Case':<6}  {'Channel':<21}  {'Intensity':>10}"]
    for result in results:
        sd = result.shake_down
        pair = f"{number(sd.initial_state)} -> {number(sd.lower_state)}"
        if not result.channels:
            lines.append(f"{pair:>8}  {sd.case:<6}  {'-':<21}  {'-':>10}   {result.note}")
        for channel in result.channels:
            lines.append(f"{pair:>8}  {sd.case:<6}  {channel.name:<21}  {channel.intensity:10.6f}")
    return lines
