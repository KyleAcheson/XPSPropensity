# XPSoverlaps

Approximates XPS shake-down intensities (Dyson norms) from orbital overlaps between
an initial N-electron state and a core-ionised (N−1)-electron final state, using
propensity rules. The theory is in `docs/shakedown_cases.tex` and `docs/chp.tex`.

## Language

### States and configurations

**Initial state**:
An N-electron state before ionisation, usually a valence-excited state written as a single excitation from the reference determinant.
_Avoid_: ground state (unless it is the reference), neutral state

**Final state**:
The core-ionised (N−1)-electron state: one electron removed from the core orbital c, valence occupation as in the shake-down target b.
_Avoid_: cation state, ionised state

**Reference determinant**:
The closed-shell determinant (each active orbital 2 or 0) from which initial states are written as excitations.
_Avoid_: ground-state configuration (it need not be a computed state)

**Pump**:
The single excitation i → k from the reference determinant that defines an initial state a.
_Avoid_: excitation (when it is ambiguous which one)

**Dominant configuration**:
The CSF with the largest weight in a state's CI expansion; it stands for the whole state in the propensity rules.
_Avoid_: dominant determinant, main configuration

### Orbitals

**Initial orbital (ψ)**:
An orbital of the initial calculation.
_Avoid_: writing it as φ or ϕ

**Final orbital (φ)**:
An orbital relaxed in the core-ionised final state: from the final calculation (overlaps method) or from the initial orbital relaxed in the core-hole potential (CHP).
_Avoid_: relaxed orbital written as ψ

**Orbital overlap (S_ab)**:
S_ab = ⟨φ_a|ψ_b⟩; the first index is always the final (relaxed) orbital, the second the initial orbital. Absolute values in the overlaps method, signed in CHP.

**Core orbital (c)**:
A RAS1 orbital from which the XPS electron is removed.

**Valence orbitals**:
The RAS2 orbitals; the only orbitals that take orbital roles i, j, k, d.

**SA orbitals**:
The natural orbitals of the state-averaged density of an SA-RASSCF calculation; every orbital-level quantity uses these.
_Avoid_: "the orbitals" when a state-specific set could be meant

**Reference orbitals**:
The SA-RASSCF orbitals in which a CASPT2 calculation's CI vectors are expressed.
_Avoid_: CASPT2 orbitals, PT2 orbitals

**State-specific natural orbitals**:
Natural orbitals of one state's density taken from the SA calculation (for MS-CASPT2, the MS-mixed density). Not used at present.
_Avoid_: state-specific orbitals (ambiguous with SS-RASSCF orbitals), SS orbitals

**SS-RASSCF orbitals**:
Orbitals optimised separately for one state in its own RASSCF calculation. Not used (see ADR-0001).

**Orbital order**:
The ranking of valence orbitals as HOMO, HOMO−1, … and LUMO, LUMO+1, …, by SA populations.
_Avoid_: energy order, orbital ladder

**Orbital mapping**:
The correspondence between initial and final active orbitals, by largest overlap.
_Avoid_: reordering, matching (as a verb that suggests changing the calculation)

### Shake-downs

**Shake-down**:
A transition from initial state a to a lower-energy initial state b that differs by one moved electron, seen through the final state built on b.
_Avoid_: relaxation, decay

**Shake-down electron**:
The electron that moves between a and b; where it comes from sets the case.

**Orbital roles**:
The labels c, i, k, j, d given to orbitals in one shake-down: c core; i pump hole; k pump particle; j receives (3e ES) or gives (5e ES) the shake-down electron, or is the HOMO in 5e GS; d the HOMO spectator in ES cases.
_Avoid_: roles (bare)

**Pump hole (i)** / **Pump particle (k)**:
The orbital emptied / filled by the pump.

**Spectator (d)**:
The HOMO when it is none of i, j in an ES case; doubly occupied throughout and left out of the rules.

**Case**:
One of 3e GS, 5e GS, 3e ES, 5e ES, set by where the shake-down electron comes from. GS: b is the reference determinant; ES: b is an excited state. 3e/5e: the number of electrons the final state has in the orbitals the rule involves (c, i, k and, if present, j).

**Not computable**:
A shake-down that is listed but not evaluated, e.g. because its electron would move up in the orbital order or a role falls on a core or RAS3 orbital.

### Propensity rules

**Dyson norm**:
|⟨Φ_F|â_c|Ψ_I⟩|², the intensity being approximated.

**Propensity rule**:
An approximation of a Dyson norm by products of orbital overlaps, for one case and channel.
_Avoid_: selection rule

**Channel**:
A final spin coupling reachable from an initial spin: D1 (singlet, ES), D2 and Q (triplet, ES), and the closed-shell doublet (singlet, GS). Each has a weight and an amplitude.
_Avoid_: transition (for a channel)

**Pi (direct)** / **Omega (exchange)**:
The two terms of an ES amplitude: D1 = |Pi + Omega|², D2 = ⅓|Pi − Omega|², Q = ⅔|Pi − Omega|².

**Minors (M1–M4)**:
The 2×2 minors of the valence overlap block that the 5e rules are written in.

### Methods

**Overlaps method**:
Obtains S from an initial and a final calculation through the orbital mapping.

**Core-hole potential (CHP)**:
Obtains S from the initial calculation alone: initial orbitals relaxed in v_ch = −J_c of the core orbital, to first order, S_pq = −(pq|cc)/(ε_p − ε_q) (see ADR-0003).
_Avoid_: core-hole approximation, frozen-orbital method

**Hole shift**:
Replacing ε_p by ε_p − (pp|cc) in the CHP denominators, so the gaps are those of the core-ionised state.

**All-orders CHP (F − J_c)**:
Diagonalising diag(ε) − J_c in the initial orbitals; the eigenvectors are the relaxed orbitals to all orders in v_ch.
_Avoid_: exact CHP
