---
status: accepted
---

# CHP is first order in a Coulomb-only core-hole potential

The core-hole potential (CHP) method relaxes the initial orbitals in v_ch = −J_c of the core orbital c, to first order: S_pq = −(pq|cc)/(ε_p − ε_q) for p ≠ q and S_pp = 1, with ε the initial orbital energies and (pq|cc) from an FCIDUMP over the initial active space (`docs/chp.tex`). We chose this as the simplest model that needs only the initial calculation. To be revisited: the all-orders form is planned, after its theory has been worked through.

## Considered Options

- **Hole shift** (ε_p → ε_p − (pp|cc), optionally + ½(pc|pc)) and **all-orders CHP** (diagonalise diag(ε) − J_c, optionally + ½K_c): both reduce the overestimate described below without any orbital transformation. Mean |S| over the core orbitals for the bromothiophene singlet, against the overlaps method:

  | Pair | Overlaps method | First order | Hole shift | All orders |
  |---|---|---|---|---|
  | 34/38 | 0.126 / 0.127 | 0.38 | 0.11 | 0.18 / 0.095 |
  | 34/40 | 0.0059 / 0.022 | 0.028 | 0.033 | 0.0063 / 0.027 |

  The ½K_c term changed little.

## Consequences

- With Coulomb only, S is the same for every final spin; doublet and quartet differ only through the channel formulas.
- The sign of v_ch matters: it sets the relative sign of Pi (first order in v_ch) and Omega (second order), and so the D1 versus D2/Q interference.
- The ε are diagonal generalised-Fock elements of SA natural orbitals, not canonical orbital energies, and their order is not guaranteed.
- The hole shifts levels unevenly (e.g. mo:38 from −0.33 to −0.74 Eh, next to mo:39), so initial-state gaps can badly overestimate the mixing; |S| above 0.5 is flagged as unreliable.
