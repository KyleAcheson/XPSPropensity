---
status: accepted
---

# Symmetry: one calculation per irrep, finals chosen by (spin, Γ_I × Γ_c)

A calculation run with point-group symmetry holds the states of one irrep, so the input takes a list of Initial blocks as well as of Final blocks, one per irrep. Each block's irrep is read from its files (Molcas `LSYM`), not from the input. The exact selection rule is Γ_F = Γ_I × Γ_c: ionising core orbital c of an initial state of irrep Γ_I reaches only final states of that irrep. The propensity rules model the final state with its core hole in the ionised orbital c. The final state built on b then has irrep Γ_b × Γ_c, so a shake-down a → b needs Γ_a = Γ_b, and moving one electron from p to q gives Γ_b = Γ_a × Γ_p × Γ_q. Each Initial block is therefore analysed on its own: its own SA orbitals, orbital order, reference determinant and shake-downs. Final calculations are shared across Initial blocks. For each core-hole irrep Γ_c among the RAS1 orbitals, the overlaps method uses the Final block keyed by (channel spin, Γ_I × Γ_c). Results are labelled with Γ_c and Γ_F, and CHP gives one such result per core orbital. The derivation is in `docs/shakedown_cases.tex`.

## Cross-core shake-downs are outside the rules

The rules assume the core block of S is the identity (CVS: S_cc = 1, S_cc′ = 0, S_ca = S_ac = 0). A cross-core shake-down leaves the final state's hole in a core orbital c′ other than the ionised c, so under that assumption every cross-core shake-down is zero. Symmetry decides which of those zeros are exact:

| Final hole c′ | Γ_b | Under the rules | In reality |
|---|---|---|---|
| c′ = c | Γ_a | the rule | — |
| c′ ≠ c, Γ_c′ = Γ_c | Γ_a | 0 | can be non-zero if the final core orbitals are rotated relative to the initial ones (near-degenerate 2p_x/2p_y) |
| c′ ≠ c, Γ_c′ ≠ Γ_c | Γ_a | 0 | 0 by symmetry |
| c′ ≠ c, Γ_c′ ≠ Γ_c | ≠ Γ_a, Γ_b × Γ_c′ = Γ_a × Γ_c | 0 | can be non-zero through core–valence overlaps or final-state CI mixing |
| any | ≠ Γ_a, product not matching | 0 | 0 by symmetry |

Rows 2 and 4 are where the assumption can fail, and each gets a check that writes a warning in the log:

- **Row 2:** the overlaps method prints the initial/final RAS1 overlap block per irrep and warns when it is far from the identity. CHP does not relax the core orbitals, so its core block is the identity exactly.
- **Row 4:** for example (Cs), an A″ state ionised from S 2p_z (a″) reaching the A′ final state "ground state + 2p_x hole". The shake-down electron changes irrep, so every rule evaluated with symmetry-blocked overlaps is exactly zero. Its amplitude needs core–valence overlaps (here S_ic′ S_ck) or final-state CI mixing, which the dominant-configuration approximation drops. These pairs are not searched for. Each Initial block of a run with symmetry gets one warning line saying they are not included.

## Considered Options

- **Pairing an initial irrep only with the final of the same irrep**: rejected. The core hole changes the irrep. In Cs, an A′ state ionised from the out-of-plane S 2p (a″) reaches A″ final states, so one Initial block needs every Final block whose irrep is Γ_I × Γ_c for some core orbital.
- **Evaluating cross-core shake-downs with the existing rules**: rejected. The result is exactly zero by symmetry, and a and b come from different SA orbital sets, so their orbital roles are not defined in one orbital set.
- **Listing each row-4 pair as not computed**: rejected. Symmetry alone matches almost every lower-energy pair across Initial blocks (in Cs, every a″ → a′ pair through some core pair). Keeping only one-electron moves would need an orbital mapping between Initial blocks, a lot of machinery for rows that are never computed.
- **One input file per initial irrep**: rejected. The closing summary has to cover all initial irreps of a molecule in one run.

## Consequences

- Final blocks are keyed by (spin, irrep) instead of spin alone. Two Initial blocks with the same irrep, or two Final blocks with the same (spin, irrep), are input errors. An Initial and a Final block may share an irrep.
- With symmetry an orbital ID carries its irrep, as `mo:<index within irrep>_i<irrep number>` (e.g. `mo:34_i1`), and `homo`/`lumo` take the same form (`34_i1`). Calculations without symmetry keep `mo:N`.
- The overlaps method gives one value per core hole. Core orbitals of the same irrep share one final calculation and so one value, with no factor for their number.
- Within the rules, a ground-state shake-down needs Γ_I totally symmetric, since the reference is closed-shell. Initial blocks of other irreps have only excited-state shake-downs there, and their transitions to the ground state are cross-core.
- Non-Abelian groups (degenerate irreps) are outside this rule. Molcas does not run them.
- An Initial block whose active orbitals cannot be mapped onto a final calculation it needs is reported in its own part and skipped. The other blocks and the closing summary are still produced, and the run exits with code 1. The mapping error itself stays a hard error.
