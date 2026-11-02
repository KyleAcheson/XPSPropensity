---
status: accepted
---

# All-orders CHP diagonalises diag(ε) − J_c over the valence orbitals

All-orders CHP takes S from the eigenvectors of diag(ε) − J_c, S_pq = U_qp, with (J_c)_pq = (pq|cc) of one core orbital c, over the valence (RAS2 and, if there is one, RAS3) initial orbitals (`docs/chp_all_orders.tex`). It is the all-orders sum of the first-order CHP of ADR-0003 and rests on the same model: the SA natural orbitals are treated as eigenfunctions of diag(ε), although the Fock matrix is not diagonal in them (F_pq up to 0.013 Eh in the bromothiophene valence block, against (pq|cc) up to 0.15 Eh). Each eigenvector is assigned to an initial orbital by all the rules of the orbital mapping in the overlaps method (largest overlap, contested orbitals by overlap × AO-composition similarity, the same warnings), with the phase S_aa > 0, so the two methods label relaxed orbitals the same way. A relaxed orbital is reported by the ID of the initial orbital it is assigned to, marked (e.g. `mo:38*`) when the assignment draws a warning. First-order CHP stays the default; all orders is chosen in the input (`order: all`).

## Considered Options

- **Full-Fock CHP (F − J_c)**, F = F_I + F_A from the FCIDUMP: deferred, not rejected. Against the SA natural orbitals it gives S ≠ 1 at J_c = 0, blaming the hole for mixing F causes itself; against the eigenvectors of F, the initial orbitals are no longer those the CI vectors are expressed in, which ADR-0001 and ADR-0002 rule out. It changes S by 0.01–0.04 for bromothiophene. To be revisited if the canonical-orbital assumption proves too crude, together with orbital transformations.
- **Following the eigenvectors in λ** (diagonalise diag(ε) − λJ_c from λ = 0 to 1, matching each step to the last): deferred. It gives the analytic continuation of the perturbation series, and its labels change continuously with the data, but it can label differently from the orbital mapping: for bromothiophene (core mo:31) the hole moves ψ_38 through an avoided crossing with ψ_36, the two labellings exchange φ_36 and φ_38, and S_38,34 is +0.369 followed against −0.096 by largest overlap.
- **Including the core (RAS1) orbitals**: deferred. The core orbitals take no orbital role and their effect on the valence S is small; to be revisited if the core-valence separation is dropped.
- **RAS2 only**: rejected. Unlike first order, all-orders S between two RAS2 orbitals depends on every coupled orbital, so relaxation into RAS3, part of the valence space in core-spectroscopy RASSCF, is kept. Rejecting RAS3 calculations in the loader was also considered and rejected.
- **−J_c + ½K_c** (spin-averaged exchange): deferred. It changed S little and would make S depend on the final spin.

## Consequences

- S is orthogonal, not antisymmetric, and S_aa < 1; the propensity rules still set S_aa to 1 and report the actual values, as for the other methods.
- The hole level shift −(pp|cc) is included, so the first-order failure at strongly shifted levels (e.g. mo:38 of bromothiophene, |S| up to 0.78) is removed, but strongly mixed pairs remain (S_38,38 ≈ 0.6).
- Both orders use the same (pq|cc) block from the integral file, so all orders needs no new input files.
- All orders needs the AO compositions of the relaxed orbitals, and so the initial calculation's AO basis and an AO overlap provider, which first order does not use.
