# Result interpretation and complete numerical evidence

The manuscript uses conclusion-first result paragraphs. Tables retain the full single-fault comparisons; this artifact preserves the numerical details and simultaneous intervals that were shortened in the prose.

| Scientific question | Evidence within `reproduction/ISCAS/research/review-completion-2026-10-12/results/` |
|---|---|
| Does representation or prior explain forward loss? | `full-model-fixed/`, `joint-bridge/`, `onefactor/`, `decoder-comparisons.json` |
| Does physical ambiguity imply a two-fault witness? | `circuit-distance-audit.json`, `distance-two-certificates.json` |
| When do multiple faults determine performance? | `strata/`, `low-noise-summary.json`, `low-noise-comparisons.json`, `direct/` |
| Do selected schedules generalize to fresh samples? | `order-space/result.json`, `order-space/comparisons.json`, `order-space/confirmation/` |
| Do gate count, idling, and extraction distance explain reliability? | `deployment/`, `sensitivity/`, `idle-reweighted-risk.json`, `native/`, `hirai/`, `hirai-excess/` |
| Can order matter when all optimal single-fault risks vanish? | `bb/`, `bb-comparisons.json`, `bb-memory-distance-lower-bound.json` |
| What changes when prepared auxiliaries and nonlocal couplers are allowed? | `zhao/`, `baseline-comparisons.json` |

Historical primary bridge data and the paired 8192-shot experiment are in `reproduction/ISCAS/research/mechanism-intervention-2026-10-11/results/`. Figure 2 and Table I summarize different interventions: the figure searches 64 orders with the initial block fixed, while the historical macro reversal changes all four blocks.

Representative details shortened from the manuscript:

- Complete-model forward risk with fixed strong prior is 123.600; the matched-prior risk equals 115.200. Table I carries the original and strong sliding risks without repeating them in the prose.
- Historical original-only/strong-only discordant counts are 140/72 for forward and 72/104 for macro reversal. Exact paired p-values are 3.49e-6 and 0.0192. These are paired decoder comparisons, separate from independent circuit comparisons.
- Matching the second-instance strong prior reduces failures from 1042 to 795, 971 to 723, and 1064 to 835 for forward, macro reversal, and check reversal. All pass the 54-test Holm family.
- At 1e-6, second-instance macro-minus-forward differences are +1.856e-3 under original decoding and -2.560e-4 under strong decoding. The simultaneous intervals exclude zero. At 1e-4, macro reversal increases original failure by 2.05 percentage points, interval [0.211, 3.883]. Strong check-reversal differences instead favor forward, interval [0.113, 3.542] percentage points.
- Primary check reversal improves both sliding rules but worsens the lookup at 1e-6, interval [3.643, 3.677]e-5.
- Native distance-three/five extraction uses 160/576 CX and 28 slots; both optimal risks vanish. Normal Hirai complete-model excess is 0.1333 for original decoding and zero for strong decoding.
- BB-108 check reversal at 1e-7 decreases strong failures from 231 to 67 and lookup failures from 183 to 67, even though every BB single-fault lookup risk is zero. The finite-noise lookup includes multiple-fault record collisions and fallback decisions.

All counts, sample arrays, intervals, and original computational identities are retained. No experiment was rerun or numerical result changed to simplify the manuscript.

