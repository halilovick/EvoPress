# Exact-budget replay batch: findings (2026-10-09)

Run: 97/97 evaluations completed on one A40 (test batch of 3 on 2026-10-08, then
four chunks 20:51–06:37 UTC). Mean 359 s per evaluation (331–431 s). Rebuilt
environment with the September versions (torch 2.8.0, transformers 4.56.2,
flash-attn 2.8.3, datasets 5.0.1; `../environment_freeze.txt`). All quantized
candidates meet T = 26,982,023,168 bits exactly. All 8 re-evaluated finals (E1,
E3 s0–2, J12 s0–2, DW s0) reproduce the recorded W2, C4 and calibration KL **exactly**
(`reproduction.md`).

Tables: `rq1_baselines.md`, `rq3_contrasts.md`. Numbers below are taken from them.
Interpretation is preliminary and must be checked with the supervisor before it
enters the thesis.

## RQ1: joint search against simpler compression at the same budget

| s | Configuration | W2 PPL | C4 PPL | Calib. KL |
| --- | --- | --- | --- | --- |
| 0 | Quantization-only search (E2) | 5.229 ± 0.005 | 8.411 ± 0.007 | 0.0731 |
| 0 | Uniform 3-bit (E1) | 5.500 | 8.555 | 0.0904 |
| 0.125 | Joint search (J12) | 6.621 ± 0.202 | 9.646 ± 0.063 | 0.2084 |
| 0.125 | Best heuristic: block-influence mask + near-uniform | 8.102 | 11.289 | 0.3462 |
| 0.125 | Random masks + near-uniform (3) | 20.16 ± 4.04 | 26.26 ± 6.77 | 1.053 |
| 0.25 | Joint search (E3) | 8.961 ± 0.115 | 12.497 ± 0.129 | 0.4613 |
| 0.25 | Independent: depth-only masks + E2 profile shifted (3) | 9.703 ± 0.550 | 14.268 ± 0.829 | 0.5711 |
| 0.25 | Heuristic masks + near-uniform (6 masks) | 28 to 40,192 | 38 to 6,876 | 1.36 to 6.31 |

1. **Joint search beats every heuristic and independent baseline at the same
   budget.** At 25 %, all six heuristic masks fail (W2 ≥ 28). The block-influence
   score removes a contiguous block of late sublayers (attention 21–28, MLP 22–29)
   and collapses (W2 5,316), although its 12.5 % mask (four of each) is the best
   heuristic (8.10). Sublayer scores computed one at a time do not compose when many
   neighbouring sublayers are removed. This is a concrete instance of the
   independence/additivity problem of Chapter 2.
2. **The independent composition is worse than joint search** by about 0.74 W2,
   1.77 C4 and 0.110 KL on average. The difference already exists in 16 bits: the
   depth-only masks give 9.30 ± 0.50 W2 in 16-bit precision, the joint masks
   8.61 ± 0.11. The joint searches found better masks. Caveat: the depth-only masks
   come from the cheap screening search (8,192 WikiText-2 tokens, G20/O16), so this
   comparison is not matched in search effort.
3. **The overall ordering at fixed storage is unchanged:** E2 < E1 < J12 < E3. Removing
   structure to fund precision does not pay off for Mistral-7B at 4.3×.
4. **Where the loss comes from** (same masks, 16-bit vs at T): 25 %: dense 4.83 → 16-bit
   mask 8.61 (+3.78) → at T 8.96 (+0.35); 12.5 %: 4.83 → 6.10 (+1.27) → 6.62 (+0.52).
   Removal accounts for most of the degradation; quantizing the remaining
   projections at the reinvested precision costs little. The measured 16-bit value
   of our 25 % masks (8.61 / 12.09) is close to the supplied published depth-only
   reference (8.66 / 12.04), but it is not the same mask.

## Precision allocation for a fixed joint mask (RQ1/RQ3)

| Mask | W2: own / E2 shifted / near-uniform | KL: own / E2 shifted / near-uniform |
| --- | --- | --- |
| E3 s0 | 8.945 / 9.055 / 9.125 | 0.4670 / 0.4726 / 0.4755 |
| E3 s1 | 8.828 / 8.828 / 8.914 | 0.4609 / 0.4643 / 0.4694 |
| E3 s2 | 9.109 / 9.055 / 9.195 | 0.4560 / 0.4550 / 0.4621 |
| J12 s0 | 6.414 / 6.516 / 6.801 | 0.1992 / 0.2098 / 0.2270 |
| J12 s1 | 6.895 / 6.695 / 6.996 | 0.2154 / 0.2312 / 0.2489 |
| J12 s2 | 6.555 / 6.684 / 7.035 | 0.2105 / 0.2200 / 0.2392 |

5. **The jointly found allocation is better than the near-uniform one in all six
   masks (W2 and KL).**
6. **It is only marginally better than the mask-independent E2 profile shifted to
   the budget:**
   * on the search objective (KL) in 5 of 6 masks;
   * on W2 in 3 of 6, tied in 1, worse in 2.

   Given the mask, a quantization-only allocation works almost as well as the joint
   one. The joint search's advantage comes mainly from the mask.

## RQ3: interaction contrasts (I < 0: each mask prefers its own allocation)

7. **Repair-free shared-gene exchanges** (E3 s1 × DW s0; J12 s0 × s1):
   * |I| ≤ 0.014 in log-PPL (about 1.4 % PPL) and ≤ 0.006 in KL;
   * the sign disagrees across metrics in both pairs (W2 −/+, C4 +/−, KL −/−).

   There is no consistent interaction in the cleanest measurement.
8. **The plain crossing (D_A, Q_B) of the same two pairs** gives much more negative
   contrasts:
   * KL −0.020 and −0.029, 5–7× the shared-gene values;
   * C4 −0.019 and −0.043.

   This happens although only 1–4 genes are repaired. For the J12 pair, the spread
   over repair seeds (0.053 W2 log-PPL) is larger than the contrast itself. Stale
   genes and repair inflate the naive interaction measure.
9. **Repaired shared-gene exchanges (5 seed pairs)** give negative I on all three
   metrics in all five pairs (KL −0.006 to −0.021). These are larger than the spread
   over repair seeds. However, the repair changes 2–6 of the mask owner's
   optimized genes in *both* crossed candidates. Any perturbation of an optimized
   configuration tends to raise J, which makes both δ positive and I negative. The
   replays contain no control for the size of that effect, so these contrasts are
   an upper bound on the interaction, not evidence for it. KL is also measured on
   the calibration data the searches optimized (in-sample).
10. **Combined reading:** findings 6–9 point the same way. Given the mask, the
    precision allocation matters little, and the measurable dependence of the
    allocation on the mask is small. Within the precision of these measurements
    (3 seeds, two repair-free pairs), the data do not support a strong interaction
    between structural and quantization decisions.

## Caveats for the thesis text

* n = 3 seeds per condition; two repair-free pairs; heuristic baselines use one
  fixed (near-uniform) precision rule.
* Independent depth-only masks are from a much cheaper search than the joint runs.
* W2 and KL can disagree for single candidates (e.g. J12 s1 mask: other seeds'
  allocations are better on W2 but worse on KL). Report both.
* The replays were run in a rebuilt environment; exact reproduction of all 8
  originals is the evidence that they are comparable.
