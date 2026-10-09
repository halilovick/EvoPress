# Full-space zero-shot LM-eval (planned 2026-10-09, not yet run)

Eleven models at the exact budget T = 26,982,023,168 bits (except the dense
reference): dense (E0), uniform 3-bit (E1), quantization-only search E2 seeds
0–2, joint search at s = 0.125 (J12) and s = 0.25 (E3) seeds 0–2. Final
candidates are used exactly as stored (no repair); `lmeval_plan.json` is written
by `scripts/plan_exact_lmeval.py` and its jobs are identical to the tier-0 jobs
of the replay batch that reproduced the recorded perplexities exactly.

Settings (same as the screening LM-eval runs, `lmeval.py`): ARC-Easy, PIQA,
WinoGrande; 0-shot; batch size 4; float16; SDPA attention; fast tokenizer;
full evaluation sets; lm_eval 0.4.13; harness default seeds. Reported score:
acc_norm for ARC-Easy and PIQA, acc for WinoGrande (screening convention);
acc is kept as well.

Run: `evo_exact_lmeval.py` (one model load, candidates assembled with the
replay code). Verify and summarize: `scripts/summarize_exact_lmeval.py`
(completeness, identical settings, full task sets, realized cost, bit-width
hashes against the replay batch, dense model against the screening dense
LM-eval). Commands: `thesis_results/GPU_RUNBOOK_2026-10-09.md`.
