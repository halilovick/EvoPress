# Mistral Downstream LM-Eval Comparison

This summary evaluates whether the Mistral joint compression result transfers from perplexity to downstream multiple-choice tasks. Higher scores are better.

## Macro Average

| Method | Macro score |
| --- | ---: |
| Joint G50 depth + q_proj quant | 0.643 |

## Task Averages

| Method | Task | Metric | Runs | Mean | Std | Min | Max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Joint G50 depth + q_proj quant | arc_easy | acc_norm,none | 3 | 0.602 | 0.006 | 0.598 | 0.609 |
| Joint G50 depth + q_proj quant | piqa | acc_norm,none | 3 | 0.736 | 0.003 | 0.733 | 0.739 |
| Joint G50 depth + q_proj quant | winogrande | acc,none | 3 | 0.591 | 0.008 | 0.582 | 0.597 |

## Paired Deltas

| Task | Seed | Metric | Depth | Independent | Joint G50 | Joint - depth | Joint - independent |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |

## Interpretation Checklist

- If joint G50 improves macro score over independent composition, the joint-search advantage transfers to task accuracy.
- If perplexity improves but task scores do not, report the result as a limitation and keep downstream alignment as future work.
- Limited LM-eval runs are smoke tests only. Final reported task metrics should not use `--limit`.

## Source Runs

| Method | Seed | Run ID | Task | Metric | Score |
| --- | ---: | --- | --- | --- | ---: |
| Joint G50 depth + q_proj quant | 0 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed0` | arc_easy | acc_norm,none | 0.598 |
| Joint G50 depth + q_proj quant | 0 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed0` | piqa | acc_norm,none | 0.736 |
| Joint G50 depth + q_proj quant | 0 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed0` | winogrande | acc,none | 0.594 |
| Joint G50 depth + q_proj quant | 1 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed1` | arc_easy | acc_norm,none | 0.599 |
| Joint G50 depth + q_proj quant | 1 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed1` | piqa | acc_norm,none | 0.739 |
| Joint G50 depth + q_proj quant | 1 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed1` | winogrande | acc,none | 0.597 |
| Joint G50 depth + q_proj quant | 2 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed2` | arc_easy | acc_norm,none | 0.609 |
| Joint G50 depth + q_proj quant | 2 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed2` | piqa | acc_norm,none | 0.733 |
| Joint G50 depth + q_proj quant | 2 | `lmeval_interactionaware_joint_g50_mistral_s0.25_qproj3.0_tasks_seed2` | winogrande | acc,none | 0.582 |
