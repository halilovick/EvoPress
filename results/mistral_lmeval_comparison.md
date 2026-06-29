# Mistral Downstream LM-Eval Comparison

This summary evaluates whether the Mistral joint compression result transfers from perplexity to downstream multiple-choice tasks. Higher scores are better.

## Macro Average

| Method | Macro score |
| --- | ---: |
| Dense FP16 | 0.787 |
| Depth-only | 0.647 |
| Independent depth + q_proj quant | 0.647 |
| Joint G50 depth + q_proj quant | 0.642 |

## Task Averages

| Method | Task | Metric | Runs | Mean | Std | Min | Max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | arc_easy | acc_norm,none | 1 | 0.801 |  | 0.801 | 0.801 |
| Dense FP16 | piqa | acc_norm,none | 1 | 0.819 |  | 0.819 | 0.819 |
| Dense FP16 | winogrande | acc,none | 1 | 0.742 |  | 0.742 | 0.742 |
| Depth-only | arc_easy | acc_norm,none | 3 | 0.588 | 0.022 | 0.564 | 0.607 |
| Depth-only | piqa | acc_norm,none | 3 | 0.737 | 0.007 | 0.731 | 0.744 |
| Depth-only | winogrande | acc,none | 3 | 0.617 | 0.007 | 0.612 | 0.625 |
| Independent depth + q_proj quant | arc_easy | acc_norm,none | 3 | 0.583 | 0.025 | 0.555 | 0.602 |
| Independent depth + q_proj quant | piqa | acc_norm,none | 3 | 0.739 | 0.007 | 0.733 | 0.746 |
| Independent depth + q_proj quant | winogrande | acc,none | 3 | 0.620 | 0.011 | 0.609 | 0.631 |
| Joint G50 depth + q_proj quant | arc_easy | acc_norm,none | 3 | 0.580 | 0.045 | 0.537 | 0.627 |
| Joint G50 depth + q_proj quant | piqa | acc_norm,none | 3 | 0.739 | 0.004 | 0.736 | 0.743 |
| Joint G50 depth + q_proj quant | winogrande | acc,none | 3 | 0.608 | 0.002 | 0.606 | 0.611 |

## Paired Deltas

| Task | Seed | Metric | Depth | Independent | Joint G50 | Joint - depth | Joint - independent |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| arc_easy | 0 | acc_norm,none | 0.607 | 0.602 | 0.627 | 0.020 | 0.025 |
| arc_easy | 1 | acc_norm,none | 0.564 | 0.555 | 0.537 | -0.027 | -0.019 |
| arc_easy | 2 | acc_norm,none | 0.593 | 0.591 | 0.575 | -0.017 | -0.016 |
| piqa | 0 | acc_norm,none | 0.731 | 0.733 | 0.743 | 0.011 | 0.009 |
| piqa | 1 | acc_norm,none | 0.737 | 0.737 | 0.739 | 0.003 | 0.003 |
| piqa | 2 | acc_norm,none | 0.744 | 0.746 | 0.736 | -0.009 | -0.011 |
| winogrande | 0 | acc,none | 0.625 | 0.631 | 0.608 | -0.017 | -0.023 |
| winogrande | 1 | acc,none | 0.615 | 0.620 | 0.606 | -0.009 | -0.014 |
| winogrande | 2 | acc,none | 0.612 | 0.609 | 0.611 | -0.001 | 0.002 |

## Interpretation Checklist

- If joint G50 improves macro score over independent composition, the joint-search advantage transfers to task accuracy.
- If perplexity improves but task scores do not, report the result as a limitation and keep downstream alignment as future work.
- Limited LM-eval runs are smoke tests only. Final reported task metrics should not use `--limit`.

## Source Runs

| Method | Seed | Run ID | Task | Metric | Score |
| --- | ---: | --- | --- | --- | ---: |
| Dense FP16 | 0 | `lmeval_dense_mistral_tasks_seed0_retry4` | arc_easy | acc_norm,none | 0.801 |
| Dense FP16 | 0 | `lmeval_dense_mistral_tasks_seed0_retry4` | piqa | acc_norm,none | 0.819 |
| Dense FP16 | 0 | `lmeval_dense_mistral_tasks_seed0_retry4` | winogrande | acc,none | 0.742 |
| Depth-only | 0 | `lmeval_depth_mistral_s0.25_tasks_seed0` | arc_easy | acc_norm,none | 0.607 |
| Depth-only | 0 | `lmeval_depth_mistral_s0.25_tasks_seed0` | piqa | acc_norm,none | 0.731 |
| Depth-only | 0 | `lmeval_depth_mistral_s0.25_tasks_seed0` | winogrande | acc,none | 0.625 |
| Depth-only | 1 | `lmeval_depth_mistral_s0.25_tasks_seed1` | arc_easy | acc_norm,none | 0.564 |
| Depth-only | 1 | `lmeval_depth_mistral_s0.25_tasks_seed1` | piqa | acc_norm,none | 0.737 |
| Depth-only | 1 | `lmeval_depth_mistral_s0.25_tasks_seed1` | winogrande | acc,none | 0.615 |
| Depth-only | 2 | `lmeval_depth_mistral_s0.25_tasks_seed2` | arc_easy | acc_norm,none | 0.593 |
| Depth-only | 2 | `lmeval_depth_mistral_s0.25_tasks_seed2` | piqa | acc_norm,none | 0.744 |
| Depth-only | 2 | `lmeval_depth_mistral_s0.25_tasks_seed2` | winogrande | acc,none | 0.612 |
| Independent depth + q_proj quant | 0 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed0` | arc_easy | acc_norm,none | 0.602 |
| Independent depth + q_proj quant | 0 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed0` | piqa | acc_norm,none | 0.733 |
| Independent depth + q_proj quant | 0 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed0` | winogrande | acc,none | 0.631 |
| Independent depth + q_proj quant | 1 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed1` | arc_easy | acc_norm,none | 0.555 |
| Independent depth + q_proj quant | 1 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed1` | piqa | acc_norm,none | 0.737 |
| Independent depth + q_proj quant | 1 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed1` | winogrande | acc,none | 0.620 |
| Independent depth + q_proj quant | 2 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed2` | arc_easy | acc_norm,none | 0.591 |
| Independent depth + q_proj quant | 2 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed2` | piqa | acc_norm,none | 0.746 |
| Independent depth + q_proj quant | 2 | `lmeval_independent_depth_quant_mistral_s0.25_qproj3.0_tasks_seed2` | winogrande | acc,none | 0.609 |
| Joint G50 depth + q_proj quant | 0 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed0_retry4` | arc_easy | acc_norm,none | 0.627 |
| Joint G50 depth + q_proj quant | 0 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed0_retry4` | piqa | acc_norm,none | 0.743 |
| Joint G50 depth + q_proj quant | 0 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed0_retry4` | winogrande | acc,none | 0.608 |
| Joint G50 depth + q_proj quant | 1 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed1` | arc_easy | acc_norm,none | 0.537 |
| Joint G50 depth + q_proj quant | 1 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed1` | piqa | acc_norm,none | 0.739 |
| Joint G50 depth + q_proj quant | 1 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed1` | winogrande | acc,none | 0.606 |
| Joint G50 depth + q_proj quant | 2 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed2` | arc_easy | acc_norm,none | 0.575 |
| Joint G50 depth + q_proj quant | 2 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed2` | piqa | acc_norm,none | 0.736 |
| Joint G50 depth + q_proj quant | 2 | `lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed2` | winogrande | acc,none | 0.611 |
