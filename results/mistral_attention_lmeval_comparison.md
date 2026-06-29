# Mistral Downstream LM-Eval Comparison

This summary evaluates whether the Mistral joint compression result transfers from perplexity to downstream multiple-choice tasks. Higher scores are better.

## Macro Average

| Method | Macro score |
| --- | ---: |
| Dense FP16 | 0.788 |
| Depth-only | 0.648 |
| Independent depth + attention quant | 0.596 |
| Joint G50 depth + attention quant | 0.601 |

## Task Averages

| Method | Task | Metric | Runs | Mean | Std | Min | Max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | arc_easy | acc_norm,none | 1 | 0.801 |  | 0.801 | 0.801 |
| Dense FP16 | piqa | acc_norm,none | 1 | 0.820 |  | 0.820 | 0.820 |
| Dense FP16 | winogrande | acc,none | 1 | 0.742 |  | 0.742 | 0.742 |
| Depth-only | arc_easy | acc_norm,none | 3 | 0.588 | 0.022 | 0.564 | 0.608 |
| Depth-only | piqa | acc_norm,none | 3 | 0.737 | 0.008 | 0.730 | 0.745 |
| Depth-only | winogrande | acc,none | 3 | 0.619 | 0.009 | 0.612 | 0.628 |
| Independent depth + attention quant | arc_easy | acc_norm,none | 3 | 0.535 | 0.043 | 0.509 | 0.584 |
| Independent depth + attention quant | piqa | acc_norm,none | 3 | 0.696 | 0.017 | 0.683 | 0.715 |
| Independent depth + attention quant | winogrande | acc,none | 3 | 0.558 | 0.059 | 0.522 | 0.626 |
| Joint G50 depth + attention quant | arc_easy | acc_norm,none | 3 | 0.534 | 0.025 | 0.505 | 0.553 |
| Joint G50 depth + attention quant | piqa | acc_norm,none | 3 | 0.707 | 0.018 | 0.694 | 0.728 |
| Joint G50 depth + attention quant | winogrande | acc,none | 3 | 0.561 | 0.031 | 0.530 | 0.593 |

## Paired Deltas

| Task | Seed | Metric | Depth | Independent | Joint G50 | Joint - depth | Joint - independent |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| arc_easy | 0 | acc_norm,none | 0.608 | 0.584 | 0.505 | -0.102 | -0.079 |
| arc_easy | 1 | acc_norm,none | 0.564 | 0.511 | 0.553 | -0.011 | 0.043 |
| arc_easy | 2 | acc_norm,none | 0.593 | 0.509 | 0.543 | -0.050 | 0.034 |
| piqa | 0 | acc_norm,none | 0.730 | 0.715 | 0.694 | -0.036 | -0.022 |
| piqa | 1 | acc_norm,none | 0.737 | 0.683 | 0.700 | -0.038 | 0.017 |
| piqa | 2 | acc_norm,none | 0.745 | 0.690 | 0.728 | -0.017 | 0.038 |
| winogrande | 0 | acc,none | 0.628 | 0.626 | 0.560 | -0.069 | -0.066 |
| winogrande | 1 | acc,none | 0.616 | 0.522 | 0.530 | -0.086 | 0.008 |
| winogrande | 2 | acc,none | 0.612 | 0.526 | 0.593 | -0.019 | 0.066 |

## Interpretation Checklist

- If joint G50 improves macro score over independent composition, the joint-search advantage transfers to task accuracy.
- If perplexity improves but task scores do not, report the result as a limitation and keep downstream alignment as future work.
- Limited LM-eval runs are smoke tests only. Final reported task metrics should not use `--limit`.

## Source Runs

| Method | Seed | Run ID | Task | Metric | Score |
| --- | ---: | --- | --- | --- | ---: |
| Dense FP16 | 0 | `lmeval_attention_dense_mistral_tasks_seed0` | arc_easy | acc_norm,none | 0.801 |
| Dense FP16 | 0 | `lmeval_attention_dense_mistral_tasks_seed0` | piqa | acc_norm,none | 0.820 |
| Dense FP16 | 0 | `lmeval_attention_dense_mistral_tasks_seed0` | winogrande | acc,none | 0.742 |
| Depth-only | 0 | `lmeval_attention_depth_mistral_s0.25_tasks_seed0` | arc_easy | acc_norm,none | 0.608 |
| Depth-only | 0 | `lmeval_attention_depth_mistral_s0.25_tasks_seed0` | piqa | acc_norm,none | 0.730 |
| Depth-only | 0 | `lmeval_attention_depth_mistral_s0.25_tasks_seed0` | winogrande | acc,none | 0.628 |
| Depth-only | 1 | `lmeval_attention_depth_mistral_s0.25_tasks_seed1` | arc_easy | acc_norm,none | 0.564 |
| Depth-only | 1 | `lmeval_attention_depth_mistral_s0.25_tasks_seed1` | piqa | acc_norm,none | 0.737 |
| Depth-only | 1 | `lmeval_attention_depth_mistral_s0.25_tasks_seed1` | winogrande | acc,none | 0.616 |
| Depth-only | 2 | `lmeval_attention_depth_mistral_s0.25_tasks_seed2` | arc_easy | acc_norm,none | 0.593 |
| Depth-only | 2 | `lmeval_attention_depth_mistral_s0.25_tasks_seed2` | piqa | acc_norm,none | 0.745 |
| Depth-only | 2 | `lmeval_attention_depth_mistral_s0.25_tasks_seed2` | winogrande | acc,none | 0.612 |
| Independent depth + attention quant | 0 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed0` | arc_easy | acc_norm,none | 0.584 |
| Independent depth + attention quant | 0 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed0` | piqa | acc_norm,none | 0.715 |
| Independent depth + attention quant | 0 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed0` | winogrande | acc,none | 0.626 |
| Independent depth + attention quant | 1 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed1` | arc_easy | acc_norm,none | 0.511 |
| Independent depth + attention quant | 1 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed1` | piqa | acc_norm,none | 0.683 |
| Independent depth + attention quant | 1 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed1` | winogrande | acc,none | 0.522 |
| Independent depth + attention quant | 2 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed2` | arc_easy | acc_norm,none | 0.509 |
| Independent depth + attention quant | 2 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed2` | piqa | acc_norm,none | 0.690 |
| Independent depth + attention quant | 2 | `lmeval_attention_independent_depth_quant_mistral_s0.25_attention3.0_tasks_seed2` | winogrande | acc,none | 0.526 |
| Joint G50 depth + attention quant | 0 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed0` | arc_easy | acc_norm,none | 0.505 |
| Joint G50 depth + attention quant | 0 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed0` | piqa | acc_norm,none | 0.694 |
| Joint G50 depth + attention quant | 0 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed0` | winogrande | acc,none | 0.560 |
| Joint G50 depth + attention quant | 1 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed1` | arc_easy | acc_norm,none | 0.553 |
| Joint G50 depth + attention quant | 1 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed1` | piqa | acc_norm,none | 0.700 |
| Joint G50 depth + attention quant | 1 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed1` | winogrande | acc,none | 0.530 |
| Joint G50 depth + attention quant | 2 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed2` | arc_easy | acc_norm,none | 0.543 |
| Joint G50 depth + attention quant | 2 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed2` | piqa | acc_norm,none | 0.728 |
| Joint G50 depth + attention quant | 2 | `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed2` | winogrande | acc,none | 0.593 |
