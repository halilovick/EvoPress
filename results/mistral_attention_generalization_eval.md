# Mistral Generalization Evaluation

This summary evaluates the same Mistral-7B compression configurations on multiple held-out datasets. The purpose is to check whether the joint-search result is only optimized for WikiText2 or whether it also transfers to C4 and FineWeb-Edu.

## Aggregate Perplexity

| Method | Dataset | Runs | Mean PPL | Std | Min | Max | Delta vs dense |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | wikitext2 | 1 | 5.960 |  | 5.960 | 5.960 | 0.000 |
| Depth-only | wikitext2 | 3 | 11.817 | 0.625 | 11.200 | 12.450 | 5.857 |
| Independent depth + attention quant | wikitext2 | 3 | 13.107 | 0.884 | 12.180 | 13.940 | 7.147 |
| Joint G50 depth + attention quant | wikitext2 | 3 | 12.670 | 0.814 | 11.740 | 13.250 | 6.710 |
| Dense FP16 | c4 | 1 | 8.860 |  | 8.860 | 8.860 | 0.000 |
| Depth-only | c4 | 3 | 14.760 | 0.805 | 13.970 | 15.580 | 5.900 |
| Independent depth + attention quant | c4 | 3 | 16.090 | 1.060 | 15.020 | 17.140 | 7.230 |
| Joint G50 depth + attention quant | c4 | 3 | 15.593 | 0.957 | 14.810 | 16.660 | 6.733 |
| Dense FP16 | fineweb_edu | 1 | 7.270 |  | 7.270 | 7.270 | 0.000 |
| Depth-only | fineweb_edu | 3 | 12.980 | 0.760 | 12.300 | 13.800 | 5.710 |
| Independent depth + attention quant | fineweb_edu | 3 | 14.237 | 0.945 | 13.300 | 15.190 | 6.967 |
| Joint G50 depth + attention quant | fineweb_edu | 3 | 13.783 | 0.958 | 12.970 | 14.840 | 6.513 |

## Paired Comparison

| Dataset | Seed | Depth PPL | Independent PPL | Joint G50 PPL | Joint - depth | Joint - independent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext2 | 0 | 11.800 | 13.200 | 13.250 | 1.450 | 0.050 |
| wikitext2 | 1 | 12.450 | 13.940 | 11.740 | -0.710 | -2.200 |
| wikitext2 | 2 | 11.200 | 12.180 | 13.020 | 1.820 | 0.840 |
| c4 | 0 | 14.730 | 16.110 | 16.660 | 1.930 | 0.550 |
| c4 | 1 | 15.580 | 17.140 | 14.810 | -0.770 | -2.330 |
| c4 | 2 | 13.970 | 15.020 | 15.310 | 1.340 | 0.290 |
| fineweb_edu | 0 | 12.840 | 14.220 | 14.840 | 2.000 | 0.620 |
| fineweb_edu | 1 | 13.800 | 15.190 | 12.970 | -0.830 | -2.220 |
| fineweb_edu | 2 | 12.300 | 13.300 | 13.540 | 1.240 | 0.240 |

## Interpretation Checklist

- If joint G50 is better than independent on all or most datasets, the thesis claim is stronger: joint selection improves transfer, not just WikiText2 fit.
- If joint G50 only wins on WikiText2, present it as evidence that the current objective overfits the calibration/evaluation setup and needs broader calibration data.
- If depth-only is close to or better than the combined methods, the next experiment should revisit the quantized module scope, the compression target, or the search objective.

## Source Runs

| Method | Seed | Run ID | Dataset | PPL |
| --- | ---: | --- | --- | ---: |
| Dense FP16 | 0 | `generalization_attention_dense_mistral_multidataset_seq1024_seed0` | wikitext2 | 5.960 |
| Depth-only | 0 | `generalization_attention_depth_mistral_s0.25_multidataset_seed0` | wikitext2 | 11.800 |
| Depth-only | 1 | `generalization_attention_depth_mistral_s0.25_multidataset_seed1` | wikitext2 | 12.450 |
| Depth-only | 2 | `generalization_attention_depth_mistral_s0.25_multidataset_seed2` | wikitext2 | 11.200 |
| Independent depth + attention quant | 0 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed0` | wikitext2 | 13.200 |
| Independent depth + attention quant | 1 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed1` | wikitext2 | 13.940 |
| Independent depth + attention quant | 2 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed2` | wikitext2 | 12.180 |
| Joint G50 depth + attention quant | 0 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed0` | wikitext2 | 13.250 |
| Joint G50 depth + attention quant | 1 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed1` | wikitext2 | 11.740 |
| Joint G50 depth + attention quant | 2 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed2` | wikitext2 | 13.020 |
| Dense FP16 | 0 | `generalization_attention_dense_mistral_multidataset_seq1024_seed0` | c4 | 8.860 |
| Depth-only | 0 | `generalization_attention_depth_mistral_s0.25_multidataset_seed0` | c4 | 14.730 |
| Depth-only | 1 | `generalization_attention_depth_mistral_s0.25_multidataset_seed1` | c4 | 15.580 |
| Depth-only | 2 | `generalization_attention_depth_mistral_s0.25_multidataset_seed2` | c4 | 13.970 |
| Independent depth + attention quant | 0 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed0` | c4 | 16.110 |
| Independent depth + attention quant | 1 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed1` | c4 | 17.140 |
| Independent depth + attention quant | 2 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed2` | c4 | 15.020 |
| Joint G50 depth + attention quant | 0 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed0` | c4 | 16.660 |
| Joint G50 depth + attention quant | 1 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed1` | c4 | 14.810 |
| Joint G50 depth + attention quant | 2 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed2` | c4 | 15.310 |
| Dense FP16 | 0 | `generalization_attention_dense_mistral_multidataset_seq1024_seed0` | fineweb_edu | 7.270 |
| Depth-only | 0 | `generalization_attention_depth_mistral_s0.25_multidataset_seed0` | fineweb_edu | 12.840 |
| Depth-only | 1 | `generalization_attention_depth_mistral_s0.25_multidataset_seed1` | fineweb_edu | 13.800 |
| Depth-only | 2 | `generalization_attention_depth_mistral_s0.25_multidataset_seed2` | fineweb_edu | 12.300 |
| Independent depth + attention quant | 0 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed0` | fineweb_edu | 14.220 |
| Independent depth + attention quant | 1 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed1` | fineweb_edu | 15.190 |
| Independent depth + attention quant | 2 | `generalization_attention_independent_depth_quant_mistral_s0.25_attention3.0_multidataset_seed2` | fineweb_edu | 13.300 |
| Joint G50 depth + attention quant | 0 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed0` | fineweb_edu | 14.840 |
| Joint G50 depth + attention quant | 1 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed1` | fineweb_edu | 12.970 |
| Joint G50 depth + attention quant | 2 | `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed2` | fineweb_edu | 13.540 |
