# Mistral Generalization Evaluation

This summary evaluates the same Mistral-7B compression configurations on multiple held-out datasets. The purpose is to check whether the joint-search result is only optimized for WikiText2 or whether it also transfers to C4 and FineWeb-Edu.

## Aggregate Perplexity

| Method | Dataset | Runs | Mean PPL | Std | Min | Max | Delta vs dense |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | wikitext2 | 1 | 5.960 |  | 5.960 | 5.960 | 0.000 |
| Depth-only | wikitext2 | 3 | 11.817 | 0.625 | 11.200 | 12.450 | 5.857 |
| Independent depth + q_proj quant | wikitext2 | 3 | 11.947 | 0.585 | 11.350 | 12.520 | 5.987 |
| Joint G50 depth + q_proj quant | wikitext2 | 3 | 11.243 | 0.287 | 10.920 | 11.470 | 5.283 |
| Dense FP16 | c4 | 1 | 8.860 |  | 8.860 | 8.860 | 0.000 |
| Depth-only | c4 | 3 | 14.760 | 0.805 | 13.970 | 15.580 | 5.900 |
| Independent depth + q_proj quant | c4 | 3 | 14.910 | 0.801 | 14.130 | 15.730 | 6.050 |
| Joint G50 depth + q_proj quant | c4 | 3 | 14.393 | 0.184 | 14.190 | 14.550 | 5.533 |
| Dense FP16 | fineweb_edu | 1 | 7.340 |  | 7.340 | 7.340 | 0.000 |
| Depth-only | fineweb_edu | 3 | 12.980 | 0.760 | 12.300 | 13.800 | 5.640 |
| Independent depth + q_proj quant | fineweb_edu | 3 | 13.140 | 0.838 | 12.400 | 14.050 | 5.800 |
| Joint G50 depth + q_proj quant | fineweb_edu | 3 | 12.460 | 0.223 | 12.260 | 12.700 | 5.120 |

## Paired Comparison

| Dataset | Seed | Depth PPL | Independent PPL | Joint G50 PPL | Joint - depth | Joint - independent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext2 | 0 | 11.800 | 11.970 | 11.470 | -0.330 | -0.500 |
| wikitext2 | 1 | 12.450 | 12.520 | 10.920 | -1.530 | -1.600 |
| wikitext2 | 2 | 11.200 | 11.350 | 11.340 | 0.140 | -0.010 |
| c4 | 0 | 14.730 | 14.870 | 14.440 | -0.290 | -0.430 |
| c4 | 1 | 15.580 | 15.730 | 14.190 | -1.390 | -1.540 |
| c4 | 2 | 13.970 | 14.130 | 14.550 | 0.580 | 0.420 |
| fineweb_edu | 0 | 12.840 | 12.970 | 12.420 | -0.420 | -0.550 |
| fineweb_edu | 1 | 13.800 | 14.050 | 12.260 | -1.540 | -1.790 |
| fineweb_edu | 2 | 12.300 | 12.400 | 12.700 | 0.400 | 0.300 |

## Interpretation Checklist

- If joint G50 is better than independent on all or most datasets, the thesis claim is stronger: joint selection improves transfer, not just WikiText2 fit.
- If joint G50 only wins on WikiText2, present it as evidence that the current objective overfits the calibration/evaluation setup and needs broader calibration data.
- If depth-only is close to or better than the combined methods, the next experiment should increase the quantized module scope or adjust the compression target, because q_proj-only quantization contributes limited compression.

## Source Runs

| Method | Seed | Run ID | Dataset | PPL |
| --- | ---: | --- | --- | ---: |
| Dense FP16 | 0 | `generalization_dense_mistral_multidataset_seq1024_seed0` | wikitext2 | 5.960 |
| Depth-only | 0 | `generalization_depth_mistral_s0.25_multidataset_seed0` | wikitext2 | 11.800 |
| Depth-only | 1 | `generalization_depth_mistral_s0.25_multidataset_seed1` | wikitext2 | 12.450 |
| Depth-only | 2 | `generalization_depth_mistral_s0.25_multidataset_seed2` | wikitext2 | 11.200 |
| Independent depth + q_proj quant | 0 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed0` | wikitext2 | 11.970 |
| Independent depth + q_proj quant | 1 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed1` | wikitext2 | 12.520 |
| Independent depth + q_proj quant | 2 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed2` | wikitext2 | 11.350 |
| Joint G50 depth + q_proj quant | 0 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed0` | wikitext2 | 11.470 |
| Joint G50 depth + q_proj quant | 1 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed1` | wikitext2 | 10.920 |
| Joint G50 depth + q_proj quant | 2 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed2` | wikitext2 | 11.340 |
| Dense FP16 | 0 | `generalization_dense_mistral_multidataset_seq1024_seed0` | c4 | 8.860 |
| Depth-only | 0 | `generalization_depth_mistral_s0.25_multidataset_seed0` | c4 | 14.730 |
| Depth-only | 1 | `generalization_depth_mistral_s0.25_multidataset_seed1` | c4 | 15.580 |
| Depth-only | 2 | `generalization_depth_mistral_s0.25_multidataset_seed2` | c4 | 13.970 |
| Independent depth + q_proj quant | 0 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed0` | c4 | 14.870 |
| Independent depth + q_proj quant | 1 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed1` | c4 | 15.730 |
| Independent depth + q_proj quant | 2 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed2` | c4 | 14.130 |
| Joint G50 depth + q_proj quant | 0 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed0` | c4 | 14.440 |
| Joint G50 depth + q_proj quant | 1 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed1` | c4 | 14.190 |
| Joint G50 depth + q_proj quant | 2 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed2` | c4 | 14.550 |
| Dense FP16 | 0 | `generalization_dense_mistral_multidataset_seq1024_seed0` | fineweb_edu | 7.340 |
| Depth-only | 0 | `generalization_depth_mistral_s0.25_multidataset_seed0` | fineweb_edu | 12.840 |
| Depth-only | 1 | `generalization_depth_mistral_s0.25_multidataset_seed1` | fineweb_edu | 13.800 |
| Depth-only | 2 | `generalization_depth_mistral_s0.25_multidataset_seed2` | fineweb_edu | 12.300 |
| Independent depth + q_proj quant | 0 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed0` | fineweb_edu | 12.970 |
| Independent depth + q_proj quant | 1 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed1` | fineweb_edu | 14.050 |
| Independent depth + q_proj quant | 2 | `generalization_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed2` | fineweb_edu | 12.400 |
| Joint G50 depth + q_proj quant | 0 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed0` | fineweb_edu | 12.420 |
| Joint G50 depth + q_proj quant | 1 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed1` | fineweb_edu | 12.260 |
| Joint G50 depth + q_proj quant | 2 | `generalization_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed2` | fineweb_edu | 12.700 |
