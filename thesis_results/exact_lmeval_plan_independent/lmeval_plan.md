# Zero-shot LM-eval plan (zero-shot LM-eval of the independent compositions (replay candidates))

Jobs: 6

| Job | Precision | Removed (attn, MLP) | Repair | Mask source |
| --- | --- | --- | --- | --- |
| `ind_DO12s0_QE2s0_shift` | quantized | [4, 4] | all | results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed0/final_candidate.json |
| `ind_DO12s1_QE2s1_shift` | quantized | [4, 4] | all | results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed1/final_candidate.json |
| `ind_DO12s2_QE2s2_shift` | quantized | [4, 4] | all | results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed2/final_candidate.json |
| `ind_DO25s0_QE2s0_shift` | quantized | [8, 8] | all | results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/final_candidate.json |
| `ind_DO25s1_QE2s1_shift` | quantized | [8, 8] | all | results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed1/final_candidate.json |
| `ind_DO25s2_QE2s2_shift` | quantized | [8, 8] | all | results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2/final_candidate.json |
