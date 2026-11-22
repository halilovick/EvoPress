# Zero-shot LM-eval plan (zero-shot LM-eval of the sequential (frozen-mask) final models)

Jobs: 6

| Job | Precision | Removed (attn, MLP) | Repair | Mask source |
| --- | --- | --- | --- | --- |
| `own_SFs0` | quantized | [8, 8] | none | thesis_results/datalab_raw/fullspace_extensions/fullspace_depthfrozen_g150_seed0/final_candidate.json |
| `own_SFs1` | quantized | [8, 8] | none | thesis_results/datalab_raw/fullspace_extensions/fullspace_depthfrozen_g150_seed1/final_candidate.json |
| `own_SFs2` | quantized | [8, 8] | none | thesis_results/datalab_raw/fullspace_extensions/fullspace_depthfrozen_g150_seed2/final_candidate.json |
| `own_MSs0` | quantized | [8, 8] | none | thesis_results/datalab_raw/fullspace_extensions/fullspace_matchedseq_d50_q100/frozen_q100_seed0/final_candidate.json |
| `own_MSs1` | quantized | [8, 8] | none | thesis_results/datalab_raw/fullspace_extensions/fullspace_matchedseq_d50_q100/frozen_q100_seed1/final_candidate.json |
| `own_MSs2` | quantized | [8, 8] | none | thesis_results/datalab_raw/fullspace_extensions/fullspace_matchedseq_d50_q100/frozen_q100_seed2/final_candidate.json |
