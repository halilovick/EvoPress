# Joint Compression Attribution Replay

- status: `completed`
- depth source: `standard_joint` / `results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed0/final_candidate.json`
- quant source: `independent` / `results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed0/final_candidate.json`
- model: `mistralai/Mistral-7B-v0.3`
- quant database: `outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit`

## Metrics

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | 11.46875 |
| C4 PPL | 14.3828125 |
| FineWeb-Edu PPL | 12.3984375 |
| Compression ratio | 1.400303557622533 |
| Active parameters | 5503193088 |
| Average active bitwidth | 3.0 |

## Active Budget

- valid before repair: `True`
- repaired: `False`
- repair changes: `0`
