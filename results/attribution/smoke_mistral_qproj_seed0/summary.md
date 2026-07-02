# Joint Compression Attribution Replay

- status: `completed`
- depth source: `independent_depth_seed0` / `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/final_candidate.json`
- quant source: `independent_quant_seed0` / `results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed0/final_candidate.json`
- model: `mistralai/Mistral-7B-v0.3`
- quant database: `outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit`

## Metrics

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | 14.4921875 |
| C4 PPL | None |
| FineWeb-Edu PPL | None |
| Compression ratio | 1.400303557622533 |
| Active parameters | 5503193088 |
| Average active bitwidth | 3.0 |

## Active Budget

- valid before repair: `True`
- repaired: `False`
- repair changes: `0`
