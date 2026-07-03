# Joint Compression Attribution Replay

- status: `completed`
- depth source: `standard_joint` / `results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed0/final_candidate.json`
- quant source: `standard_joint` / `results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed0/final_candidate.json`
- model: `mistralai/Mistral-7B-v0.3`
- quant database: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`

## Metrics

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | 13.25 |
| C4 PPL | 16.65625 |
| FineWeb-Edu PPL | 14.8359375 |
| Compression ratio | 1.5469698122081734 |
| Active parameters | 5503193088 |
| Average active bitwidth | 3.0 |

## Active Budget

- valid before repair: `True`
- repaired: `False`
- repair changes: `0`
