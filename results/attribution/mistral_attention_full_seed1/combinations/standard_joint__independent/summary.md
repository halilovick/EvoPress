# Joint Compression Attribution Replay

- status: `completed`
- depth source: `standard_joint` / `results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed1/final_candidate.json`
- quant source: `independent` / `results/runs/thesis_attention_quant_mistral_attention3.0_g20_o16_seed1/final_candidate.json`
- model: `mistralai/Mistral-7B-v0.3`
- quant database: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`

## Metrics

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | 11.8046875 |
| C4 PPL | 14.9296875 |
| FineWeb-Edu PPL | 13.09375 |
| Compression ratio | 1.5469698122081734 |
| Active parameters | 5503193088 |
| Average active bitwidth | 3.0 |

## Active Budget

- valid before repair: `False`
- repaired: `True`
- repair changes: `2`
