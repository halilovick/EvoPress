# Joint Compression Attribution Replay

- status: `completed`
- depth source: `independent` / `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2/final_candidate.json`
- quant source: `independent` / `results/runs/thesis_attention_quant_mistral_attention3.0_g20_o16_seed2/final_candidate.json`
- model: `mistralai/Mistral-7B-v0.3`
- quant database: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`

## Metrics

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | 12.3515625 |
| C4 PPL | 15.1015625 |
| FineWeb-Edu PPL | 13.6171875 |
| Compression ratio | 1.5469698122081734 |
| Active parameters | 5503193088 |
| Average active bitwidth | 3.0 |

## Active Budget

- valid before repair: `False`
- repaired: `True`
- repair changes: `3`
