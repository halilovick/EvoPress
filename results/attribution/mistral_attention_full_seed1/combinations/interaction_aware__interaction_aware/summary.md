# Joint Compression Attribution Replay

- status: `completed`
- depth source: `interaction_aware` / `results/runs/thesis_attention_interactionaware_joint_mistral_s0.25_attention3.0_g50_o16_seed1/final_candidate.json`
- quant source: `interaction_aware` / `results/runs/thesis_attention_interactionaware_joint_mistral_s0.25_attention3.0_g50_o16_seed1/final_candidate.json`
- model: `mistralai/Mistral-7B-v0.3`
- quant database: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`

## Metrics

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | 12.4453125 |
| C4 PPL | 15.28125 |
| FineWeb-Edu PPL | 13.5078125 |
| Compression ratio | 1.5469698122081734 |
| Active parameters | 5503193088 |
| Average active bitwidth | 3.0 |

## Active Budget

- valid before repair: `True`
- repaired: `False`
- repair changes: `0`
