# Combined Evaluation Config

- run_id: `thesis_attention_independent_depth_quant_mistral_s0.25_attention3.0_seed0`
- method: `independent_depth_quant_eval`
- model: `mistralai/Mistral-7B-v0.3`
- sequence_length: `1024`
- eval_tokens: `524288`
- eval_datasets: `wikitext2`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `0`
- drop_layer_config: `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/layer_drop_config.txt`
- sparse_weights_path: `none`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`
- quant_config_path: `results/runs/thesis_attention_quant_mistral_attention3.0_g20_o16_seed0/quant_configuration.txt`
- quant_default_level: `3`
