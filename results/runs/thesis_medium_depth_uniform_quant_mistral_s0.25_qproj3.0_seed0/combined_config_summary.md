# Combined Evaluation Config

- run_id: `thesis_medium_depth_uniform_quant_mistral_s0.25_qproj3.0_seed0`
- method: `depth_uniform_quant_eval`
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
- quant_weights_path: `outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit`
- quant_config_path: `none`
- quant_default_level: `3`
