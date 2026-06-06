# Combined Evaluation Config

- run_id: `combined_tiny_depth025_sparse_searchcfg_seed0`
- method: `combined_depth_sparse_eval`
- model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- sequence_length: `1024`
- eval_tokens: `4096`
- eval_datasets: `wikitext2`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `0`
- drop_layer_config: `outputs/experiments/depth_tinyllama_s0.25_seed0/layer_drop_config.txt`
- sparse_weights_path: `outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db`
- sparse_config_path: `results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/sparse_configuration.txt`
- sparse_default_level: `0`
- quant_weights_path: `none`
- quant_config_path: `none`
- quant_default_level: `0`
