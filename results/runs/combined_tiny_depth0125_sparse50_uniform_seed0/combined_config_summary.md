# Combined Evaluation Config

- run_id: `combined_tiny_depth0125_sparse50_uniform_seed0`
- method: `combined_depth_sparse_eval`
- model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- sequence_length: `1024`
- eval_tokens: `4096`
- eval_datasets: `wikitext2`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `0`
- drop_layer_config: `outputs/experiments/depth_tinyllama_s0.125_seed0/layer_drop_config.txt`
- sparse_weights_path: `outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `none`
- quant_config_path: `none`
- quant_default_level: `0`
