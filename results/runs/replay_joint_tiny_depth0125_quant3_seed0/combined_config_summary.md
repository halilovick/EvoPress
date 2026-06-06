# Combined Evaluation Config

- run_id: `replay_joint_tiny_depth0125_quant3_seed0`
- method: `combined_depth_quant_eval`
- model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- sequence_length: `1024`
- eval_tokens: `4096`
- eval_datasets: `wikitext2`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `0`
- drop_layer_config: `outputs/experiments/joint_tiny_depth0125_quant3_g10_seed0/joint_drop_config.txt`
- sparse_weights_path: `none`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `outputs/experiments/quant_db_tinyllama_qproj_bits234/quant_db/TinyLlama-1.1B-Chat-v1.0/3bit`
- quant_config_path: `outputs/experiments/joint_tiny_depth0125_quant3_g10_seed0/joint_quant_config.txt`
- quant_default_level: `0`
