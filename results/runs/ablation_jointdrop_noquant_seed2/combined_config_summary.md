# Combined Evaluation Config

- run_id: `ablation_jointdrop_noquant_seed2`
- method: `depth_jointmask_eval`
- model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- sequence_length: `1024`
- eval_tokens: `4096`
- eval_datasets: `wikitext2`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `2`
- drop_layer_config: `outputs/experiments/joint_tiny_depth0125_quant3_g10_seed2/joint_drop_config.txt`
- sparse_weights_path: `none`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `none`
- quant_config_path: `none`
- quant_default_level: `0`
