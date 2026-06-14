# Combined Evaluation Config

- run_id: `replay_joint_mistral_debug_seed0`
- method: `replay_joint_mistral_eval`
- model: `mistralai/Mistral-7B-v0.3`
- sequence_length: `512`
- eval_tokens: `4096`
- eval_datasets: `wikitext2`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `0`
- drop_layer_config: `outputs/experiments/debug_reporting_joint_mistral_seed0_retry1/joint_drop_config.txt`
- sparse_weights_path: `none`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit`
- quant_config_path: `outputs/experiments/debug_reporting_joint_mistral_seed0_retry1/joint_quant_config.txt`
- quant_default_level: `0`
