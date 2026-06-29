# Combined Evaluation Config

- run_id: `generalization_attention_joint_g50_mistral_s0.25_attention3.0_multidataset_seed1`
- method: `generalization_joint_g50_eval`
- model: `mistralai/Mistral-7B-v0.3`
- sequence_length: `1024`
- eval_tokens: `131072`
- eval_datasets: `wikitext2 c4 fineweb_edu`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `1`
- drop_layer_config: `results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed1/joint_drop_config.txt`
- sparse_weights_path: `none`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`
- quant_config_path: `results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed1/joint_quant_config.txt`
- quant_default_level: `3`
