# Combined Evaluation Config

- run_id: `generalization_attention_depth_mistral_s0.25_multidataset_seed2`
- method: `generalization_depth_eval`
- model: `mistralai/Mistral-7B-v0.3`
- sequence_length: `1024`
- eval_tokens: `131072`
- eval_datasets: `wikitext2 c4 fineweb_edu`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `2`
- drop_layer_config: `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2/layer_drop_config.txt`
- sparse_weights_path: `none`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `none`
- quant_config_path: `none`
- quant_default_level: `0`
