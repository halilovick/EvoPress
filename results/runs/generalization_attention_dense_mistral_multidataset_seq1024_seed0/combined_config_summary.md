# Combined Evaluation Config

- run_id: `generalization_attention_dense_mistral_multidataset_seq1024_seed0`
- method: `generalization_dense_eval`
- model: `mistralai/Mistral-7B-v0.3`
- sequence_length: `1024`
- eval_tokens: `131072`
- eval_datasets: `wikitext2 c4 fineweb_edu`
- dtype: `float16`
- attention_impl: `sdpa`
- seed: `0`
- drop_layer_config: `none`
- sparse_weights_path: `none`
- sparse_config_path: `none`
- sparse_default_level: `0`
- quant_weights_path: `none`
- quant_config_path: `none`
- quant_default_level: `0`
