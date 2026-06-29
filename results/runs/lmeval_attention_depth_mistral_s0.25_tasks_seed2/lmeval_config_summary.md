# Mistral LM-Eval Config

- run_id: `lmeval_attention_depth_mistral_s0.25_tasks_seed2`
- method: `lmeval_depth`
- model: `mistralai/Mistral-7B-v0.3`
- model_args: `pretrained=mistralai/Mistral-7B-v0.3,low_cpu_mem_usage=True,dtype=float16`
- tasks: `arc_easy,piqa,winogrande`
- batch_size: `4`
- max_batch_size: `none`
- limit: `none`
- num_fewshot: `0`
- device: `cuda:0`
- seed: `2`
- drop_layer_config: `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2/layer_drop_config.txt`
- quant_weights_path: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`
- quant_config_path: `none`
