# Mistral LM-Eval Config

- run_id: `lmeval_attention_joint_g50_mistral_s0.25_attention3.0_tasks_seed0`
- method: `lmeval_joint_g50`
- model: `mistralai/Mistral-7B-v0.3`
- model_args: `pretrained=mistralai/Mistral-7B-v0.3,low_cpu_mem_usage=True,dtype=float16`
- tasks: `arc_easy,piqa,winogrande`
- batch_size: `4`
- max_batch_size: `none`
- limit: `none`
- num_fewshot: `0`
- device: `cuda:0`
- seed: `0`
- drop_layer_config: `results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed0/joint_drop_config.txt`
- quant_weights_path: `outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit`
- quant_config_path: `results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed0/joint_quant_config.txt`
