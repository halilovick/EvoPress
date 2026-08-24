from tqdm import trange

import torch
import torch.nn.functional as F

from src.common_utils import to
from src.teacher_logits_cache import (
    evict_tensor_reference_file_cache,
    materialize_tensor_reference,
)
from src.memory_utils import release_cpu_memory
from src.model_utils import Catcher, CatcherExit, get_layers, get_lm_head, get_lm_logits


@torch.no_grad()
def compute_perplexity(model, data, batch_size: int = 1):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    nll_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in trange(0, num_samples, batch_size, desc="Computing perplexity", leave=False):
        j = min(i + batch_size, num_samples)

        # batch_size = 1, j = 1, i = 0 -> data[0:1] -> [data[0]] -> data[0]
        # data =[[10, 20, 30, 40], [50, 60, 70, 80]] -> inputs = [[10, 20, 30, 40]] 
        inputs = torch.cat(data[i:j]).to(device) 
        
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        # Update number of processed tokens
        tokens_processed += shift_labels.numel()
    # Compute perplexity
    ppl = nll_running.exp().item()
    return ppl



@torch.no_grad()
def compute_kl_div(model, data, target_logits, batch_size: int = 1):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    kl_div_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in trange(0, num_samples, batch_size, desc="Computing KL Divergence", leave=False):
        torch.cuda.empty_cache()
        j = min(i + batch_size, num_samples)
       
        inputs = torch.cat(data[i:j]).to(device)
        target_refs = target_logits[i:j]
        target_items = [
            materialize_tensor_reference(item)
            for item in target_refs
        ]
        target_cpu = (
            target_items[0]
            if len(target_items) == 1
            else torch.cat(target_items)
        )
        targets = target_cpu.to(device)
        del target_items, target_cpu
        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Don't predict last token (not required, can be removed)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, :-1, :]
        
      
        #Squeeze on GPU 
        torch.cuda.empty_cache()
        for i in range(0, shift_logits.shape[1], 1024):
            j = min(i + 1024, shift_logits.shape[1])
            shift_logits_batch = shift_logits[:, i:j, :]
            shift_targets_batch = shift_targets[:, i:j, :]
            loss_batch = F.kl_div(
                shift_logits_batch.reshape(-1, shift_logits_batch.size(-1)).log_softmax(dim=-1),
                shift_targets_batch.reshape(-1, shift_targets_batch.size(-1)).log_softmax(dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            # Calculate negative log likelihood
            a = shift_targets_batch.numel() / (tokens_processed + shift_targets_batch.numel())
            b = tokens_processed / (tokens_processed + shift_targets_batch.numel())
            kl_div_running = a * loss_batch + b * kl_div_running
            # Update number of processed tokens
            tokens_processed += shift_targets_batch.numel()
            del shift_logits_batch, shift_targets_batch, loss_batch
            torch.cuda.empty_cache()      

        # All computations using this teacher batch are complete. Release any
        # mmap-backed CPU tensor/storage, then tell Linux those file pages are
        # not useful for page cache. This keeps the 32 GiB teacher cache from
        # competing with the 16 GiB cgroup memory limit.
        del inputs, targets, lm_logits, shift_logits, shift_targets
        release_cpu_memory()
        for target_ref in target_refs:
            evict_tensor_reference_file_cache(target_ref)
        
 
    return kl_div_running.item()


@torch.no_grad()
def compute_sparse_kl_div(model, data, target_logits):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    kl_div_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in trange(0, num_samples, desc="Computing Sparse KL Divergence", leave=False):
        inputs = to(data[i], device=device)
        targets, target_ids = to(target_logits[i], device=device)
        # Forward pass through the model
        lm_logits = model(inputs).logits.gather(dim=-1, index=target_ids)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, :-1, :]
        # Compute loss
        loss = F.kl_div(
            shift_logits.reshape(-1, shift_logits.size(-1)).log_softmax(dim=-1),
            shift_targets.reshape(-1, shift_targets.size(-1)).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        # Calculate negative log likelihood
        a = shift_targets.numel() / (tokens_processed + shift_targets.numel())
        b = tokens_processed / (tokens_processed + shift_targets.numel())
        kl_div_running = a * loss + b * kl_div_running
        # Update number of processed tokens
        tokens_processed += shift_targets.numel()
    return kl_div_running.item()


@torch.no_grad()
def compute_perplexity_layer_per_layer(
    model, data, device: torch.device = "cuda", offload: bool = False, batch_size: int = 1
):
    num_samples = len(data)

    # Get layers
    input_embeddings = model.get_input_embeddings()
    layers = get_layers(model)
    lm_head = get_lm_head(model)

    # Process input embeddings
    input_embeddings = input_embeddings.to(device)
    layers[0] = layers[0].to(device)
    layers[0] = Catcher(layers[0], offload=offload)
    for i in range(0, num_samples, batch_size):
        try:
            j = min(i + batch_size, num_samples)
            input_ids = torch.cat(data[i:j], dim=0).to(device)
            # call model.forward to trigger the Catcher
            model(input_ids, attention_mask=torch.ones_like(input_ids))
        except CatcherExit:
            pass
    inputs = layers[0].inputs
    input_kwargs = layers[0].input_kwargs
    layers[0] = layers[0].module
    input_embeddings = input_embeddings.cpu()

    # Process layers
    for layer_id in trange(len(layers), desc="Processing evaluation data layer-by-layer", leave=False):
        layer = layers[layer_id].to(device)
        for i, (inps, inp_kwargs) in enumerate(zip(inputs, input_kwargs)):
            out = layer(inps.to(device), **inp_kwargs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if offload:
                out = out.cpu()
            inputs[i] = out
        # Offload layer
        layers[layer_id] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    # Compute perplexity
    lm_head = lm_head.to(device)
    # Running estimate of negative log-likelihood
    nll_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    for i, inps in zip(range(0, num_samples, batch_size), inputs):
        # Get input_ids
        j = min(i + batch_size, num_samples)
        input_ids = torch.cat(data[i:j], dim=0).to(device)
        # Forward pass through the model
        lm_logits = get_lm_logits(inps.to(device), model)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:]
        # Compute loss
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        # Update number of processed tokens
        tokens_processed += shift_labels.numel()
    # put lm_head back to original device
    lm_head = lm_head.cpu()
    # Compute perplexity
    ppl = nll_running.exp().item()
    return ppl
