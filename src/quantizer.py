from typing import Iterable, Dict, List, Any, Optional, Union


import os
import torch
import torch.nn as nn
import torch.distributed as dist


from src import dist_utils
from src.activation_cache import DiskActivationCache
from src.common_utils import to, maybe_first_element
from src.io_utils import torch_save
from src.memory_utils import release_cpu_memory
from src.model_utils import InputCollector, ForwardInterrupt, LINEAR_LAYERS, select_layers
from src.quant_utils import QLinear

from src.fast_obq import FastOBQ


class Quantizer:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        quantizable_modules: str,
        pre_block_modules: List[str],
        save_dir: Union[str, os.PathLike],
        block_modules: str,
        obq_kwargs: Dict[str, Any] = {},
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        activation_cache_dir: Optional[Union[str, os.PathLike]] = None,
        drop_saved_file_cache: bool = False,
        verbose: bool = False,
        attempt_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.quantizable_modules = quantizable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.save_dir = save_dir
        self.obq_kwargs = obq_kwargs
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.activation_cache_dir = activation_cache_dir
        self.drop_saved_file_cache = drop_saved_file_cache
        self.verbose = verbose
        self.attempt_id = attempt_id
        self.activation_cache_summary = None
        self.saved_reconstruction_metadata = {}

    @torch.no_grad()
    def quantize(self, bitwidth_options: List[int], calibration_bitwidth: int):
        device = self.device or next(self.model.parameters()).device
        activation_cache = None
        if self.activation_cache_dir is not None:
            activation_cache = DiskActivationCache(
                self.activation_cache_dir,
                drop_file_cache=self.drop_saved_file_cache,
                attempt_id=self.attempt_id,
            )
        # prepare pre blocks modules
        blocks = self._get_submodule(self.block_modules)
        pre_blocks = [self._get_submodule(module_name) for module_name in self.pre_block_modules]
        blocks[0] = blocks[0].to(device)
        for module in pre_blocks:
            module.to(device)
        # Cache
        if hasattr(self.model.config, "use_cache"):
            use_cache = self.model.config.use_cache
            self.model.config.use_cache = False
        # Input preparation #
        # collect inputs to block 0
        # this captures the hidden states that enter the firs transformer block, then interrupts
        blocks[0] = InputCollector(
            blocks[0],
            cpu_offload=self.cpu_offload_activations,
            input_cache=activation_cache,
        )
        # TODO make namedtuple
        for sample_index, (inp_args, inp_kwargs) in enumerate(self.data_loader):
            try:
                self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            except ForwardInterrupt:
                pass
            if (
                self.verbose
                and activation_cache is not None
                and (sample_index + 1) % 100 == 0
            ):
                dist_utils.print_on_main(
                    f"Cached {sample_index + 1} block-input samples to disk.",
                    flush=True,
                )
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module
        if activation_cache is not None:
            activation_cache.finish_collection()
            dist_utils.print_on_main(
                f"Disk activation cache contains {len(activation_cache)} samples at "
                f"{activation_cache.root}.",
                flush=True,
            )
        self.data_loader = None
        release_cpu_memory()

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()

        # offload pre_blocks
        if self.cpu_offload_modules:
            for module in pre_blocks:
                module.cpu()

        # Block pruning #
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            # this selects 0.self_attn.q_proj, 0.self_attn.k_proj, 0.self_attn.v_proj, 0.self_attn.o_proj, 0.mlp.up_proj, 0.mlp.down_proj, 0.mlp_gate_proj
            layers = select_layers(self.model, layer_prefix, self.quantizable_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(bitwidth_options, layers)

            for _, inp_args, inp_kwargs in self._input_items(
                activation_cache, input_args, input_kwargs
            ):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                del out

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            self._quant_group(handles, bitwidth_options, calibration_bitwidth)

            for input_index, inp_args, inp_kwargs in self._input_items(
                activation_cache, input_args, input_kwargs
            ):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))  # me
                out = maybe_first_element(out)
                if self.cpu_offload_activations or activation_cache is not None:
                    out = out.cpu()
                if activation_cache is not None:
                    inp_args, inp_kwargs = self._replace_hidden_state(
                        inp_args, inp_kwargs, out
                    )
                    activation_cache.replace(input_index, inp_args, inp_kwargs)
                else:
                    # change only first input argument
                    if len(inp_args) > 0:
                        inp_args[0].data = out
                    elif "hidden_states" in inp_kwargs:
                        inp_kwargs["hidden_states"] = out
                    else:
                        raise ValueError("Unsupported block input format.")
                del out

            if activation_cache is not None:
                activation_cache.mark_block_complete(block_id + 1, len(blocks))

            if self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache
        if activation_cache is not None:
            self.activation_cache_summary = activation_cache.summary()

    @staticmethod
    def _input_items(activation_cache, input_args, input_kwargs):
        if activation_cache is not None:
            for index in range(len(activation_cache)):
                inp_args, inp_kwargs = activation_cache.load(index)
                yield index, inp_args, inp_kwargs
            return
        for index, (inp_args, inp_kwargs) in enumerate(
            zip(input_args, input_kwargs)
        ):
            yield index, inp_args, inp_kwargs

    @staticmethod
    def _replace_hidden_state(inp_args, inp_kwargs, hidden_states):
        if len(inp_args) > 0:
            if not isinstance(inp_args, (tuple, list)):
                raise TypeError(
                    "Disk-backed block input arguments must be a tuple or list."
                )
            # Mirror the legacy in-memory path. Mutating the loaded tensor also
            # preserves any within-record aliases to the hidden-state object.
            inp_args[0].data = hidden_states
        elif "hidden_states" in inp_kwargs:
            inp_kwargs = dict(inp_kwargs)
            inp_kwargs["hidden_states"] = hidden_states
        else:
            raise ValueError("Unsupported block input format.")
        return inp_args, inp_kwargs

    def _get_submodule(self, module_name: str):
        return self.model.get_submodule(module_name)

    def _prepare_hooks_and_handles(self, bitwidth_options: List[int], layers: Dict[str, nn.Module]):
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0])

                return _hook

            handles[layer_name] = self._create_handle(bitwidth_options, layer)
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        return handles, hooks

    def _create_handle(self, bitwidth_options, layer):
        return FastOBQ(layer, bitwidth_options=bitwidth_options, **self.obq_kwargs)

    def _quant_group(self, handles: Dict[str, FastOBQ], bitwidth_options: List[int], calibration_bitwidth: int):
        for handle_name, handle in handles.items():

            if self.verbose:
                dist_utils.print_on_main(f"Quantizing {handle_name}")
            qweight_dict, scale_dict, zero_dict, perm = handle.quantize(bitwidth_options)

            for bits in bitwidth_options:
                qlayer = QLinear(
                    qweight_dict[bits],
                    scale_dict[bits],
                    zero_dict[bits],
                    bias=handle.layer.bias,
                    perm=perm,
                    bits=8 if bits > 4 else 4,
                )
                if dist_utils.is_main():
                    dequantized_weight = qlayer.get_weight().cpu()
                    module_dir = os.path.join(self.save_dir, handle_name)
                    os.makedirs(module_dir, exist_ok=True)
                    weight_path = os.path.join(module_dir, f"{int(bits)}.pth")
                    file_sha256 = torch_save(
                        dequantized_weight,
                        weight_path,
                        drop_file_cache=self.drop_saved_file_cache,
                        compute_sha256=True,
                    )
                    self.saved_reconstruction_metadata.setdefault(
                        handle_name, {}
                    )[str(int(bits))] = {
                        "shape": list(dequantized_weight.shape),
                        "dtype": str(dequantized_weight.dtype).removeprefix("torch."),
                        "numel": int(dequantized_weight.numel()),
                        "contiguous": bool(dequantized_weight.is_contiguous()),
                        "file_size_bytes": os.path.getsize(weight_path),
                        "file_sha256": file_sha256,
                    }
                    del dequantized_weight
                # Replace original layer by quantized layer with given bitwidth
                if bits == calibration_bitwidth:
                    parent_name, child_name = handle_name.rsplit(".", 1)
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, qlayer)
            handle.reset()
