# runtime/setup/patch/dit/model.py
from __future__ import annotations
from collections import OrderedDict
import tensorrt as trt

from tensorrt_llm._common import default_net
from ...functional import Tensor
from ...module import Module, ModuleList
from ..modeling_utils import PretrainedConfig, PretrainedModel
from .modules import DiTBlock

class DiT(PretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dim = config.hidden_size
        self.depth = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.dim_head = config.dim_head

        self.transformer_blocks = ModuleList([
            DiTBlock(dim=self.dim, heads=self.heads, dim_head=self.dim_head, ff_mult=2, dropout=0.1)
            for _ in range(self.depth)
        ])

    def forward(self, x, t, rope_cos, rope_sin, input_lengths, scale=1.0):
        h = x
        for blk in self.transformer_blocks:
            h = blk(h, t, rope_cos=rope_cos, rope_sin=rope_sin, input_lengths=input_lengths, scale=scale)
        h.mark_output("hidden", x.dtype)
        return h

    def prepare_inputs(self, **kwargs):
        max_batch_size = kwargs.get("max_batch_size", 2)
        bs_range = [2, 2, max_batch_size]
        max_seq_len = kwargs.get("max_seq_len", 4096)
        head_dim = self.dim_head
        hidden = self.dim

        if default_net().plugin_config.remove_input_padding:
            num_frames_range = [[200, max_seq_len, max_seq_len * max_batch_size]]
            x = Tensor("x", dtype=self.config.dtype, shape=[-1, hidden],
                       dim_range=OrderedDict([("num_frames", num_frames_range), ("hidden", [[hidden]])]))
            t = Tensor("t", dtype=self.config.dtype, shape=[-1, hidden],
                       dim_range=OrderedDict([("num_frames", num_frames_range), ("hidden", [[hidden]])]))
            rope_cos = Tensor("rope_cos", dtype=self.config.dtype, shape=[-1, head_dim],
                              dim_range=OrderedDict([("num_frames", num_frames_range), ("head_dim", [[head_dim]])]))
            rope_sin = Tensor("rope_sin", dtype=self.config.dtype, shape=[-1, head_dim],
                              dim_range=OrderedDict([("num_frames", num_frames_range), ("head_dim", [[head_dim]])]))
        else:
            t_range = [[100, max_seq_len // 2, max_seq_len]]
            x = Tensor("x", dtype=self.config.dtype, shape=[-1, -1, hidden],
                       dim_range=OrderedDict([("batch", [bs_range]), ("time", t_range), ("hidden", [[hidden]])]))
            t = Tensor("t", dtype=self.config.dtype, shape=[-1, hidden],
                       dim_range=OrderedDict([("batch", [bs_range]), ("hidden", [[hidden]])]))
            rope_cos = Tensor("rope_cos", dtype=self.config.dtype, shape=[-1, -1, head_dim],
                              dim_range=OrderedDict([("batch", [bs_range]), ("time", t_range), ("head_dim", [[head_dim]])]))
            rope_sin = Tensor("rope_sin", dtype=self.config.dtype, shape=[-1, -1, head_dim],
                              dim_range=OrderedDict([("batch", [bs_range]), ("time", t_range), ("head_dim", [[head_dim]])]))
        input_lengths = Tensor("input_lengths", dtype=trt.int32, shape=[-1],
                               dim_range=OrderedDict([("batch", [[2, 2, max_batch_size]])]))
        return {"x": x, "t": t, "rope_cos": rope_cos, "rope_sin": rope_sin, "input_lengths": input_lengths}
