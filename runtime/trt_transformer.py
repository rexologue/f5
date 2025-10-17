# runtime/trt_transformer.py (ИСПРАВЛЕНО под существующие имена)
from __future__ import annotations
import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import tensorrt as trt

from src.core.dit import TimestepEmbedding, TextEmbedding, InputEmbedding
from src.core.modules import AdaLayerNorm_Final


class _DiTRuntime:
    """
    Низкоуровневая обёртка над TRT DiT engine.
    Работает с оригинальными именами входов из prepare_inputs:
      - x (вместо x_embed)
      - t (вместо t_embed)
      - rope_cos, rope_sin, input_lengths
    """
    
    def __init__(
        self,
        engine_dir: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        from pathlib import Path
        
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Ищем .engine файл
        engine_path = Path(engine_dir)
        if engine_path.is_file() and engine_path.suffix == ".engine":
            engine_file = engine_path
        else:
            candidates = list(engine_path.glob("*.engine"))
            if not candidates:
                raise FileNotFoundError(f"No .engine file found in {engine_dir}")
            engine_file = candidates[0]
        
        print(f"[DiT TRT] Loading engine: {engine_file}")
        
        # Загружаем engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_file, "rb") as f:
            engine_bytes = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine from {engine_file}")
        
        self.context = self.engine.create_execution_context()
        
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context")
        
        # Определяем входы/выходы
        self._setup_bindings()
        
        print(f"[DiT TRT] Engine loaded successfully")
        print(f"  Inputs:  {self.input_names}")
        print(f"  Outputs: {self.output_names}")
    
    def _setup_bindings(self):
        """Определяем входы/выходы engine."""
        self.input_names = []
        self.output_names = []
        self.bindings = {}
        
        # Проверяем API версию
        if hasattr(self.engine, "num_io_tensors"):
            # Modern API (TRT 10+)
            num_bindings = self.engine.num_io_tensors
            
            for i in range(num_bindings):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_names.append(name)
                
                self.bindings[name] = i
        
        else:
            # Legacy API
            num_bindings = self.engine.num_bindings
            
            for i in range(num_bindings):
                name = self.engine.get_binding_name(i)
                
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                
                self.bindings[name] = i
        
        # Проверяем ожидаемые входы (оригинальные имена из prepare_inputs)
        expected_inputs = {"x", "t", "rope_cos", "rope_sin", "input_lengths"}
        actual_inputs = set(self.input_names)
        
        if not expected_inputs.issubset(actual_inputs):
            missing = expected_inputs - actual_inputs
            raise RuntimeError(
                f"TRT engine missing expected inputs: {missing}\n"
                f"Available inputs: {actual_inputs}"
            )
    
    @torch.inference_mode()
    def __call__(
        self,
        x_embed: torch.Tensor,        # [B, T, H] - переименуем в x для TRT
        t_embed: torch.Tensor,        # [B, H] - переименуем в t для TRT
        rope_cos: torch.Tensor,       # [B, T, head_dim]
        rope_sin: torch.Tensor,       # [B, T, head_dim]
        input_lengths: torch.Tensor,  # [B] int32
    ) -> torch.Tensor:
        """
        Запускаем TRT engine.
        Внимание: x_embed -> x, t_embed -> t для совместимости с prepare_inputs.
        """
        B, T, H = x_embed.shape
        
        # Приводим к нужному dtype
        x = x_embed.to(self.dtype).contiguous()
        t = t_embed.to(self.dtype).contiguous()
        rope_cos = rope_cos.to(self.dtype).contiguous()
        rope_sin = rope_sin.to(self.dtype).contiguous()
        input_lengths = input_lengths.to(torch.int32).contiguous()
        
        # Устанавливаем формы входов
        stream = torch.cuda.current_stream(self.device)
        
        if hasattr(self.context, "set_input_shape"):
            # Modern API
            self.context.set_input_shape("x", tuple(x.shape))
            self.context.set_input_shape("t", tuple(t.shape))
            self.context.set_input_shape("rope_cos", tuple(rope_cos.shape))
            self.context.set_input_shape("rope_sin", tuple(rope_sin.shape))
            self.context.set_input_shape("input_lengths", tuple(input_lengths.shape))
        else:
            # Legacy API
            self.context.set_binding_shape(self.bindings["x"], tuple(x.shape))
            self.context.set_binding_shape(self.bindings["t"], tuple(t.shape))
            self.context.set_binding_shape(self.bindings["rope_cos"], tuple(rope_cos.shape))
            self.context.set_binding_shape(self.bindings["rope_sin"], tuple(rope_sin.shape))
            self.context.set_binding_shape(self.bindings["input_lengths"], tuple(input_lengths.shape))
        
        # Получаем форму выхода
        output_name = self.output_names[0]  # обычно 'hidden'
        
        if hasattr(self.context, "get_tensor_shape"):
            output_shape = tuple(self.context.get_tensor_shape(output_name))
        else:
            output_shape = tuple(self.context.get_binding_shape(self.bindings[output_name]))
        
        # Аллокация выхода
        hidden = torch.empty(output_shape, device=self.device, dtype=self.dtype)
        
        # Выполнение
        if hasattr(self.context, "set_tensor_address"):
            # Modern API
            self.context.set_tensor_address("x", x.data_ptr())
            self.context.set_tensor_address("t", t.data_ptr())
            self.context.set_tensor_address("rope_cos", rope_cos.data_ptr())
            self.context.set_tensor_address("rope_sin", rope_sin.data_ptr())
            self.context.set_tensor_address("input_lengths", input_lengths.data_ptr())
            self.context.set_tensor_address(output_name, hidden.data_ptr())
            
            ok = self.context.execute_async_v3(stream.cuda_stream)
        
        else:
            # Legacy API
            bindings = [0] * self.engine.num_bindings
            bindings[self.bindings["x"]] = x.data_ptr()
            bindings[self.bindings["t"]] = t.data_ptr()
            bindings[self.bindings["rope_cos"]] = rope_cos.data_ptr()
            bindings[self.bindings["rope_sin"]] = rope_sin.data_ptr()
            bindings[self.bindings["input_lengths"]] = input_lengths.data_ptr()
            bindings[self.bindings[output_name]] = hidden.data_ptr()
            
            if hasattr(self.context, "execute_async_v2"):
                ok = self.context.execute_async_v2(bindings, stream.cuda_stream)
            else:
                ok = self.context.execute_v2(bindings)
        
        if not ok:
            raise RuntimeError("TRT DiT execution failed")
        
        return hidden.to(x_embed.dtype)


class F5DiTTRT(nn.Module):
    """
    Гибридный DiT:
      - Embeddings + финальные слои → PyTorch
      - Transformer blocks → TensorRT
    """
    
    def __init__(
        self,
        ckpt_path: str,
        trt_dit_dir: str,
        *,
        mel_dim: int = 100,
        hidden: int = 1024,
        num_heads: int = 16,
        dim_head: int = 64,
        text_num_embeds: int | None = None,
        text_dim: int | None = None,
        text_mask_padding: bool = True,
        text_embedding_average_upsampling: bool = False,
        conv_layers: int = 0,
        conv_mult: int = 2,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        tokenizer_vocab_size: int | None = None,
    ):
        super().__init__()
        
        self.mel_dim = mel_dim
        self.hidden = hidden
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.text_dim = text_dim if text_dim else mel_dim
        self.dtype = dtype
        self.device = torch.device(device)
        
        # === PyTorch компоненты ===
        self.time_embed = TimestepEmbedding(hidden)
        
        num_embeds = tokenizer_vocab_size if tokenizer_vocab_size is not None else (text_num_embeds or 256)
        self.text_embed = TextEmbedding(
            text_num_embeds=num_embeds,
            text_dim=self.text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
            conv_mult=conv_mult,
        )
        
        self.input_embed = InputEmbedding(
            mel_dim=mel_dim,
            text_dim=self.text_dim,
            out_dim=hidden,
        )

        assert self.text_embed.text_embed.num_embeddings == (tokenizer_vocab_size + 1), \
            f"Embedding size {self.text_embed.text_embed.num_embeddings} != vocab_size+1 {tokenizer_vocab_size+1}"
        
        self.norm_out = AdaLayerNorm_Final(hidden)
        self.proj_out = nn.Linear(hidden, mel_dim)
        
        self.text_cond: Optional[torch.Tensor] = None
        self.text_uncond: Optional[torch.Tensor] = None
        
        # Загрузка host-весов
        self._load_host_weights(ckpt_path)
        self.to(self.device).to(self.dtype)
        
        # === TRT DiT ядро ===
        self.dit = _DiTRuntime(trt_dit_dir, device=device, dtype=dtype)
    
    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None
    
    def _rope_cos_sin(self, B: int, T: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(T, device=self.device, dtype=self.dtype).unsqueeze(0)
        idx = torch.arange(head_dim, device=self.device, dtype=self.dtype)
        
        freqs = torch.exp(-math.log(10000.0) * idx / max(1, head_dim - 1))
        ang = pos[:, :, None] * freqs[None, None, :]
        
        cos = torch.cos(ang).expand(B, T, head_dim).contiguous()
        sin = torch.sin(ang).expand(B, T, head_dim).contiguous()
        
        return cos, sin
    
    def _masked_lengths(self, mask: Optional[torch.Tensor], B: int, T: int) -> torch.Tensor:
        if mask is None:
            return torch.full((B,), T, device=self.device, dtype=torch.long)
        return mask.long().sum(-1)
    
    def _get_text_embed(
        self,
        text: torch.Tensor,
        seq_len: int,
        drop_text: bool,
        audio_mask: Optional[torch.Tensor],
        cache: bool,
    ) -> torch.Tensor:
        if cache:
            if drop_text:
                if self.text_uncond is None or self.text_uncond.shape[1] != seq_len:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True, audio_mask=audio_mask)
                return self.text_uncond
            else:
                if self.text_cond is None or self.text_cond.shape[1] != seq_len:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False, audio_mask=audio_mask)
                return self.text_cond
        else:
            return self.text_embed(text, seq_len, drop_text=drop_text, audio_mask=audio_mask)
    
    def _load_host_weights(self, ckpt_path: str):
        try:
            sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            return
        
        sd = sd_all.get("ema_model_state_dict", sd_all)
        
        with torch.no_grad():
            # time_embed
            for key in ["weight", "bias"]:
                for idx in [0, 2]:
                    src = f"ema_model.time_embed.time_mlp.{idx}.{key}"
                    if src in sd:
                        getattr(self.time_embed.time_mlp[idx], key).copy_(sd[src])
            
            # input_embed.proj
            for key in ["weight", "bias"]:
                src = f"ema_model.input_embed.proj.{key}"
                if src in sd:
                    getattr(self.input_embed.proj, key).copy_(sd[src])
            
            # input_embed.conv_pos_embed
            for idx in [0, 2]:
                for key in ["weight", "bias"]:
                    src = f"ema_model.input_embed.conv_pos_embed.conv1d.{idx}.{key}"
                    if src in sd:
                        getattr(self.input_embed.conv_pos_embed.conv1d[idx], key).copy_(sd[src])
            
            # norm_out
            for key in ["weight", "bias"]:
                src = f"ema_model.norm_out.linear.{key}"
                if src in sd:
                    getattr(self.norm_out.linear, key).copy_(sd[src])
            
            # proj_out
            for key in ["weight", "bias"]:
                src = f"ema_model.proj_out.{key}"
                if src in sd:
                    getattr(self.proj_out, key).copy_(sd[src])
            
            # text_embed
            for cand in [
                "ema_model.text_embed.text_embed.weight",
                "ema_model.transformer.text_embed.text_embed.weight",
            ]:
                if cand in sd and sd[cand].shape == self.text_embed.text_embed.weight.shape:
                    self.text_embed.text_embed.weight.copy_(sd[cand])
                    break
    
    @torch.inference_mode()
    def forward(
        self,
        *,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        
        if time.ndim == 0:
            time = time.repeat(B)
        
        t_emb = self.time_embed(time)
        rope_cos, rope_sin = self._rope_cos_sin(B, T, self.dim_head)
        input_lengths = self._masked_lengths(mask, B, T)
        
        def _one_pass(x_in, cond_in, drop_a, drop_t):
            text_embed = self._get_text_embed(
                text=text,
                seq_len=T,
                drop_text=drop_t,
                audio_mask=mask,
                cache=cache,
            )
            
            x_emb = self.input_embed(
                x=x_in.to(self.dtype),
                cond=cond_in.to(self.dtype),
                text_embed=text_embed.to(self.dtype),
                drop_audio_cond=drop_a,
            )
            
            h = self.dit(
                x_embed=x_emb,
                t_embed=t_emb,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                input_lengths=input_lengths,
            )
            
            y = self.norm_out(h, t_emb)
            y = self.proj_out(y).to(x.dtype)
            
            return y
        
        if not cfg_infer:
            return _one_pass(x, cond, drop_audio_cond, drop_text)
        
        y_cond = _one_pass(x, cond, drop_a=False, drop_t=False)
        y_uncond = _one_pass(x, cond, drop_a=True, drop_t=True)
        
        return torch.cat([y_cond, y_uncond], dim=0)