"""从零手写的 HF Llama-2 -> Megatron-Core checkpoint converter（单文件版）。

本文件是 ``llama_hf_to_megatron/`` 包 + ``scripts/`` + ``tests/`` 的合并结果，
内容与原代码一致，只是去掉了包内相对 import、把每个脚本的 main() 变成子命令。
原包和原脚本保持不变，两者可以并存。

模块顺序（对应原文件）：
    config.py -> tp.py -> qkv.py -> mlp.py -> pp.py -> hf_loader.py
    -> mapping.py -> megatron_model.py -> checkpoint.py -> validate.py
    -> scripts/*.py（子命令） -> tests/*.py（测试函数）

命令行:
    python llama_hf_to_megatron_single.py make-tiny        --out /tmp/tiny-llama
    python llama_hf_to_megatron_single.py inspect          --model /tmp/tiny-llama
    python llama_hf_to_megatron_single.py convert          --hf ... --out ... [--tp N --pp M]
    python llama_hf_to_megatron_single.py validate-logits   --hf ... --ckpt ...
    torchrun --nproc_per_node=N llama_hf_to_megatron_single.py validate-tp --hf ... --ckpt ...
    python llama_hf_to_megatron_single.py validate-pp      --hf ... --ckpt ...
    python llama_hf_to_megatron_single.py test             [-- -k qkv -s]
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, replace as dataclass_replace

# 本项目刻意不用 Transformer Engine / Apex（见 build_megatron_llama 的说明），
# megatron-core 每次 import 都会为此刷一堆 fallback 警告，这里统一屏蔽。
for _msg in (
    ".*Transformer Engine and Apex are not installed.*",
    ".*Apex is not installed.*",
    ".*Transformer Engine is not installed.*",
):
    warnings.filterwarnings("ignore", message=_msg)

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = [
    "LlamaShape",
    "parse_hf_config",
    "build_transformer_config",
    "convert_hf_to_megatron",
    "convert_layer",
    "expected_shapes",
]


# =============================================================================
# config：HF Llama config -> Megatron-Core TransformerConfig
#
# 第 1 步（note.md §3 / §30 的 "parse config"）：
# converter 的第一件事永远是 config mapping，参数映射对了但 config 错了，
# logits 一样对不上。
# =============================================================================

@dataclass
class LlamaShape:
    """从 HF config 抽出来的、converter 真正关心的形状信息。"""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int  # HF: num_key_value_heads
    ffn_hidden_size: int  # HF: intermediate_size
    vocab_size: int
    max_position_embeddings: int
    kv_channels: int  # head_dim
    rms_norm_eps: float
    rotary_base: float
    tie_word_embeddings: bool

    @property
    def heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_query_groups

    @property
    def qkv_total_dim(self) -> int:
        """linear_qkv 的输出维度 = (q + k + v) 的 head 数 * head_dim。"""
        return (self.num_attention_heads + 2 * self.num_query_groups) * self.kv_channels


def parse_hf_config(hf_config) -> LlamaShape:
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    return LlamaShape(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        ffn_hidden_size=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        kv_channels=head_dim,
        rms_norm_eps=hf_config.rms_norm_eps,
        rotary_base=getattr(hf_config, "rope_theta", 10000.0),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
    )


def build_transformer_config(
    shape: LlamaShape,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    params_dtype=None,
) -> TransformerConfig:
    """Llama 结构对应的 Megatron TransformerConfig。

    这些字段就是 "Llama 的架构约定"：
      - RMSNorm 而不是 LayerNorm
      - SwiGLU: gated_linear_unit=True + activation_func=silu
      - 所有 linear 无 bias
    """
    if params_dtype is None:
        params_dtype = torch.float32

    return TransformerConfig(
        num_layers=shape.num_layers,
        hidden_size=shape.hidden_size,
        num_attention_heads=shape.num_attention_heads,
        num_query_groups=shape.num_query_groups,
        ffn_hidden_size=shape.ffn_hidden_size,
        kv_channels=shape.kv_channels,
        layernorm_epsilon=shape.rms_norm_eps,
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_activation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        sequence_parallel=False,
        params_dtype=params_dtype,
        bf16=params_dtype == torch.bfloat16,
        fp16=params_dtype == torch.float16,
        attention_softmax_in_fp32=True,
        persist_layer_norm=False,
        bias_dropout_fusion=False,
        apply_rope_fusion=False,
        masked_softmax_fusion=False,
    )


# =============================================================================
# tp：TP 切分的两个基本操作（note.md §12-§15）
#
# Column Parallel: Y = X W^T，W 按 **输出维度** 切  -> dim=0
# Row Parallel:    输入本身已经被切开，W 按 **输入维度** 切 -> dim=1
# =============================================================================


def split_column(weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """ColumnParallelLinear: linear_qkv / linear_fc1 / word_embeddings / output_layer。"""
    assert weight.shape[0] % tp_size == 0, (
        f"输出维度 {weight.shape[0]} 不能被 tp_size={tp_size} 整除"
    )
    return torch.chunk(weight, tp_size, dim=0)[tp_rank].clone()


def split_row(weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """RowParallelLinear: linear_proj / linear_fc2。"""
    assert weight.shape[1] % tp_size == 0, (
        f"输入维度 {weight.shape[1]} 不能被 tp_size={tp_size} 整除"
    )
    return torch.chunk(weight, tp_size, dim=1)[tp_rank].clone()


def gather_column(shards) -> torch.Tensor:
    return torch.cat(list(shards), dim=0)


def gather_row(shards) -> torch.Tensor:
    return torch.cat(list(shards), dim=1)


# =============================================================================
# qkv：HF 的 q_proj/k_proj/v_proj  <->  Megatron 的 linear_qkv
#
# note.md §5 / §16 / §17 说的坑就在这里：**不能简单 torch.cat([q, k, v], dim=0)**。
#
# Megatron-Core 的 `linear_qkv.weight` 是按 **query group 交错(interleaved)** 排列的：
#
#     group 0: [q_0 ... q_{r-1}, k_0, v_0]
#     group 1: [q_r ... q_{2r-1}, k_1, v_1]
#     ...
#     group G-1: [...,            k_{G-1}, v_{G-1}]
#
# 其中 r = heads_per_group = num_attention_heads // num_query_groups。
#
# 这么排的原因是 TP：沿 dim=0 直接 chunk 就能让每个 rank 恰好拿到
# "连续若干个完整 group 的 q/k/v"，不需要任何额外通信或重排。
#
# 对 Llama-2 7B（MHA，G = 32，r = 1）就是最朴素的：
#
#     [q_0, k_0, v_0, q_1, k_1, v_1, ..., q_31, k_31, v_31]
#
# 而**不是** [q_all, k_all, v_all]。
# =============================================================================


def merge_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """HF 的三个矩阵 -> 一个 Megatron linear_qkv.weight（未做 TP 切分）。

    q: [num_attention_heads * head_dim, hidden]
    k: [num_query_groups   * head_dim, hidden]
    v: [num_query_groups   * head_dim, hidden]
    return: [(num_attention_heads + 2*num_query_groups) * head_dim, hidden]
    """
    assert num_attention_heads % num_query_groups == 0
    heads_per_group = num_attention_heads // num_query_groups
    hidden = q.shape[1]

    q = q.reshape(num_query_groups, heads_per_group * head_dim, hidden)
    k = k.reshape(num_query_groups, head_dim, hidden)
    v = v.reshape(num_query_groups, head_dim, hidden)

    # 每个 group 内部按 [q..., k, v] 拼，再把所有 group 顺次展开。
    qkv = torch.cat([q, k, v], dim=1)
    return qkv.reshape(-1, hidden)


def split_qkv(
    qkv: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
):
    """merge_qkv 的逆运算，用于 §24 的 reconstruction test。"""
    assert num_attention_heads % num_query_groups == 0
    heads_per_group = num_attention_heads // num_query_groups
    hidden = qkv.shape[1]

    qkv = qkv.reshape(num_query_groups, (heads_per_group + 2) * head_dim, hidden)
    q = qkv[:, : heads_per_group * head_dim, :]
    k = qkv[:, heads_per_group * head_dim : (heads_per_group + 1) * head_dim, :]
    v = qkv[:, (heads_per_group + 1) * head_dim :, :]

    return (
        q.reshape(num_attention_heads * head_dim, hidden),
        k.reshape(num_query_groups * head_dim, hidden),
        v.reshape(num_query_groups * head_dim, hidden),
    )


def shard_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """按 group 交错排好之后再切 TP —— 这一步就是 QKVMapping 的全部内容。

    因为 layout 已经是 group-major，所以 TP 切分退化成一次普通的
    `torch.chunk(qkv, tp_size, dim=0)`；每个 rank 得到 G/tp_size 个完整 group。
    """
    assert num_query_groups % tp_size == 0, (
        f"num_query_groups={num_query_groups} 必须能被 tp_size={tp_size} 整除，"
        "否则 KV head 无法在 rank 之间均分（真实 Megatron 会做 KV 复制来兜底）"
    )
    full = merge_qkv(q, k, v, num_attention_heads, num_query_groups, head_dim)
    return torch.chunk(full, tp_size, dim=0)[tp_rank].clone()


# =============================================================================
# mlp：Gated MLP (SwiGLU) mapping
#     HF 的 gate_proj/up_proj <-> Megatron 的 linear_fc1（note.md §6 / §18）
#
# Megatron 的 SwiGLU 实现是：
#
#     h = linear_fc1(x)              # [*, 2 * ffn_per_rank]
#     a, b = torch.chunk(h, 2, -1)   # 前一半是 gate，后一半是 up
#     y = linear_fc2(silu(a) * b)
#
# 所以 TP=1 时 linear_fc1.weight = cat([gate, up], dim=0)。
#
# TP>1 时**顺序很关键**：每个 rank 上必须是
#
#     [gate_shard_i, up_shard_i]
#
# 也就是「先各自切、再拼」，而不是「先拼、再切」。
# 后者会让 rank 0 拿到整个 gate、rank 1 拿到整个 up，chunk(h, 2, -1) 之后
# gate/up 完全错位。
# =============================================================================


def merge_fc1(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.cat([gate, up], dim=0)


def split_fc1(fc1: torch.Tensor):
    """merge_fc1 的逆运算。"""
    assert fc1.shape[0] % 2 == 0
    ffn = fc1.shape[0] // 2
    return fc1[:ffn].clone(), fc1[ffn:].clone()


def shard_fc1(
    gate: torch.Tensor, up: torch.Tensor, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """GatedMLPMapping：先各自 chunk，再 cat。"""
    assert gate.shape == up.shape
    assert gate.shape[0] % tp_size == 0, (
        f"ffn_hidden_size={gate.shape[0]} 不能被 tp_size={tp_size} 整除"
    )
    gate_shard = torch.chunk(gate, tp_size, dim=0)[tp_rank]
    up_shard = torch.chunk(up, tp_size, dim=0)[tp_rank]
    return torch.cat([gate_shard, up_shard], dim=0).clone()


def gather_fc1(shards) -> torch.Tensor:
    """把各 rank 的 linear_fc1 还原成完整的 [gate; up]（reconstruction test 用）。"""
    gates, ups = [], []
    for shard in shards:
        g, u = split_fc1(shard)
        gates.append(g)
        ups.append(u)
    return torch.cat(gates + ups, dim=0)


# =============================================================================
# pp：Pipeline Parallel 的 layer 划分与 "谁拥有哪些非 layer 参数"
#     （note.md §20 / §21）
#
# PP 下每个 stage 只保存自己那一段：
#
#     - 只有第一个 stage (pre_process)  拥有 embedding
#     - 只有最后一个 stage (post_process) 拥有 final_layernorm 和 output_layer
#     - 中间 stage 只有 decoder.layers.*
#
# 注意 Megatron 的 checkpoint 里 layer 的编号是 **stage 内的局部编号**：
# global layer 16 在 PP rank 1 上叫 decoder.layers.0。
# =============================================================================


def get_pp_layer_range(num_layers: int, pp_rank: int, pp_size: int):
    """返回该 stage 负责的 [start, end) 全局 layer 区间。

    这里只实现均分（Megatron 还支持 uneven partition / VPP，属于后续话题）。
    """
    assert num_layers % pp_size == 0, (
        f"num_layers={num_layers} 不能被 pp_size={pp_size} 整除；"
        "真实 Megatron 需要 --decoder-first/last-pipeline-num-layers 处理不均匀切分"
    )
    per_stage = num_layers // pp_size
    start = pp_rank * per_stage
    return start, start + per_stage


def is_pre_process(pp_rank: int, pp_size: int) -> bool:
    return pp_rank == 0


def is_post_process(pp_rank: int, pp_size: int) -> bool:
    return pp_rank == pp_size - 1


def rank_of(tp_rank: int, pp_rank: int, tp_size: int) -> int:
    """Megatron 默认 rank 排布是 TP 最内层、PP 最外层。

        rank = pp_rank * tp_size + tp_rank

    TP=2, PP=2 时：
                 PP0      PP1
        TP0    rank 0   rank 2
        TP1    rank 1   rank 3
    """
    return pp_rank * tp_size + tp_rank


# =============================================================================
# hf_loader：HF 侧的加载工具（note.md §3 / §27）
#
# 除了加载真的 Llama-2 7B，这里还提供一个 tiny-Llama 生成器：
# converter 的 bug 用 hidden=128 / 2 layers 的模型 debug 成本远低于 7B。
# =============================================================================


def load_hf_llama(model_path: str, dtype=torch.float32, device="cpu"):
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    return model.to(device)


def print_state_dict(model, limit: int | None = None):
    for i, (name, tensor) in enumerate(model.state_dict().items()):
        if limit is not None and i >= limit:
            print("...")
            break
        print(f"{name:60s} {tuple(tensor.shape)}")


def make_tiny_llama(
    save_dir: str,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int | None = None,
    intermediate_size: int = 256,
    vocab_size: int = 1000,
    max_position_embeddings: int = 256,
    seed: int = 0,
    dtype=torch.float32,
):
    """随机初始化一个结构与 Llama-2 完全一致的小模型并存成 HF 格式。

    num_key_value_heads 可以设成 < num_attention_heads 来顺便验证 GQA 分支
    （Llama-2 7B 本身是 MHA，覆盖不到 GQA 的交错逻辑）。
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads or num_attention_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(dtype)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "generation_config.json"), "w") as f:
        json.dump({"_from_model_config": True}, f)
    return model


# =============================================================================
# mapping：HF state_dict -> 某个 (tp_rank, pp_rank) 的 Megatron state_dict
#
# note.md §4 的映射表在这里落地：
#
#     model.embed_tokens.weight              -> embedding.word_embeddings.weight
#     model.layers.i.input_layernorm         -> decoder.layers.j.input_layernorm
#     q_proj / k_proj / v_proj               -> decoder.layers.j.self_attention.linear_qkv
#     o_proj                                 -> decoder.layers.j.self_attention.linear_proj
#     post_attention_layernorm               -> decoder.layers.j.pre_mlp_layernorm
#     gate_proj / up_proj                    -> decoder.layers.j.mlp.linear_fc1
#     down_proj                              -> decoder.layers.j.mlp.linear_fc2
#     model.norm.weight                      -> decoder.final_layernorm.weight
#     lm_head.weight                         -> output_layer.weight
#
# 其中 i 是全局 layer 号，j 是 PP stage 内的局部 layer 号。
# =============================================================================


def convert_layer(
    hf_sd: dict,
    global_layer: int,
    local_layer: int,
    shape: LlamaShape,
    tp_rank: int,
    tp_size: int,
) -> dict:
    p = f"model.layers.{global_layer}."
    out = f"decoder.layers.{local_layer}."

    qkv = shard_qkv(
        hf_sd[p + "self_attn.q_proj.weight"],
        hf_sd[p + "self_attn.k_proj.weight"],
        hf_sd[p + "self_attn.v_proj.weight"],
        tp_rank=tp_rank,
        tp_size=tp_size,
        num_attention_heads=shape.num_attention_heads,
        num_query_groups=shape.num_query_groups,
        head_dim=shape.kv_channels,
    )
    fc1 = shard_fc1(
        hf_sd[p + "mlp.gate_proj.weight"],
        hf_sd[p + "mlp.up_proj.weight"],
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    return {
        out + "input_layernorm.weight": hf_sd[p + "input_layernorm.weight"].clone(),
        out + "self_attention.linear_qkv.weight": qkv,
        out + "self_attention.linear_proj.weight": split_row(
            hf_sd[p + "self_attn.o_proj.weight"], tp_rank, tp_size
        ),
        out + "pre_mlp_layernorm.weight": hf_sd[
            p + "post_attention_layernorm.weight"
        ].clone(),
        out + "mlp.linear_fc1.weight": fc1,
        out + "mlp.linear_fc2.weight": split_row(
            hf_sd[p + "mlp.down_proj.weight"], tp_rank, tp_size
        ),
    }


def convert_hf_to_megatron(
    hf_sd: dict,
    shape: LlamaShape,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
) -> dict:
    """产出单个 model-parallel rank 的完整 Megatron state_dict。"""
    sd = {}

    if is_pre_process(pp_rank, pp_size):
        sd["embedding.word_embeddings.weight"] = split_column(
            hf_sd["model.embed_tokens.weight"], tp_rank, tp_size
        )

    start, end = get_pp_layer_range(shape.num_layers, pp_rank, pp_size)
    for global_layer in range(start, end):
        sd.update(
            convert_layer(
                hf_sd,
                global_layer=global_layer,
                local_layer=global_layer - start,
                shape=shape,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        )

    if is_post_process(pp_rank, pp_size):
        sd["decoder.final_layernorm.weight"] = hf_sd["model.norm.weight"].clone()
        if not shape.tie_word_embeddings:
            # tied 的情况下 Megatron 不存 output_layer，权重与 embedding 共享。
            sd["output_layer.weight"] = split_column(
                hf_sd["lm_head.weight"], tp_rank, tp_size
            )

    return sd


def expected_shapes(
    shape: LlamaShape, tp_size: int = 1, pp_size: int = 1, pp_rank: int = 0
) -> dict:
    """note.md §23 的 shape 断言表，用于在不启动 Megatron 的情况下自检。"""
    h = shape.hidden_size
    exp = {}
    if is_pre_process(pp_rank, pp_size):
        exp["embedding.word_embeddings.weight"] = (shape.vocab_size // tp_size, h)
    start, end = get_pp_layer_range(shape.num_layers, pp_rank, pp_size)
    for j in range(end - start):
        out = f"decoder.layers.{j}."
        exp[out + "input_layernorm.weight"] = (h,)
        exp[out + "self_attention.linear_qkv.weight"] = (shape.qkv_total_dim // tp_size, h)
        exp[out + "self_attention.linear_proj.weight"] = (
            h,
            shape.num_attention_heads * shape.kv_channels // tp_size,
        )
        exp[out + "pre_mlp_layernorm.weight"] = (h,)
        exp[out + "mlp.linear_fc1.weight"] = (2 * shape.ffn_hidden_size // tp_size, h)
        exp[out + "mlp.linear_fc2.weight"] = (h, shape.ffn_hidden_size // tp_size)
    if is_post_process(pp_rank, pp_size):
        exp["decoder.final_layernorm.weight"] = (h,)
        if not shape.tie_word_embeddings:
            exp["output_layer.weight"] = (shape.vocab_size // tp_size, h)
    return exp


# =============================================================================
# megatron_model：构建 Megatron-Core GPTModel，并初始化 model-parallel 环境
#
# 用的是 **local spec**（`get_gpt_layer_local_spec`）而不是 TE spec：
# 纯 PyTorch 实现，不依赖 Transformer Engine，数值上更容易和 HF 对齐，
# 适合做 converter 验证。生产训练再换 TE spec。
# =============================================================================


def init_distributed(tp_size: int = 1, pp_size: int = 1, seed: int = 1234):
    """torchrun 启动时用环境变量；单进程调试时自己拉一个 world_size=1 的 group。"""
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29513")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )
        model_parallel_cuda_manual_seed(seed)

    return (
        parallel_state.get_tensor_model_parallel_rank(),
        parallel_state.get_pipeline_model_parallel_rank(),
    )


def build_megatron_llama(
    shape: LlamaShape,
    tp_size: int = 1,
    pp_size: int = 1,
    pre_process: bool = True,
    post_process: bool = True,
    dtype=torch.float32,
    device="cuda",
) -> GPTModel:
    config = build_transformer_config(
        shape,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        params_dtype=dtype,
    )
    spec = get_gpt_layer_local_spec(normalization="RMSNorm")

    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=shape.vocab_size,
        max_sequence_length=shape.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=shape.tie_word_embeddings,
        position_embedding_type="rope",
        rotary_base=int(shape.rotary_base),
        parallel_output=False,  # 让 logits 在 TP 维度上做 all-gather，方便和 HF 比
    )
    return model.to(device=device, dtype=dtype).eval()


# =============================================================================
# checkpoint：落盘（note.md §21 / §22）
#
# 提供两种格式：
#
# 1. ``torch`` —— Megatron 经典布局，纯 torch.save，不需要起 distributed：
#
#        ckpt/
#          latest_checkpointed_iteration.txt
#          llama_shape.json                # 本项目自己加的 meta，方便复现
#          iter_0000001/
#            mp_rank_00/model_optim_rng.pt          # PP=1
#            mp_rank_00_000/model_optim_rng.pt      # PP>1 时带 pp 后缀
#
# 2. ``torch_dist`` —— Megatron-Core 的 distributed checkpoint（DCP）。
#    必须在 torchrun 下、且已经建好真实的分布式模型才能用，
#    因为它保存的是 model.sharded_state_dict() 里的 ShardedTensor 元信息。
# =============================================================================

CHECKPOINT_VERSION = 3.0


def mp_rank_dir(tp_rank: int, pp_rank: int, pp_size: int) -> str:
    if pp_size == 1:
        return f"mp_rank_{tp_rank:02d}"
    return f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"


def save_torch_checkpoint(
    ckpt_dir: str,
    state_dict: dict,
    shape: LlamaShape,
    tp_rank: int,
    pp_rank: int,
    tp_size: int,
    pp_size: int,
    iteration: int = 1,
):
    sub = os.path.join(ckpt_dir, f"iter_{iteration:07d}", mp_rank_dir(tp_rank, pp_rank, pp_size))
    os.makedirs(sub, exist_ok=True)
    torch.save(
        {
            "model": state_dict,
            "checkpoint_version": CHECKPOINT_VERSION,
            "iteration": iteration,
            "tensor_model_parallel_size": tp_size,
            "pipeline_model_parallel_size": pp_size,
        },
        os.path.join(sub, "model_optim_rng.pt"),
    )
    return sub


def write_meta(ckpt_dir: str, shape: LlamaShape, tp_size: int, pp_size: int, iteration: int = 1):
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(iteration))
    with open(os.path.join(ckpt_dir, "llama_shape.json"), "w") as f:
        json.dump(
            {
                "shape": asdict(shape),
                "tensor_model_parallel_size": tp_size,
                "pipeline_model_parallel_size": pp_size,
                "iteration": iteration,
            },
            f,
            indent=2,
        )


def read_meta(ckpt_dir: str):
    with open(os.path.join(ckpt_dir, "llama_shape.json")) as f:
        meta = json.load(f)
    return (
        LlamaShape(**meta["shape"]),
        meta["tensor_model_parallel_size"],
        meta["pipeline_model_parallel_size"],
        meta["iteration"],
    )


def load_torch_checkpoint(ckpt_dir: str, tp_rank: int, pp_rank: int, pp_size: int, iteration: int = 1):
    path = os.path.join(
        ckpt_dir,
        f"iter_{iteration:07d}",
        mp_rank_dir(tp_rank, pp_rank, pp_size),
        "model_optim_rng.pt",
    )
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def save_dist_checkpoint(ckpt_dir: str, model, iteration: int = 1):
    """Megatron-Core DCP：由每个 rank 各自调用，框架负责聚合分片元信息。"""
    from megatron.core import dist_checkpointing

    path = os.path.join(ckpt_dir, f"iter_{iteration:07d}")
    os.makedirs(path, exist_ok=True)
    sharded = model.sharded_state_dict()
    dist_checkpointing.save(sharded, path)
    return path


def load_dist_checkpoint(ckpt_dir: str, model, iteration: int = 1):
    from megatron.core import dist_checkpointing

    path = os.path.join(ckpt_dir, f"iter_{iteration:07d}")
    sharded = model.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded, path)
    model.load_state_dict(loaded, strict=False)
    return model


# =============================================================================
# validate：数值验证（note.md §23-§26）
#
# 三层验证，越往后越贵：
#
# 1. shape 断言                       -> expected_shapes
# 2. reconstruction test（切了再拼回去）-> tests/test_tp.py
# 3. forward 对齐：逐层 hidden state + 最终 logits  -> 本节
#
# 第 3 步的关键是**逐层比**：只看 logits，出错时你不知道 bug 在哪一层；
# 按 §26 的顺序 layer-by-layer 比较，第一个 diff 爆炸的层就是 bug 所在。
# =============================================================================


def causal_mask(seq_len: int, device, batch_size: int = 1) -> torch.Tensor:
    """Megatron 的约定：True = 需要被 mask 掉。"""
    m = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    return m.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float()
    b = b.float()
    abs_diff = (a - b).abs()
    return {
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "rel_mean": (abs_diff / b.abs().clamp_min(1e-8)).mean().item(),
        "max_val": b.abs().max().item(),
    }


class HiddenStateCollector:
    """在 HF / Megatron 的每个 decoder layer 上挂 hook，收集输出 hidden state。

    Megatron 内部 hidden state 是 [s, b, h]，HF 是 [b, s, h]，
    统一转成 [b, s, h] 再比较。
    """

    def __init__(self, layers, layout: str):
        assert layout in ("sbh", "bsh")
        self.layout = layout
        self.outputs = {}
        self.handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(layers)
        ]

    def _make_hook(self, idx):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.layout == "sbh":
                h = h.transpose(0, 1)
            self.outputs[idx] = h.detach().float()

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def hf_forward(hf_model, input_ids):
    out = hf_model(input_ids=input_ids)
    return out.logits.float()


@torch.no_grad()
def megatron_forward(mg_model, input_ids):
    b, s = input_ids.shape
    position_ids = (
        torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s).contiguous()
    )
    logits = mg_model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=causal_mask(s, input_ids.device, b),
    )
    return logits.float()


def compare_layerwise(hf_model, mg_model, input_ids, tol: float = 1e-3, verbose=True):
    """逐层 + 最终 logits 比较，返回 (all_ok, report)。"""
    hf_col = HiddenStateCollector(hf_model.model.layers, layout="bsh")
    mg_col = HiddenStateCollector(mg_model.decoder.layers, layout="sbh")
    try:
        hf_logits = hf_forward(hf_model, input_ids)
        mg_logits = megatron_forward(mg_model, input_ids)
    finally:
        hf_col.remove()
        mg_col.remove()

    report = []
    all_ok = True
    for i in sorted(mg_col.outputs):
        if i not in hf_col.outputs:
            continue
        st = diff_stats(mg_col.outputs[i], hf_col.outputs[i])
        ok = st["max_abs"] <= tol * max(1.0, st["max_val"])
        all_ok &= ok
        report.append((f"layer {i}", st, ok))

    st = diff_stats(mg_logits, hf_logits)
    ok = st["max_abs"] <= tol * max(1.0, st["max_val"])
    all_ok &= ok
    report.append(("logits", st, ok))

    if verbose:
        print(f"{'stage':>10s} {'max_abs':>12s} {'mean_abs':>12s} {'rel_mean':>12s}  ok")
        for name, st, ok in report:
            print(
                f"{name:>10s} {st['max_abs']:12.3e} {st['mean_abs']:12.3e} "
                f"{st['rel_mean']:12.3e}  {'PASS' if ok else 'FAIL'}"
            )
        hf_top = hf_logits[0, -1].argmax().item()
        mg_top = mg_logits[0, -1].argmax().item()
        print(f"argmax(last token): HF={hf_top} Megatron={mg_top}")

    return all_ok, report


# =============================================================================
# scripts：命令行子命令（原 scripts/*.py 合并而来）
#
#   make-tiny        <- scripts/make_tiny_llama.py
#   inspect          <- scripts/01_inspect_hf.py
#   convert          <- scripts/02_convert.py
#   validate-logits  <- scripts/03_validate_logits.py
#   validate-tp      <- scripts/04_validate_tp.py   （需 torchrun）
#   validate-pp      <- scripts/05_validate_pp.py
#   test             <- pytest tests/  （测试函数也在本文件里）
# =============================================================================

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def cmd_make_tiny(args):
    """生成一个 tiny Llama（HF 格式），用于快速迭代 converter。"""
    model = make_tiny_llama(
        args.out,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"saved tiny llama to {args.out}  ({n / 1e6:.2f}M params)")
    print(model.config)
    return 0


def cmd_inspect(args):
    """Step 1（note.md §3）：把 HF Llama 的 state_dict 和 config 打出来看。"""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.model)
    shape = parse_hf_config(hf_config)

    print("=== HF config -> LlamaShape ===")
    for k, v in shape.__dict__.items():
        print(f"  {k:26s} = {v}")
    print(f"  {'heads_per_group':26s} = {shape.heads_per_group}")
    print(f"  {'qkv_total_dim':26s} = {shape.qkv_total_dim}")

    if not args.config_only:
        print("\n=== HF state_dict ===")
        model = load_hf_llama(args.model, dtype=torch.float32, device="cpu")
        print_state_dict(model, limit=args.limit)

    print(f"\n=== 期望的 Megatron 参数形状 (tp={args.tp}) ===")
    for k, v in expected_shapes(shape, tp_size=args.tp).items():
        print(f"  {k:60s} {v}")
    return 0


def cmd_convert(args):
    """Step 2：HF -> Megatron checkpoint（TP/PP-aware），单进程离线转换。

    单进程就能产出所有 rank 的分片 —— 这正是 note.md §22 建议的
    "先不要碰 distributed checkpoint" 的做法。
    """
    dtype = DTYPES[args.dtype]

    from transformers import AutoConfig

    shape = parse_hf_config(AutoConfig.from_pretrained(args.hf))
    print(f"[config] layers={shape.num_layers} hidden={shape.hidden_size} "
          f"heads={shape.num_attention_heads} kv_groups={shape.num_query_groups} "
          f"ffn={shape.ffn_hidden_size} vocab={shape.vocab_size} "
          f"tied_embeddings={shape.tie_word_embeddings}")

    print(f"[load] {args.hf} dtype={args.dtype}")
    hf_model = load_hf_llama(args.hf, dtype=dtype, device="cpu")
    hf_sd = hf_model.state_dict()

    write_meta(args.out, shape, args.tp, args.pp)
    total = 0
    for pp_rank in range(args.pp):
        for tp_rank in range(args.tp):
            sd = convert_hf_to_megatron(
                hf_sd, shape, tp_rank=tp_rank, tp_size=args.tp,
                pp_rank=pp_rank, pp_size=args.pp,
            )
            if args.check_shapes:
                exp = expected_shapes(shape, tp_size=args.tp, pp_size=args.pp, pp_rank=pp_rank)
                assert set(exp) == set(sd), (
                    f"参数名不匹配\n  多: {set(sd) - set(exp)}\n  少: {set(exp) - set(sd)}"
                )
                for k, want in exp.items():
                    assert tuple(sd[k].shape) == want, f"{k}: got {tuple(sd[k].shape)}, want {want}"

            n = sum(v.numel() for v in sd.values())
            total += n
            path = save_torch_checkpoint(
                args.out, sd, shape, tp_rank, pp_rank, args.tp, args.pp
            )
            print(f"[save] rank {rank_of(tp_rank, pp_rank, args.tp)} "
                  f"(tp={tp_rank}, pp={pp_rank})  {n / 1e6:8.2f}M params -> {path}")

    print(f"[done] 共 {args.tp * args.pp} 个分片, {total / 1e6:.2f}M params, shape 断言全部通过")
    return 0


def cmd_validate_logits(args):
    """Step 3（note.md §25 / §26）：TP=1 / PP=1 下逐层对比 HF 与 Megatron。

    单进程单卡即可运行 —— 这是最应该先跑通的一步。
    """
    dtype = DTYPES[args.dtype]
    shape, tp_size, pp_size, iteration = read_meta(args.ckpt)
    assert tp_size == 1 and pp_size == 1, (
        f"该子命令只处理 TP=1/PP=1，当前 checkpoint 是 tp={tp_size} pp={pp_size}；"
        "请用 validate-tp（torchrun）"
    )

    init_distributed(1, 1)
    device = "cuda"

    print("[build] Megatron GPTModel (local spec, RMSNorm + SwiGLU + RoPE)")
    mg_model = build_megatron_llama(shape, dtype=dtype, device=device)
    sd = load_torch_checkpoint(args.ckpt, 0, 0, 1, iteration)
    missing, unexpected = mg_model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("_extra_state")]
    missing = [k for k in missing if not k.endswith("_extra_state")]
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    print(f"[load] {len(sd)} 个张量全部对上（忽略 _extra_state）")

    hf_model = load_hf_llama(args.hf, dtype=dtype, device=device)

    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, shape.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    print(f"\n[compare] dtype={args.dtype} tol={args.tol} "
          f"input={tuple(input_ids.shape)}")
    ok, _ = compare_layerwise(hf_model, mg_model, input_ids, tol=args.tol)
    print("\nRESULT:", "PASS — converter 与 HF 数值一致" if ok else "FAIL — 见上表第一个 FAIL 的层")
    return 0 if ok else 1


def cmd_validate_tp(args):
    """Step 4（note.md §12-§19）：TP>1 下验证 checkpoint 分片。

    必须用 torchrun 启动，每个 rank 只加载自己的分片：

        torchrun --nproc_per_node=2 llama_hf_to_megatron_single.py validate-tp \\
            --hf /tmp/tiny-llama --ckpt /tmp/ckpt_tiny_tp2
    """
    dtype = DTYPES[args.dtype]
    shape, tp_size, pp_size, iteration = read_meta(args.ckpt)
    assert pp_size == 1, "本子命令只验证 TP；PP 用 validate-pp"

    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert world == tp_size, f"需要 torchrun --nproc_per_node={tp_size}，当前 world_size={world}"

    tp_rank, pp_rank = init_distributed(tp_size, pp_size)
    is_main = torch.distributed.get_rank() == 0

    mg_model = build_megatron_llama(shape, tp_size=tp_size, dtype=dtype, device="cuda")
    sd = load_torch_checkpoint(args.ckpt, tp_rank, pp_rank, pp_size, iteration)
    missing, unexpected = mg_model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("_extra_state")]
    unexpected = [k for k in unexpected if not k.endswith("_extra_state")]
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    print(f"[rank {torch.distributed.get_rank()}] tp_rank={tp_rank} 载入 {len(sd)} 个张量，"
          f"linear_qkv shard = "
          f"{tuple(sd['decoder.layers.0.self_attention.linear_qkv.weight'].shape)}")

    hf_model = load_hf_llama(args.hf, dtype=dtype, device="cuda")

    torch.manual_seed(args.seed)  # 所有 rank 必须喂同一份输入
    input_ids = torch.randint(
        0, shape.vocab_size, (args.batch_size, args.seq_len), device="cuda"
    )

    torch.distributed.barrier()
    ok, _ = compare_layerwise(hf_model, mg_model, input_ids, tol=args.tol, verbose=is_main)

    flag = torch.tensor([1 if ok else 0], device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    ok = bool(flag.item())

    if args.save_dist:
        path = save_dist_checkpoint(args.save_dist, mg_model, iteration)
        if is_main:
            print(f"[dist-ckpt] saved to {path}")

    if is_main:
        print("\nRESULT:", f"PASS — TP={tp_size} 分片与 HF 数值一致" if ok else "FAIL")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0 if ok else 1


def build_stage(shape, pp_rank, pp_size, dtype, device):
    """把 config.num_layers 改成 per-stage 层数来"伪装"成一个 PP stage。

    这是模拟手段：真实 Megatron 里 TransformerBlock 会自己按
    num_layers // pp_size 只建本 stage 的层。
    """
    start, end = get_pp_layer_range(shape.num_layers, pp_rank, pp_size)
    stage_shape = dataclass_replace(shape, num_layers=end - start)
    return build_megatron_llama(
        stage_shape,
        tp_size=1,
        pp_size=1,
        pre_process=is_pre_process(pp_rank, pp_size),
        post_process=is_post_process(pp_rank, pp_size),
        dtype=dtype,
        device=device,
    )


def cmd_validate_pp(args):
    """Step 5（note.md §20 / §21）：PP 分片的验证（单进程模拟）。

    真实 Megatron 的 PP 需要 pipeline schedule + 跨 rank send/recv。为了在单卡上把
    "PP 分片是否正确" 这件事验证清楚，这里做一个**串行模拟**：

        stage 0 (pre_process=True,  post_process=False)  ->  hidden states
        stage 1 (pre_process=False, post_process=True )  ->  logits

    每个 stage 只加载自己那个 mp_rank_XX_YYY 目录里的权重，
    hidden states 用 decoder_input 手动传给下一个 stage。

    这样能验证的东西和真 PP 一样重要：
      - layer 的全局编号 -> stage 内局部编号 的换算对不对
      - embedding / final_layernorm / output_layer 是否落在正确的 stage
      - 拼起来的整条链路是否和 HF 数值一致
    """
    dtype = DTYPES[args.dtype]
    shape, tp_size, pp_size, iteration = read_meta(args.ckpt)
    assert tp_size == 1, "PP 模拟只支持 TP=1（TP 已由 validate-tp 覆盖）"
    assert pp_size > 1, "checkpoint 的 pp_size=1，没什么可模拟的"

    init_distributed(1, 1)
    device = "cuda"

    stages = []
    for pp_rank in range(pp_size):
        model = build_stage(shape, pp_rank, pp_size, dtype, device)
        sd = load_torch_checkpoint(args.ckpt, 0, pp_rank, pp_size, iteration)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("_extra_state")]
        unexpected = [k for k in unexpected if not k.endswith("_extra_state")]
        assert not missing and not unexpected, (
            f"pp_rank={pp_rank}: missing={missing} unexpected={unexpected}"
        )
        start, end = get_pp_layer_range(shape.num_layers, pp_rank, pp_size)
        print(f"[stage {pp_rank}] global layers [{start}, {end})  "
              f"pre={is_pre_process(pp_rank, pp_size)} "
              f"post={is_post_process(pp_rank, pp_size)}  "
              f"{len(sd)} tensors")
        stages.append(model)

    hf_model = load_hf_llama(args.hf, dtype=dtype, device=device)

    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, shape.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    b, s = input_ids.shape
    position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, s).contiguous()
    mask = causal_mask(s, device, b)

    # HF 侧收集每层输出，Megatron 侧按 stage 收集后拼成全局层序
    hf_col = HiddenStateCollector(hf_model.model.layers, layout="bsh")
    mg_hidden = {}
    cols = [HiddenStateCollector(m.decoder.layers, layout="sbh") for m in stages]
    try:
        with torch.no_grad():
            hf_logits = hf_forward(hf_model, input_ids)
            hidden = None
            for pp_rank, model in enumerate(stages):
                if hidden is not None:
                    # 真实 PP 里这一步是 recv from previous stage：
                    # TransformerBlock 在 pre_process=False 时只认 input_tensor，
                    # 不看 forward 的 hidden_states 参数。
                    model.set_input_tensor(hidden)
                out = model(
                    input_ids=input_ids if pp_rank == 0 else None,
                    position_ids=position_ids,
                    attention_mask=mask,
                )
                if is_post_process(pp_rank, pp_size):
                    mg_logits = out.float()
                else:
                    hidden = out
                start, _ = get_pp_layer_range(shape.num_layers, pp_rank, pp_size)
                for local, h in cols[pp_rank].outputs.items():
                    mg_hidden[start + local] = h
    finally:
        hf_col.remove()
        for c in cols:
            c.remove()

    print(f"\n{'stage':>10s} {'max_abs':>12s} {'mean_abs':>12s} {'rel_mean':>12s}  ok")
    all_ok = True
    for i in sorted(mg_hidden):
        st = diff_stats(mg_hidden[i], hf_col.outputs[i])
        ok = st["max_abs"] <= args.tol * max(1.0, st["max_val"])
        all_ok &= ok
        print(f"{'layer ' + str(i):>10s} {st['max_abs']:12.3e} {st['mean_abs']:12.3e} "
              f"{st['rel_mean']:12.3e}  {'PASS' if ok else 'FAIL'}")
    st = diff_stats(mg_logits, hf_logits)
    ok = st["max_abs"] <= args.tol * max(1.0, st["max_val"])
    all_ok &= ok
    print(f"{'logits':>10s} {st['max_abs']:12.3e} {st['mean_abs']:12.3e} "
          f"{st['rel_mean']:12.3e}  {'PASS' if ok else 'FAIL'}")

    print("\nRESULT:", f"PASS — PP={pp_size} 分片串起来与 HF 一致" if all_ok else "FAIL")
    return 0 if all_ok else 1


# =============================================================================
# tests：原 tests/*.py 的测试函数（由 `test` 子命令用 pytest 收集本文件执行）
#
#   test_qkv.py         -> test_merge_shape_* / test_layout_is_group_interleaved / ...
#   test_mlp.py         -> test_shard_keeps_gate_up_pairing_per_rank / ...
#   test_tp.py          -> test_column_split_* / test_pp_* / test_rank_layout_tp2_pp2
#   test_mapping.py     -> test_names_and_shapes_match_expectation / test_full_roundtrip_*
#   test_logits_tiny.py -> test_tiny_llama_matches_hf（需要 GPU）
#
# 没装 pytest 时用一个只满足装饰器语法的 stub，保证本文件仍可正常 import。
# =============================================================================

try:
    import pytest
except ImportError:  # pragma: no cover
    class _MarkStub:
        def parametrize(self, *a, **kw):
            return lambda f: f

        def skipif(self, *a, **kw):
            return lambda f: f

    class _PytestStub:
        mark = _MarkStub()

        def fixture(self, *a, **kw):
            return (lambda f: f) if not a else a[0]

        def raises(self, *a, **kw):
            raise RuntimeError("需要安装 pytest 才能运行测试")

    pytest = _PytestStub()


# ---- tests/test_qkv.py ------------------------------------------------------


def make_qkv(heads, groups, head_dim, hidden):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(heads * head_dim, hidden, generator=g)
    k = torch.randn(groups * head_dim, hidden, generator=g)
    v = torch.randn(groups * head_dim, hidden, generator=g)
    return q, k, v


def test_merge_shape_mha():
    """Llama-2 7B: 4096 -> 12288。"""
    q, k, v = make_qkv(32, 32, 128, 4096)
    qkv = merge_qkv(q, k, v, 32, 32, 128)
    assert qkv.shape == (12288, 4096)


def test_merge_shape_gqa():
    q, k, v = make_qkv(32, 8, 128, 4096)
    qkv = merge_qkv(q, k, v, 32, 8, 128)
    assert qkv.shape == ((32 + 16) * 128, 4096)


def test_layout_is_group_interleaved():
    """MHA 下 layout 必须是 [q0,k0,v0,q1,k1,v1,...]，而不是 [q_all,k_all,v_all]。"""
    heads, groups, head_dim, hidden = 4, 4, 8, 16
    q, k, v = make_qkv(heads, groups, head_dim, hidden)
    qkv = merge_qkv(q, k, v, heads, groups, head_dim)

    for h in range(heads):
        base = h * 3 * head_dim
        assert torch.equal(qkv[base : base + head_dim], q[h * head_dim : (h + 1) * head_dim])
        assert torch.equal(
            qkv[base + head_dim : base + 2 * head_dim], k[h * head_dim : (h + 1) * head_dim]
        )
        assert torch.equal(
            qkv[base + 2 * head_dim : base + 3 * head_dim], v[h * head_dim : (h + 1) * head_dim]
        )

    naive = torch.cat([q, k, v], dim=0)
    assert not torch.equal(qkv, naive), "交错 layout 不应该等于朴素 cat"


def test_split_is_inverse_of_merge():
    for heads, groups in [(32, 32), (32, 8), (4, 2)]:
        q, k, v = make_qkv(heads, groups, 8, 16)
        qkv = merge_qkv(q, k, v, heads, groups, 8)
        q2, k2, v2 = split_qkv(qkv, heads, groups, 8)
        assert torch.equal(q, q2) and torch.equal(k, k2) and torch.equal(v, v2)


def test_tp_shard_reconstruction():
    """§24：切开再 gather 回来必须与原始权重逐位相同。"""
    heads, groups, head_dim, hidden = 8, 4, 8, 16
    q, k, v = make_qkv(heads, groups, head_dim, hidden)
    for tp_size in (1, 2, 4):
        shards = [
            shard_qkv(q, k, v, r, tp_size, heads, groups, head_dim) for r in range(tp_size)
        ]
        assert all(s.shape[0] == (heads + 2 * groups) * head_dim // tp_size for s in shards)
        q2, k2, v2 = split_qkv(gather_column(shards), heads, groups, head_dim)
        assert torch.equal(q, q2) and torch.equal(k, k2) and torch.equal(v, v2)


def test_each_rank_gets_whole_groups():
    """TP shard 之后，每个 rank 上的 q/k/v head 数必须成比例。"""
    heads, groups, head_dim, hidden, tp = 8, 4, 8, 16, 2
    q, k, v = make_qkv(heads, groups, head_dim, hidden)
    shard = shard_qkv(q, k, v, 0, tp, heads, groups, head_dim)
    q0, k0, v0 = split_qkv(shard, heads // tp, groups // tp, head_dim)
    assert torch.equal(q0, q[: heads // tp * head_dim])
    assert torch.equal(k0, k[: groups // tp * head_dim])
    assert torch.equal(v0, v[: groups // tp * head_dim])


# ---- tests/test_mlp.py ------------------------------------------------------


def make_gate_up(ffn, hidden):
    g = torch.Generator().manual_seed(1)
    return torch.randn(ffn, hidden, generator=g), torch.randn(ffn, hidden, generator=g)


def test_merge_shape_llama2_7b():
    gate, up = make_gate_up(11008, 4096)
    assert merge_fc1(gate, up).shape == (22016, 4096)


def test_split_is_inverse():
    gate, up = make_gate_up(64, 16)
    g2, u2 = split_fc1(merge_fc1(gate, up))
    assert torch.equal(gate, g2) and torch.equal(up, u2)


def test_shard_keeps_gate_up_pairing_per_rank():
    """每个 rank 上必须是 [gate_shard, up_shard]；这正是"先切再拼"的意义。"""
    ffn, hidden, tp = 64, 16, 4
    gate, up = make_gate_up(ffn, hidden)
    per = ffn // tp
    for r in range(tp):
        shard = shard_fc1(gate, up, r, tp)
        assert shard.shape == (2 * per, hidden)
        g_part, u_part = split_fc1(shard)
        assert torch.equal(g_part, gate[r * per : (r + 1) * per])
        assert torch.equal(u_part, up[r * per : (r + 1) * per])


def test_naive_chunk_of_cat_is_wrong():
    """反例：先 cat 再 chunk，rank0 会整块拿到 gate，SwiGLU 直接算错。"""
    ffn, hidden, tp = 8, 4, 2
    gate, up = make_gate_up(ffn, hidden)
    wrong = torch.chunk(merge_fc1(gate, up), tp, dim=0)[0]
    right = shard_fc1(gate, up, 0, tp)
    assert not torch.equal(wrong, right)
    assert torch.equal(wrong, gate)  # rank0 拿到的全是 gate —— 典型 bug


def test_gather_reconstruction():
    ffn, hidden = 64, 16
    gate, up = make_gate_up(ffn, hidden)
    for tp in (1, 2, 4):
        shards = [shard_fc1(gate, up, r, tp) for r in range(tp)]
        g2, u2 = split_fc1(gather_fc1(shards))
        assert torch.equal(gate, g2) and torch.equal(up, u2)


# ---- tests/test_tp.py -------------------------------------------------------


def test_column_split_shapes_and_reconstruction():
    """linear_qkv [12288, 4096]，TP=2 -> [6144, 4096]。"""
    w = torch.randn(12288, 4096)
    shards = [split_column(w, r, 2) for r in range(2)]
    assert all(s.shape == (6144, 4096) for s in shards)
    assert torch.equal(gather_column(shards), w)


def test_row_split_shapes_and_reconstruction():
    """down_proj [4096, 11008]，TP=2 -> [4096, 5504]。"""
    w = torch.randn(4096, 11008)
    shards = [split_row(w, r, 2) for r in range(2)]
    assert all(s.shape == (4096, 5504) for s in shards)
    assert torch.equal(gather_row(shards), w)


def test_column_and_row_split_along_different_dims():
    w = torch.randn(8, 6)
    assert split_column(w, 0, 2).shape == (4, 6)
    assert split_row(w, 0, 2).shape == (8, 3)


def test_indivisible_raises():
    with pytest.raises(AssertionError):
        split_column(torch.randn(7, 4), 0, 2)
    with pytest.raises(AssertionError):
        split_row(torch.randn(4, 7), 0, 2)


def test_pp_layer_range():
    assert get_pp_layer_range(32, 0, 2) == (0, 16)
    assert get_pp_layer_range(32, 1, 2) == (16, 32)
    assert get_pp_layer_range(32, 3, 4) == (24, 32)
    with pytest.raises(AssertionError):
        get_pp_layer_range(31, 0, 2)


def test_pp_ownership():
    assert is_pre_process(0, 2) and not is_pre_process(1, 2)
    assert is_post_process(1, 2) and not is_post_process(0, 2)
    assert is_pre_process(0, 1) and is_post_process(0, 1)


def test_rank_layout_tp2_pp2():
    """TP 在内、PP 在外：rank = pp_rank * tp_size + tp_rank。"""
    assert rank_of(0, 0, 2) == 0
    assert rank_of(1, 0, 2) == 1
    assert rank_of(0, 1, 2) == 2
    assert rank_of(1, 1, 2) == 3


# ---- tests/test_mapping.py --------------------------------------------------
#
# 端到端 parameter mapping 测试：不需要 GPU、不需要真实权重。
# 思路：造一个假的 HF state_dict，跑完整 mapping，然后把所有 rank 的分片
# 拼回去，逐位比对回原始 HF 权重（note.md §24 的 reconstruction test）。

LLAMA2_7B = LlamaShape(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=32,
    ffn_hidden_size=11008,
    vocab_size=32000,
    max_position_embeddings=4096,
    kv_channels=128,
    rms_norm_eps=1e-5,
    rotary_base=10000.0,
    tie_word_embeddings=False,
)

TOY = LlamaShape(
    num_layers=4,
    hidden_size=16,
    num_attention_heads=4,
    num_query_groups=2,
    ffn_hidden_size=32,
    vocab_size=64,
    max_position_embeddings=64,
    kv_channels=4,
    rms_norm_eps=1e-5,
    rotary_base=10000.0,
    tie_word_embeddings=False,
)


def fake_hf_sd(shape: LlamaShape) -> dict:
    g = torch.Generator().manual_seed(7)

    def r(*sizes):
        return torch.randn(*sizes, generator=g)

    h, hd = shape.hidden_size, shape.kv_channels
    sd = {
        "model.embed_tokens.weight": r(shape.vocab_size, h),
        "model.norm.weight": r(h),
        "lm_head.weight": r(shape.vocab_size, h),
    }
    for i in range(shape.num_layers):
        p = f"model.layers.{i}."
        sd.update(
            {
                p + "input_layernorm.weight": r(h),
                p + "post_attention_layernorm.weight": r(h),
                p + "self_attn.q_proj.weight": r(shape.num_attention_heads * hd, h),
                p + "self_attn.k_proj.weight": r(shape.num_query_groups * hd, h),
                p + "self_attn.v_proj.weight": r(shape.num_query_groups * hd, h),
                p + "self_attn.o_proj.weight": r(h, shape.num_attention_heads * hd),
                p + "mlp.gate_proj.weight": r(shape.ffn_hidden_size, h),
                p + "mlp.up_proj.weight": r(shape.ffn_hidden_size, h),
                p + "mlp.down_proj.weight": r(h, shape.ffn_hidden_size),
            }
        )
    return sd


def test_llama2_7b_expected_shapes_tp1():
    exp = expected_shapes(LLAMA2_7B)
    assert exp["embedding.word_embeddings.weight"] == (32000, 4096)
    assert exp["decoder.layers.0.self_attention.linear_qkv.weight"] == (12288, 4096)
    assert exp["decoder.layers.0.self_attention.linear_proj.weight"] == (4096, 4096)
    assert exp["decoder.layers.0.mlp.linear_fc1.weight"] == (22016, 4096)
    assert exp["decoder.layers.0.mlp.linear_fc2.weight"] == (4096, 11008)
    assert exp["output_layer.weight"] == (32000, 4096)
    assert len([k for k in exp if k.endswith("linear_qkv.weight")]) == 32


def test_llama2_7b_expected_shapes_tp2():
    exp = expected_shapes(LLAMA2_7B, tp_size=2)
    assert exp["decoder.layers.0.self_attention.linear_qkv.weight"] == (6144, 4096)
    assert exp["decoder.layers.0.mlp.linear_fc1.weight"] == (11008, 4096)
    assert exp["decoder.layers.0.mlp.linear_fc2.weight"] == (4096, 5504)
    assert exp["embedding.word_embeddings.weight"] == (16000, 4096)


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 2), (2, 2), (1, 4)])
def test_names_and_shapes_match_expectation(tp, pp):
    hf_sd = fake_hf_sd(TOY)
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            sd = convert_hf_to_megatron(
                hf_sd, TOY, tp_rank=tp_rank, tp_size=tp, pp_rank=pp_rank, pp_size=pp
            )
            exp = expected_shapes(TOY, tp_size=tp, pp_size=pp, pp_rank=pp_rank)
            assert set(sd) == set(exp)
            for k, want in exp.items():
                assert tuple(sd[k].shape) == want, k


def test_tied_embeddings_skips_output_layer():
    shape = dataclass_replace(TOY, tie_word_embeddings=True)
    sd = convert_hf_to_megatron(fake_hf_sd(shape), shape)
    assert "output_layer.weight" not in sd


def test_tp_larger_than_kv_groups_raises():
    with pytest.raises(AssertionError, match="num_query_groups"):
        convert_hf_to_megatron(fake_hf_sd(TOY), TOY, tp_rank=0, tp_size=4)


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_full_roundtrip_back_to_hf(tp, pp):
    """把所有 rank 的分片拼回去，必须逐位等于原始 HF 权重。"""
    shape = TOY
    hf_sd = fake_hf_sd(shape)
    shards = {
        (t, p): convert_hf_to_megatron(
            hf_sd, shape, tp_rank=t, tp_size=tp, pp_rank=p, pp_size=pp
        )
        for p in range(pp)
        for t in range(tp)
    }

    emb = gather_column([shards[(t, 0)]["embedding.word_embeddings.weight"] for t in range(tp)])
    assert torch.equal(emb, hf_sd["model.embed_tokens.weight"])

    head = gather_column(
        [shards[(t, pp - 1)]["output_layer.weight"] for t in range(tp)]
    )
    assert torch.equal(head, hf_sd["lm_head.weight"])
    assert torch.equal(
        shards[(0, pp - 1)]["decoder.final_layernorm.weight"], hf_sd["model.norm.weight"]
    )

    per_stage = shape.num_layers // pp
    for g in range(shape.num_layers):
        p, local = g // per_stage, g % per_stage
        out = f"decoder.layers.{local}."
        src = f"model.layers.{g}."

        qkv = gather_column([shards[(t, p)][out + "self_attention.linear_qkv.weight"] for t in range(tp)])
        # gather 出来的顺序是 rank0 的 group、rank1 的 group…… 正好还是全局 group 顺序
        q, k, v = split_qkv(
            qkv, shape.num_attention_heads, shape.num_query_groups, shape.kv_channels
        )
        assert torch.equal(q, hf_sd[src + "self_attn.q_proj.weight"])
        assert torch.equal(k, hf_sd[src + "self_attn.k_proj.weight"])
        assert torch.equal(v, hf_sd[src + "self_attn.v_proj.weight"])

        fc1 = gather_fc1([shards[(t, p)][out + "mlp.linear_fc1.weight"] for t in range(tp)])
        gate, up = split_fc1(fc1)
        assert torch.equal(gate, hf_sd[src + "mlp.gate_proj.weight"])
        assert torch.equal(up, hf_sd[src + "mlp.up_proj.weight"])

        o = gather_row([shards[(t, p)][out + "self_attention.linear_proj.weight"] for t in range(tp)])
        assert torch.equal(o, hf_sd[src + "self_attn.o_proj.weight"])
        fc2 = gather_row([shards[(t, p)][out + "mlp.linear_fc2.weight"] for t in range(tp)])
        assert torch.equal(fc2, hf_sd[src + "mlp.down_proj.weight"])

        for mg, hf in [
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
        ]:
            assert torch.equal(shards[(0, p)][out + mg], hf_sd[src + hf])


# ---- tests/test_logits_tiny.py ----------------------------------------------
#
# 端到端 GPU 测试：tiny Llama -> checkpoint -> Megatron -> 逐层比对 HF。
# 需要一张 GPU。MHA 和 GQA 两种配置都覆盖。


@pytest.fixture(scope="module")
def dist():
    init_distributed(1, 1)
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 GPU 才能构建 Megatron 模型"
)
@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_tiny_llama_matches_hf(tmp_path_factory, dist, num_kv_heads):
    out = tmp_path_factory.mktemp(f"tiny{num_kv_heads}")
    hf_model = make_tiny_llama(
        str(out),
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,
        intermediate_size=256,
        vocab_size=1000,
    )
    shape = parse_hf_config(hf_model.config)
    sd = convert_hf_to_megatron(hf_model.state_dict(), shape)

    mg_model = build_megatron_llama(shape, dtype=torch.float32, device="cuda")
    missing, unexpected = mg_model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.endswith("_extra_state")]
    assert not [k for k in unexpected if not k.endswith("_extra_state")]

    hf_model = hf_model.cuda().eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, shape.vocab_size, (2, 16), device="cuda")

    ok, report = compare_layerwise(hf_model, mg_model, input_ids, tol=1e-4, verbose=True)
    assert ok, report


def cmd_test(args):
    """用 pytest 收集并运行本文件里的测试函数（等价于原 `pytest tests/`）。"""
    if not hasattr(pytest, "main"):
        print("未安装 pytest，无法运行测试：pip install pytest")
        return 1
    argv = [__file__, "-q"] + list(args.pytest_args)
    return pytest.main(argv)


# =============================================================================
# main：命令行入口
# =============================================================================


def _add_validate_common(sp):
    sp.add_argument("--hf", required=True)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    sp.add_argument("--seq-len", type=int, default=16)
    sp.add_argument("--batch-size", type=int, default=2)
    sp.add_argument("--tol", type=float, default=1e-3)
    sp.add_argument("--seed", type=int, default=42)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="llama_hf_to_megatron_single.py",
        description="HF Llama-2 -> Megatron-Core converter（含转换、验证、测试全部功能）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""典型流程:
  python llama_hf_to_megatron_single.py make-tiny --out /tmp/tiny-llama
  python llama_hf_to_megatron_single.py inspect --model /tmp/tiny-llama
  python llama_hf_to_megatron_single.py convert --hf /tmp/tiny-llama --out /tmp/ckpt_tiny_tp1
  python llama_hf_to_megatron_single.py validate-logits --hf /tmp/tiny-llama --ckpt /tmp/ckpt_tiny_tp1
  torchrun --nproc_per_node=2 llama_hf_to_megatron_single.py validate-tp \\
      --hf /tmp/tiny-llama --ckpt /tmp/ckpt_tiny_tp2
  python llama_hf_to_megatron_single.py validate-pp --hf /tmp/tiny-llama --ckpt /tmp/ckpt_tiny_pp2
  python llama_hf_to_megatron_single.py test
""",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-tiny", help="生成 tiny Llama（HF 格式）")
    p.add_argument("--out", required=True)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_make_tiny)

    p = sub.add_parser("inspect", help="打印 HF config / state_dict / 期望的 Megatron 形状")
    p.add_argument("--model", required=True)
    p.add_argument("--config-only", action="store_true", help="不加载权重，只看 config")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("convert", help="HF -> Megatron checkpoint（TP/PP-aware）")
    p.add_argument("--hf", required=True, help="HF 模型路径或 repo id")
    p.add_argument("--out", required=True, help="输出 checkpoint 目录")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--check-shapes", action="store_true", default=True)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("validate-logits", help="TP=1/PP=1 逐层对比 HF 与 Megatron")
    _add_validate_common(p)
    p.set_defaults(func=cmd_validate_logits)

    p = sub.add_parser("validate-tp", help="TP>1 分片验证（需 torchrun）")
    _add_validate_common(p)
    p.add_argument("--save-dist", default=None, help="额外导出 Megatron DCP 到该目录")
    p.set_defaults(func=cmd_validate_tp)

    p = sub.add_parser("validate-pp", help="PP 分片验证（单进程串行模拟）")
    _add_validate_common(p)
    p.set_defaults(func=cmd_validate_pp)

    p = sub.add_parser("test", help="运行本文件里的单元测试（pytest）")
    p.add_argument("pytest_args", nargs=argparse.REMAINDER,
                   help="透传给 pytest 的参数，例如 -k qkv -s")
    p.set_defaults(func=cmd_test)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
