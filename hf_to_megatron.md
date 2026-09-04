可以。下面我不把它做成“调用 Megatron-Bridge 的教程”，而是做成一个**从零手写 converter 的实验教程**。目标是：你最后能够自己解释并实现 **HF Llama-2 7B → Megatron-Core checkpoint** 的核心过程。

先说明一个现实情况：目前 NVIDIA 官方推荐用 Megatron-Bridge 做 Llama-2 HF → Megatron 转换；旧版 Megatron-LM 也有 `loader_llama_mistral.py` 路径。官方文档明确给出了 Llama-2 7B 的 HF→Megatron 转换流程，而当前 Bridge 的设计核心正是 **config mapping + parameter mapping + TP/PP/EP-aware sharding**。([NVIDIA Docs][1])

但我们这次**故意不用这些 converter**，自己实现一个最小版本。

---

# 1. 我们最终要做什么？

整个教程的目标：

```text
HuggingFace Llama-2 7B
        │
        │
        ▼
┌──────────────────────────┐
│  HF safetensors          │
│                          │
│  embed_tokens            │
│  q_proj / k_proj / v_proj│
│  o_proj                   │
│  gate_proj / up_proj      │
│  down_proj                │
│  layernorm                 │
│  lm_head                   │
└────────────┬─────────────┘
             │
             │ parameter mapping
             ▼
┌──────────────────────────┐
│ Megatron model            │
│                          │
│ word_embeddings           │
│ linear_qkv                │
│ linear_proj               │
│ linear_fc1                │
│ linear_fc2                │
│ input_layernorm            │
│ pre_mlp_layernorm          │
│ output_layer               │
└────────────┬─────────────┘
             │
             │ TP / PP sharding
             ▼
┌──────────────────────────┐
│ Megatron checkpoint       │
│                          │
│ TP rank 0                 │
│ TP rank 1                 │
│ ...                       │
└──────────────────────────┘
```

然后做一个最重要的验证：

```text
HF Llama logits
      ≈
Megatron logits
```

**如果 logits 对不上，converter 就不能算完成。**

---

# 2. 为什么选 Llama-2 7B？

因为它非常适合学习。

Llama-2 7B 的核心结构可以简化成：

```text
Embedding
    │
    ▼
┌───────────────────────┐
│ Transformer Layer ×32 │
│                       │
│ RMSNorm               │
│   ↓                   │
│ GQA Attention         │
│   ↓                   │
│ Residual              │
│   ↓                   │
│ RMSNorm               │
│   ↓                   │
│ SwiGLU MLP            │
│   ↓                   │
│ Residual              │
└───────────────────────┘
    │
    ▼
RMSNorm
    │
    ▼
LM Head
```

典型 Llama-2 7B 配置：

```text
hidden_size              = 4096
num_hidden_layers        = 32
num_attention_heads      = 32
num_key_value_heads      = 32
intermediate_size        = 11008
vocab_size               = 32000
max_position_embeddings  = 4096
```

注意：

**Llama-2 7B 的 GQA 实际上是 MHA，因为 `num_key_value_heads = num_attention_heads = 32`。**

这比 Qwen3 / Llama3 / DeepSeek 更容易作为第一份 converter。

---

# 3. 第一个知识点：HF 的 State Dict

首先不要碰 Megatron。

先把 HF Llama 加载出来：

```python
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype="auto"
)

state_dict = model.state_dict()

for name, tensor in state_dict.items():
    print(name, tensor.shape)
```

你会看到类似：

```text
model.embed_tokens.weight

model.layers.0.input_layernorm.weight

model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight

model.layers.0.post_attention_layernorm.weight

model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight

...

model.norm.weight
lm_head.weight
```

这一步一定要自己运行。

---

# 4. 先做一张 Parameter Mapping 表

这是整个 converter 最核心的东西。

我们先建立：

| HF                                      | Megatron                                  |
| --------------------------------------- | ----------------------------------------- |
| `model.embed_tokens.weight`             | `embedding.word_embeddings.weight`        |
| `model.layers.*.input_layernorm.weight` | `decoder.layers.*.input_layernorm.weight` |
| `q_proj.weight`                         | `linear_qkv.weight`                       |
| `k_proj.weight`                         | `linear_qkv.weight`                       |
| `v_proj.weight`                         | `linear_qkv.weight`                       |
| `o_proj.weight`                         | `linear_proj.weight`                      |
| `post_attention_layernorm.weight`       | `pre_mlp_layernorm.weight`                |
| `gate_proj.weight`                      | `linear_fc1.weight`                       |
| `up_proj.weight`                        | `linear_fc1.weight`                       |
| `down_proj.weight`                      | `linear_fc2.weight`                       |
| `model.norm.weight`                     | `decoder.final_layernorm.weight`          |
| `lm_head.weight`                        | `output_layer.weight`                     |

这里已经出现两个重要问题。

---

# 5. 第一个难点：QKV

HF 是：

```text
q_proj
k_proj
v_proj
```

Megatron 通常使用一个：

```text
linear_qkv
```

所以：

```python
qkv = torch.cat(
    [
        q_proj,
        k_proj,
        v_proj,
    ],
    dim=0
)
```

假设：

```text
hidden = 4096
```

那么：

```text
q_proj = [4096, 4096]
k_proj = [4096, 4096]
v_proj = [4096, 4096]
```

得到：

```text
qkv = [12288, 4096]
```

也就是：

```text
       Q
       │
       │ [4096,4096]
       ▼
┌──────────────┐
│              │
│ Q            │
│ K            │
│ V            │
│              │
└──────────────┘
       ▲
       │
       │ [12288,4096]
```

这就是你的第一个 **many-to-one parameter mapping**。

当前 Megatron Bridge 里面也有专门的 `QKVMapping` 来处理这种情况。([GitHub][2])

---

# 6. 第二个难点：SwiGLU

Llama 的 MLP：

```text
x
│
├──────► gate_proj ────┐
│                      │
│                      × ───► down_proj
└──────► up_proj ──────┘
```

实际上：

```python
y = down_proj(
        silu(gate_proj(x)) *
        up_proj(x)
    )
```

HF 有：

```text
gate_proj
up_proj
down_proj
```

但是 Megatron 通常把：

```text
gate_proj
up_proj
```

合并成：

```text
linear_fc1
```

因此：

```python
fc1 = torch.cat(
    [gate_proj, up_proj],
    dim=0
)
```

假设：

```text
intermediate_size = 11008
hidden_size = 4096
```

那么：

```text
gate_proj = [11008, 4096]
up_proj   = [11008, 4096]
```

于是：

```text
fc1 = [22016, 4096]
```

这就是第二个非常重要的：

**many-to-one mapping。**

Megatron Bridge 对 Llama 的当前实现也使用 `GatedMLPMapping` 处理这一类映射。([GitHub][3])

---

# 7. 到这里先不要做 TP

这是我非常推荐的学习方式。

第一版 converter：

```text
TP = 1
PP = 1
```

即：

```text
HF
 ↓
完整 Megatron model
 ↓
checkpoint
```

先解决：

```text
Architecture Mapping
```

不要同时解决：

```text
Architecture Mapping
+
Tensor Parallel
+
Pipeline Parallel
```

否则 debug 会非常痛苦。

---

# 8. 写我们的第一个 converter

我们先定义一个简单的数据结构：

```python
from dataclasses import dataclass

@dataclass
class HFWeights:
    embed_tokens: torch.Tensor

    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor

    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
```

然后：

```python
def convert_layer(hf_layer):
    qkv = torch.cat([
        hf_layer.self_attn.q_proj.weight,
        hf_layer.self_attn.k_proj.weight,
        hf_layer.self_attn.v_proj.weight,
    ], dim=0)

    fc1 = torch.cat([
        hf_layer.mlp.gate_proj.weight,
        hf_layer.mlp.up_proj.weight,
    ], dim=0)

    return {
        "linear_qkv.weight": qkv,
        "linear_proj.weight":
            hf_layer.self_attn.o_proj.weight,

        "input_layernorm.weight":
            hf_layer.input_layernorm.weight,

        "pre_mlp_layernorm.weight":
            hf_layer.post_attention_layernorm.weight,

        "linear_fc1.weight": fc1,

        "linear_fc2.weight":
            hf_layer.mlp.down_proj.weight,
    }
```

这个函数实际上已经完成了：

```text
HF Layer
   ↓
Megatron Layer
```

的核心转换。

---

# 9. 写完整的 Layer Loop

Llama-2 有 32 层：

```python
for layer_id in range(32):

    hf_layer = model.model.layers[layer_id]

    megatron_layer = convert_layer(hf_layer)

    print(
        layer_id,
        megatron_layer.keys()
    )
```

最终：

```text
layer 0
layer 1
...
layer 31
```

全部转换。

---

# 10. Embedding 和 LM Head

Embedding：

```python
embedding = model.model.embed_tokens.weight
```

直接映射：

```text
HF:

model.embed_tokens.weight

        ↓

Megatron:

embedding.word_embeddings.weight
```

---

LM Head：

```python
lm_head = model.lm_head.weight
```

映射：

```text
HF:

lm_head.weight

        ↓

Megatron:

output_layer.weight
```

Llama-2 的 embedding 和 output layer 在 HF 中是否共享，必须根据具体 checkpoint/config 判断；不要简单假设所有 Llama checkpoint 都是 tied embeddings。

---

# 11. 到这里我们得到一个“逻辑 Megatron State Dict”

例如：

```python
megatron_state_dict = {
    "embedding.word_embeddings.weight":
        embedding,

    "decoder.layers.0.input_layernorm.weight":
        ...,

    "decoder.layers.0.self_attention.linear_qkv.weight":
        ...,

    "decoder.layers.0.self_attention.linear_proj.weight":
        ...,

    "decoder.layers.0.pre_mlp_layernorm.weight":
        ...,

    "decoder.layers.0.mlp.linear_fc1.weight":
        ...,

    "decoder.layers.0.mlp.linear_fc2.weight":
        ...,

    ...
}
```

注意：

**这还不是 Megatron checkpoint。**

现在只是：

> 一个按照 Megatron 参数命名方式组织的完整 state dict。

---

# 12. 第二阶段：加入 Tensor Parallel

现在开始真正有意思的地方。

假设：

```text
TP = 2
```

对于：

```text
ColumnParallelLinear
```

我们沿：

```text
dim = 0
```

切。

例如：

```text
linear_qkv
[12288, 4096]
```

变成：

```text
TP0:
[6144, 4096]

TP1:
[6144, 4096]
```

代码：

```python
def split_column(weight, tp_rank, tp_size):

    chunks = torch.chunk(
        weight,
        tp_size,
        dim=0
    )

    return chunks[tp_rank]
```

---

# 13. 哪些参数是 Column Parallel？

对于 Llama：

```text
QKV projection
FC1
```

通常属于 column-parallel。

因此：

```python
linear_qkv
linear_fc1
```

沿：

```text
dim=0
```

切。

---

# 14. 哪些是 Row Parallel？

Output projection：

```text
linear_proj
```

以及：

```text
fc2
```

通常是 row-parallel。

所以：

```python
def split_row(weight, tp_rank, tp_size):

    chunks = torch.chunk(
        weight,
        tp_size,
        dim=1
    )

    return chunks[tp_rank]
```

例如：

```text
down_proj

[4096, 11008]
```

TP=2：

```text
TP0:
[4096, 5504]

TP1:
[4096, 5504]
```

---

# 15. 为什么一个 dim=0，一个 dim=1？

这是必须真正理解的地方。

Column Parallel：

```text
Y = X W

          W
       ┌─────┐
X ────►│ W0  │──► Y0
       ├─────┤
       │ W1  │──► Y1
       └─────┘
```

把：

```text
W
```

按输出维度切。

所以：

```text
dim=0
```

---

Row Parallel：

```text
X = [X0 X1]

      W0
X0 ───────►
            \
             +──► Y
            /
X1 ───────►
      W1
```

因此：

```text
W
```

按照输入维度切：

```text
dim=1
```

这就是为什么：

```text
Column Parallel → dim 0

Row Parallel    → dim 1
```

---

# 16. 但是 QKV 有一个坑

千万不要简单：

```python
torch.cat([Q, K, V], dim=0)
torch.chunk(qkv, TP)
```

然后就认为结束了。

必须确认 Megatron 的 `linear_qkv` layout。

Llama-2：

```text
Q heads = 32
K heads = 32
V heads = 32
```

TP=2：

```text
16 Q heads
16 K heads
16 V heads
```

因此：

```text
TP0:

Q[0:16]
K[0:16]
V[0:16]

TP1:

Q[16:32]
K[16:32]
V[16:32]
```

也就是说：

```text
Q0..Q15 K0..K15 V0..V15
```

而不是：

```text
Q0..Q31 K0..K31
```

直接按照大块切。

具体 layout 必须与你使用的 Megatron Core attention implementation 对齐；这也是官方 Bridge 为什么提供专门 `QKVMapping` 而不是简单 `torch.chunk()` 的原因。([GitHub][2])

---

# 17. 我建议你自己实现一个 QKV Sharder

```python
def shard_qkv(
    q,
    k,
    v,
    tp_rank,
    tp_size,
):
    q_chunks = torch.chunk(q, tp_size, dim=0)
    k_chunks = torch.chunk(k, tp_size, dim=0)
    v_chunks = torch.chunk(v, tp_size, dim=0)

    return torch.cat(
        [
            q_chunks[tp_rank],
            k_chunks[tp_rank],
            v_chunks[tp_rank],
        ],
        dim=0,
    )
```

这比：

```python
torch.chunk(
    torch.cat([q,k,v], dim=0),
    tp_size
)
```

更能体现你的意图。

---

# 18. FC1 也是一样

Llama：

```text
gate
up
```

先组合：

```text
FC1 = [gate, up]
```

但是 TP 下最好理解成：

```text
gate0
gate1

up0
up1
```

然后：

```text
TP0:

gate0
up0

TP1:

gate1
up1
```

因此：

```python
def shard_fc1(
    gate,
    up,
    tp_rank,
    tp_size,
):

    gate_chunk = torch.chunk(
        gate,
        tp_size,
        dim=0
    )[tp_rank]

    up_chunk = torch.chunk(
        up,
        tp_size,
        dim=0
    )[tp_rank]

    return torch.cat(
        [gate_chunk, up_chunk],
        dim=0
    )
```

这就是你真正应该掌握的 **GatedMLP mapping**。

---

# 19. 最终 TP Converter

我们可以抽象成：

```python
def convert_layer(
    hf_layer,
    tp_rank,
    tp_size,
):

    qkv = shard_qkv(
        hf_layer.self_attn.q_proj.weight,
        hf_layer.self_attn.k_proj.weight,
        hf_layer.self_attn.v_proj.weight,
        tp_rank,
        tp_size,
    )

    fc1 = shard_fc1(
        hf_layer.mlp.gate_proj.weight,
        hf_layer.mlp.up_proj.weight,
        tp_rank,
        tp_size,
    )

    o_proj = split_row(
        hf_layer.self_attn.o_proj.weight,
        tp_rank,
        tp_size,
    )

    fc2 = split_row(
        hf_layer.mlp.down_proj.weight,
        tp_rank,
        tp_size,
    )

    return {
        "linear_qkv.weight": qkv,
        "linear_proj.weight": o_proj,
        "linear_fc1.weight": fc1,
        "linear_fc2.weight": fc2,
    }
```

现在已经开始像真正的 Megatron converter 了。

---

# 20. Pipeline Parallel

假设：

```text
num_layers = 32
PP = 2
```

那么：

```text
PP0:

layer 0
...
layer 15

PP1:

layer 16
...
layer 31
```

简单实现：

```python
def get_pp_range(
    num_layers,
    pp_rank,
    pp_size,
):
    layers_per_stage = num_layers // pp_size

    start = pp_rank * layers_per_stage
    end = start + layers_per_stage

    return start, end
```

然后：

```python
start, end = get_pp_range(
    32,
    pp_rank,
    2
)

for layer_id in range(start, end):
    ...
```

真实 Megatron 还需要处理 embedding、final layernorm、output layer，以及可能的 uneven partition/VPP 等问题，所以这个只是**学习 PP 原理的最小实现**。

---

# 21. 最终 checkpoint rank 是什么？

假设：

```text
TP = 2
PP = 2
```

那么：

```text
world model parallel ranks = 4
```

可以理解成：

```text
             PP0              PP1

TP0       rank 0            rank 2

TP1       rank 1            rank 3
```

每个 rank：

```text
只保存自己的模型 shard
```

所以最终：

```text
checkpoint/
    rank0/
    rank1/
    rank2/
    rank3/
```

这就是 distributed checkpoint 的基本思想。

现代 Megatron Bridge 在转换时也是先创建分布式 Megatron model，再按 TP/PP/EP 等并行方式进行参数分发，而不是简单地生成一个完整 state dict。([GitHub][2])

---

# 22. 但是我们第一版不要自己实现完整 DCP

这里非常重要。

学习 converter 时，我建议：

### Version 1

```text
TP=1
PP=1
```

### Version 2

```text
TP=2
PP=1
```

自己实现 shard。

### Version 3

```text
TP=2
PP=2
```

自己理解 rank mapping。

### Version 4

接 Megatron Core distributed checkpoint API。

否则你会同时掉进：

```text
Transformer
+
TP
+
PP
+
DCP
+
PyTorch distributed
```

五个坑。

---

# 23. 最重要的验证：Tensor Shape

写完 converter 后，第一个测试不是跑模型。

而是：

```python
assert hf_q.shape == (4096, 4096)

assert megatron_qkv.shape == (
    12288,
    4096,
)
```

TP=2：

```python
assert megatron_qkv.shape == (
    6144,
    4096,
)
```

FC1：

```python
assert hf_gate.shape == (
    11008,
    4096,
)

assert fc1.shape == (
    22016,
    4096,
)
```

TP=2：

```python
assert fc1.shape == (
    11008,
    4096,
)
```

---

# 24. 第二个验证：参数重构

这是非常重要的单元测试。

例如：

```python
qkv_tp0
qkv_tp1
```

先：

```python
qkv = torch.cat(
    [qkv_tp0, qkv_tp1],
    dim=0
)
```

然后拆：

```text
Q
K
V
```

最后检查：

```python
torch.equal(
    q,
    hf_q
)
```

同样：

```text
K
V
gate
up
down
o_proj
```

全部检查。

这一步可以在**不启动 Megatron 的情况下**发现 80% 的 converter bug。

---

# 25. 第三个验证：HF forward vs Megatron forward

最终验证：

```text
prompt
  │
  ├── HF Llama
  │
  └── Megatron Llama
          │
          ▼
      logits
```

然后：

```python
max_diff = (
    hf_logits - megatron_logits
).abs().max()

print(max_diff)
```

或者：

```python
relative_error = (
    (hf_logits - megatron_logits).abs()
    /
    hf_logits.abs().clamp_min(1e-8)
).mean()
```

目标：

```text
误差非常小
```

但不要要求：

```text
bitwise identical
```

因为不同 kernel、softmax 实现、BF16/FP16、Transformer Engine 等都可能造成小的数值差异。NVIDIA 官方对 Llama-2 转换后的 benchmark 也报告过小幅数值差异，并指出这是不同实现算术细节导致的。([NVIDIA Docs][1])

---

# 26. 一个非常关键的 Debug 顺序

如果最后：

```text
HF logits != Megatron logits
```

千万不要一上来怀疑 TP。

按照这个顺序：

```text
① Config
       ↓
② Embedding
       ↓
③ LayerNorm
       ↓
④ QKV
       ↓
⑤ RoPE
       ↓
⑥ Attention
       ↓
⑦ O projection
       ↓
⑧ MLP
       ↓
⑨ Final Norm
       ↓
⑩ LM Head
       ↓
⑪ TP
       ↓
⑫ PP
```

最好的办法是做：

```text
layer-by-layer hidden state comparison
```

例如：

```text
HF layer 0 output
        vs
Megatron layer 0 output

HF layer 1 output
        vs
Megatron layer 1 output
```

一旦：

```text
layer 7
```

开始出现巨大差异：

> bug 就在 layer 7 或之前。

---

# 27. 我建议你再做一个“单层实验”

实际上学习 converter 最好的办法不是直接拿 7B。

而是：

```text
Llama-2 architecture
hidden_size = 128
layers = 2
heads = 4
intermediate = 256
vocab = 1000
```

随机初始化。

然后：

```text
HF tiny Llama
       ↓
你的 converter
       ↓
Megatron tiny Llama
```

因为 7B：

```text
7 billion parameters
```

出了问题非常难 debug。

而 tiny model：

```text
几 MB
```

可以随便打印：

```text
Q
K
V
QKV
FC1
FC2
```

---

# 28. 推荐你实际创建这个项目

目录：

```text
llama_hf_to_megatron/
│
├── README.md
│
├── config.py
│
├── hf_loader.py
│
├── mapping.py
│
├── qkv.py
│
├── mlp.py
│
├── tp.py
│
├── pp.py
│
├── checkpoint.py
│
├── validate.py
│
└── tests/
    ├── test_qkv.py
    ├── test_mlp.py
    ├── test_tp.py
    ├── test_mapping.py
    └── test_logits.py
```

其中最重要的是：

```text
mapping.py
qkv.py
mlp.py
tp.py
validate.py
```

---

# 29. 最终代码架构

你的 converter 最终可以长这样：

```python
def convert_hf_to_megatron(
    hf_model,
    tp_size=1,
    pp_size=1,
):

    config = parse_hf_config(
        hf_model.config
    )

    checkpoint = {}

    # Embedding
    checkpoint["embedding"] = \
        convert_embedding(
            hf_model,
            tp_size
        )

    # Transformer layers
    for layer_id in range(
        config.num_layers
    ):

        if not belongs_to_pp(
            layer_id,
            pp_size
        ):
            continue

        layer = hf_model.model.layers[
            layer_id
        ]

        converted = convert_layer(
            layer,
            tp_size=tp_size,
        )

        checkpoint.update(
            converted
        )

    # Final norm
    checkpoint["final_norm"] = \
        hf_model.model.norm.weight

    # LM head
    checkpoint["lm_head"] = \
        hf_model.lm_head.weight

    return checkpoint
```

然后：

```text
convert_hf_to_megatron()
          │
          ├── config
          │
          ├── embedding
          │
          ├── QKV
          │
          ├── MLP
          │
          ├── TP
          │
          ├── PP
          │
          ├── final norm
          │
          └── LM head
```

这就是一个真正的 converter 的骨架。

---

# 30. 学到这里，再去看 Megatron Bridge

这时候你再看 NVIDIA 当前的实现，会突然非常容易理解。

它本质上就是把我们刚才写的：

```text
parse config
      ↓
parameter mapping
      ↓
QKV mapping
      ↓
GatedMLP mapping
      ↓
TP sharding
      ↓
PP handling
      ↓
checkpoint
```

做成了一个通用框架。

比如当前 Llama Bridge 中可以看到：

```text
embedding.word_embeddings.weight
        ↕
model.embed_tokens.weight

output_layer.weight
        ↕
lm_head.weight

linear_proj.weight
        ↕
o_proj.weight

linear_fc2.weight
        ↕
down_proj.weight
```

同时使用 `QKVMapping` 和 `GatedMLPMapping` 处理 QKV / SwiGLU 这类一对多、多对一参数映射。([GitHub][3])

所以你会发现：

> **Megatron Bridge 并没有改变 converter 的本质，只是把 converter 抽象成了一个通用的 Mapping Framework。**

官方“Adding New Models”文档实际上也把新增模型的工作拆成了 **provider/config mapping、parameter mappings、parallelism、model construction、validation** 等步骤。([GitHub][4])

---

# 31. 你应该按照这个顺序真正动手

我建议不要一次做完。

```text
Day 1
│
├── HF Llama state_dict
├── config
└── 参数 shape

Day 2
│
├── QKV mapping
├── SwiGLU mapping
└── Layer mapping

Day 3
│
├── TP
├── Column Parallel
└── Row Parallel

Day 4
│
├── QKV TP
├── GatedMLP TP
└── reconstruction test

Day 5
│
├── PP
├── rank mapping
└── checkpoint layout

Day 6
│
├── Megatron Core
├── distributed checkpoint
└── load checkpoint

Day 7
│
├── HF inference
├── Megatron inference
└── logits / hidden-state validation
```

**最终你要达到的水平不是“会执行 Llama checkpoint conversion”，而是给你一个完全没被 Megatron 支持的新模型，比如：**

```text
MyModelForCausalLM
```

你可以自己从：

```text
HF config
    ↓
Megatron Provider
    ↓
Parameter Mapping
    ↓
QKV Mapping
    ↓
MLP Mapping
    ↓
TP/PP/EP
    ↓
Checkpoint
    ↓
Numerical Validation
```

完整写出 converter。

这实际上就是你后面研究 **Qwen3 → Megatron、DeepSeek-MoE → Megatron、VLM → Megatron** 时最需要的能力。

**下一步最值得做的是直接进入“代码实战 Part 1”**：我可以给你一个**完全可运行的 tiny-Llama（hidden=128、2 layers）项目**，从 `transformers` 创建 HF 模型开始，自己实现 `QKVMapping + SwiGLU Mapping + TP=2`，然后逐层验证 HF 和 Megatron 的 hidden states。这样你可以直接在自己的环境里跑，而不是只看伪代码。

[1]: https://docs.nvidia.com/megatron-core/developer-guide/latest/llama_mistral.html?utm_source=chatgpt.com "Llama, Mistral and other Llama-like model support in Megatron-LM — Megatron Core"
[2]: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md?utm_source=chatgpt.com "Megatron-Bridge/docs/bridge-tech-details.md at main · NVIDIA-NeMo/Megatron-Bridge · GitHub"
[3]: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py?utm_source=chatgpt.com "Megatron-Bridge/src/megatron/bridge/models/llama/llama_bridge.py at main · NVIDIA-NeMo/Megatron-Bridge · GitHub"
[4]: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/adding-new-models.md?utm_source=chatgpt.com "Megatron-Bridge/docs/adding-new-models.md at main · NVIDIA-NeMo/Megatron-Bridge · GitHub"
