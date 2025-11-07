# Testing

## Vicuna-7b

将模型合并: model: `MedusaLlamaForCausalLM`

权重路径：如果修改路径要修改 `config.json` 参数`medusa_head_path`,`base_model_name_or_path`

- `model/vicuna-7b-v1.3`: [FasterDecoding/medusa-vicuna-7b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3)
- `model/medusa-vicuna-7b-v1.3`: [lmsys/vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)

1. 运行`python weight_split.py --config_file config/vicuna_7b_config.json `,得到合并的模型参数(vicuna+medusa_head),新的权重在`./temp_vicuna_7b_world_1_rank_0`路径
2. 运行`main_new.py`,默认从`./temp_vicuna_7b_world_1_rank_0` load 模型

```shell
CUDA_VISIBLE_DEVICES=1 python    main_new.py
CUDA_VISIBLE_DEVICES=1 python    main_new.py --load_in_8bit
CUDA_VISIBLE_DEVICES=1 python    main_new.py --load_in_4bit
```

```
MedusaLlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
  (medusa_head): ModuleList(
    (0-4): 5 x Sequential(
      (0): ResBlock(
        (linear): Linear(in_features=4096, out_features=4096, bias=True)
        (act): SiLU()
      )
      (1): Linear(in_features=4096, out_features=32000, bias=False)
    )
  )
)

```

`KV Cache`: 提前分配内存也会影响内存占用和 config 参数`max_kv_cache_length`以及`max_position_embeddings`有关
| para_num|  
| :-------- |
|7477682176|

| data type | model memory | max memory | kv cache mem |
| :-------- | :----------: | :--------: | ------------ |
| bp16      |   14294 MB   |  15516 MB  | 1024 MB      |
| int8      |   7517 MB    |  8774 MB   | 1024 MB      |
| int4      |   4435 MB    |  5819 MB   | 1024 MB      |

### Pipeline

1. 运行`weight_split.py` 划分模型参数,保存到新的文件

```
CUDA_VISIBLE_DEVICES=1 python weight_split.py --config_file  config/vicuna_7b_config_pipe.json
```

2. 运行`pipe_main.py` 从保存路径中读取权重

参数意义:

config_file: `config/vicuna_7b_config_pipe.json`

- 模型参数： `medusa_head_path`, `base_model_name_or_path`
- pipeline 参数: `stage_num_hidden_layers_list`,`num_sub_sequences`,`sub_sequence_length`
- 分布式参数: `init_method`, `distributed_backend`
- KV Cache 参数: `max_kv_cache_length`

```
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 0 --config_file config/vicuna_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 1 --config_file config/vicuna_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 2 --config_file config/vicuna_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 3 --config_file config/vicuna_7b_config_pipe.json

```

修改 config_file 参数 `stage_num_hidden_layers_list` 和 `--world` 参数,支持不同数量 stage
(这样只需要一个 `config.json`文件)

例如：划分 3 个 stage, 修改`stage_num_hidden_layers_list` = [10,10,12]

```
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 0 --config_file config/vicuna_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 1 --config_file config/vicuna_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 3 --rank 2 --config_file config/vicuna_7b_config_pipe.json

```

#### Quantization

**Note**:

- 如果设备选择`cpu`: 半精度数量类型必须为`bfloat16 `
- 如果使用`bitsandbytes`量化，不支持 cpu

```
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 0 --config_file config/vicuna_7b_config_pipe.json  --load-in-8bit
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 1 --config_file config/vicuna_7b_config_pipe.json  --load-in-8bit
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 2 --config_file config/vicuna_7b_config_pipe.json  --load-in-8bit
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 3 --config_file config/vicuna_7b_config_pipe.json  --load-in-8bit

```

[8,8,8,8]
data type: bp16
| | stage 0 <br> (max) | stage 0 <br> (model)| stage 1 <br> (max) | stage 1 <br> (model)| stage 2 <br> (max) | stage 2 <br> (model)| stage 3 <br> (max) | stage 3 <br> (model)|
| :-------- | :-----: | ------: |:-----: | ------: | :-----: | ------: |:-----: | ------: |
| bp16 |3392|3346|3142|3096|3142|3096|4986|4756|
| int8 | 1957|1957|1707|1707|1707|1707|2791|2665|
| int4 |1266|1220|1014|969|1014|969|1923| 1617|

[8,9,9,6]
data type: bp16
| | stage 0 <br> (max) | stage 0 <br> (model)| stage 1 <br> (max) | stage 1 <br> (model)| stage 2 <br> (max) | stage 2 <br> (model)| stage 3 <br> (max) | stage 3 <br> (model)|
| :-------- | :-----: | ------: |:-----: | ------: | :-----: | ------: |:-----: | ------: |
| bp16 |3392|3346|3533|3529|3533|3529|4204|3982|
| int8 | 1957|1957|1904|1904|1904|1904|2388|2270|
| int4 |1266|1220|1129|1080|1129|1080|1694|1395 |

## Vicuna-13b

权重下载: [FasterDecoding/medusa-1.0-vicuna-13b-v1.5](https://huggingface.co/FasterDecoding/medusa-1.0-vicuna-13b-v1.5)

权重路径：`.model/medusa-1.0-vicuna-13b-v1.5` 已经包含 vicuna 和 medusa_head 权重

```
CUDA_VISIBLE_DEVICES=1 python    main.py  --model model/medusa-1.0-vicuna-13b-v1.5
CUDA_VISIBLE_DEVICES=1 python    main.py  --model model/medusa-1.0-vicuna-13b-v1.5  --load-in-8bit
CUDA_VISIBLE_DEVICES=1 python    main.py  --model model/medusa-1.0-vicuna-13b-v1.5  --load-in-4bit
```

```
CUDA_VISIBLE_DEVICES=1 python   main_new.py --config_file config/vicuna_13b_config.json
CUDA_VISIBLE_DEVICES=1 python   main_new.py --config_file config/vicuna_13b_config.json --load_in_8bit
CUDA_VISIBLE_DEVICES=1 python   main_new.py --config_file config/vicuna_13b_config.json --load_in_4bit
```

| para_num    |
| :---------- |
| 13966161920 |

| data type | model memory | max memory | kv cache mem |
| :-------- | :----------: | :--------: | ------------ |
| bp16      |   26678 MB   |  28599 MB  | 1600 MB      |
| int8      |   13901 MB   |  15726 MB  | 1600 MB      |
| int4      |   7168 MB    |  10040 MB  | 1600 MB      |

### pipeline

1. 运行`weight_split.py` 划分模型参数,保存到新的文件

```shell
CUDA_VISIBLE_DEVICES=1  python      weight_split.py  --config_file config/vicuna_13b_config_pipe.json
```

2. 运行`pipe_main.py` 从保存路径中读取权重

```shell
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 0 --config_file config/vicuna_13b_config_pipe.json  --load_in_8bit
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 1 --config_file config/vicuna_13b_config_pipe.json  --load_in_8bit
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 2 --config_file config/vicuna_13b_config_pipe.json  --load_in_8bit
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 3 --config_file config/vicuna_13b_config_pipe.json  --load_in_8bit
```

## zephyr

权重路径: `model/medusa-1.0-zephyr-7b-beta` (包含 zephyr+medusa_head)

```shell
CUDA_VISIBLE_DEVICES=1 python     main_new.py   --config_file config/zephyr_7b_config.json
CUDA_VISIBLE_DEVICES=1 python     main_new.py   --config_file config/zephyr_7b_config.json --load_in_8bit
CUDA_VISIBLE_DEVICES=1 python     main_new.py   --config_file config/zephyr_7b_config.json --load_in_4bit
```

```
MedusaMistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=2)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralAttention(
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm()
        (post_attention_layernorm): MistralRMSNorm()
      )
    )
    (norm): MistralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
  (medusa_head): ModuleList(
    (0-4): 5 x Sequential(
      (0): ResBlock(
        (linear): Linear4bit(in_features=4096, out_features=4096, bias=True)
        (act): SiLU()
      )
      (1): Linear4bit(in_features=4096, out_features=32000, bias=False)
    )
  )
)

```

`torch_dtype=bfloat16`
内存占用:
`KV Cache`: 提前分配内存也会影响内存占用和 config 参数`max_kv_cache_length`以及`max_position_embeddings`有关

| para_num   |
| :--------- |
| 7980998656 |

| data type | model memory | max memory | kv cache mem |
| :-------- | :----------: | :--------: | :----------: |
| bp16      |   15734 MB   |  20037 MB  |   4096 MB    |
| int8      |   8384 MB    |  17640 MB  |   4096 MB    |
| int4      |   5188 MB    |  9649 MB   |   4096 MB    |

### Pipeline

1. 运行`weight_split.py` 划分模型参数,保存到新的文件

```shell
CUDA_VISIBLE_DEVICES=1  python      weight_split.py  --config_file config/zephyr_7b_config_pipe.json
```

2. 运行`pipe_main.py` 从保存路径中读取权重

```shell
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 0  --config_file config/zephyr_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 1  --config_file config/zephyr_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 2  --config_file config/zephyr_7b_config_pipe.json
CUDA_VISIBLE_DEVICES=1 python    pipe_main.py    --world 4 --rank 3  --config_file config/zephyr_7b_config_pipe.json
```

[8,9,9,6]
data type: bp16
| | stage 0 <br> (max) | stage 0 <br> (model)| stage 1 <br> (max) | stage 1 <br> (model)| stage 2 <br> (max) | stage 2 <br> (model)| stage 3 <br> (max) | stage 3 <br> (model)|
| :-------- | :-----: | ------: |:-----: | ------: | :-----: | ------: |:-----: | ------: |
| bp16 |4758|3706|5068|3888|5068|3888|5227|4252|
| int8 | 3148|2211|3344|2185|3344|2185|6934|2473|
| int4 |2470|1432|2484|1318|2484|1318|2583|1562 |
