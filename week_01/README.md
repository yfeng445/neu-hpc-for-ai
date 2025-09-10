##### Transformer algorithm match functions in run.c:

- `rmsnorm`: normalization, used in many places
- `softmax`: softmax funciton in attention
- `matmul`: matrix multiplication, used in many places also
- `forward`: forward pass, including multihead attention and RoPE position encoding, rmsnorm, and matrix multiplication
- `build_tokenizer`+`encode`: tokenization
- `generate`, `chat`: generation

[Single-threaded matrix multiplication](https://github.com/yfeng445/neu-hpc-for-ai/blob/main/week_01/single-threaded.c)

[Multi threaded matrix multiplication](https://github.com/yfeng445/neu-hpc-for-ai/blob/main/week_01/multi-threaded.c)

Speedup:

- 1 thread: 1.985s
- 2 threads: 1.031s
- 4 threads: 0.409s
- 8 threads: 0.322s
