<!--
@file README.md
@description Human entry point describing the engine's goal, current progress, commands, and repository layout.
@depends_on MEMORY.md, docs/safetensors.md, docs/embed.md
@depended_on_by none
-->

# inference-engine

A GPT-2 inference engine written from scratch in Go, using only the standard library and no machine-learning framework. It loads GPT-2's real weights and does the arithmetic that turns a prompt's token IDs into the next token: embeddings, attention, transformer blocks, logits, then greedy decoding or sampling, with a key-value cache. The goal is to produce exactly the tokens GPT-2 itself produces.

## Progress

| Step | What it does | Status |
| --- | --- | --- |
| Inspect | Read a safetensors file; print each tensor's name, dtype, shape and first values | Done |
| Embed | Turn token IDs into rows of numbers | Done |
| Attend | Causal and multi-head attention, residuals, the transformer block | — |
| Score | Logits for every vocabulary token | — |
| Pick | Greedy decoding and temperature/top-p sampling | — |
| Repeat | Key-value cache | — |
| Run | All of it on GPT-2's real weights | — |

## Usage

This needs Go 1.26 or newer. Older Go versions download it automatically.

```sh
go run . inspect path/to/model.safetensors
go run . embed path/to/model.safetensors ids.json
```

`inspect` prints a JSON array, sorted by name, with one entry per tensor:

```json
[{"name":"scale","dtype":"F32","shape":[4],"first":[1,0.5,0.25,0.125]}]
```

`embed` reads a JSON array of token IDs, such as `[464, 3139, 286, 4881, 318]` for "The capital of France is". It prints one row per token: that token's `wte.weight` row plus its position's `wpe.weight` row.

```json
[[13,24],[31,42]]
```

To try it on GPT-2's real weights, download them somewhere outside this repository (548 MB):

```sh
curl -L -o /tmp/model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
go run . inspect /tmp/model.safetensors
```

Run the tests with:

```sh
go test ./...
```

## Layout

| Path | Purpose |
| --- | --- |
| `main.go` | Command-line entry point and the `inspect` and `embed` commands |
| `gpt2/` | The forward pass's arithmetic, one step per file |
| `safetensors/` | Weight-file loader with input validation |
| `safetensors/safetensorstest/` | Builds small weight files for tests |
| `docs/` | One explanation per step, including diagrams and the reasoning behind decisions |
| `MEMORY.md`, `AGENTS.md` | File index and working agreement for human and AI collaborators |

Start with [docs/safetensors.md](docs/safetensors.md) to see what the weight file contains, then [docs/embed.md](docs/embed.md).
