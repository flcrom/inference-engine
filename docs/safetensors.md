<!--
@file docs/safetensors.md
@description Explains the safetensors file layout, GPT-2's tensor names, and how the loader decodes and validates them.
@depends_on safetensors/safetensors.go
@depended_on_by README.md, MEMORY.md
-->

# Step 1: reading the weights

A trained model is a file of numbers. Before the engine can compute anything, it has to find each table of numbers in that file and read it correctly. This step builds that reader and an `inspect` command that shows what it loaded.

## What the file looks like

`model.safetensors` is one file, not a folder. It has three parts stored back to back:

```text
┌────────────┬──────────────────────────┬─────────────────────────────────────┐
│ 8 bytes    │ N bytes of JSON text     │ raw numbers (≈548 MB for GPT-2)     │
│ N, as a    │ name → dtype, shape,     │ 4 bytes per float32, packed with no │
│ little-    │ data_offsets             │ separators, row after row           │
│ endian u64 │                          │                                     │
└────────────┴──────────────────────────┴─────────────────────────────────────┘
```

For GPT-2, N is 14,283. One entry in the header looks like this:

```json
"wte.weight": {"dtype": "F32", "shape": [50257, 768], "data_offsets": [287209472, 441598976]}
```

The header never contains the numbers. `data_offsets` says where this tensor's bytes start and end, counted from the beginning of the data section. 441,598,976 − 287,209,472 = 154,389,504 bytes, which is exactly 50,257 × 768 numbers × 4 bytes. The header also has one entry that isn't a tensor, `__metadata__` (`{"format": "pt"}` for GPT-2), which the loader skips.

A tensor with shape `[2, 3]` is stored as 6 numbers in a row: the first 3 are row 0 and the next 3 are row 1. This is called row-major order. The loader keeps every tensor as one flat `[]float32` together with its shape, and later steps compute positions from the shape.

## What GPT-2's tensor names mean

The real file has 160 tensors. These names were read from GPT-2's actual header.

| Name | Shape | Meaning |
| --- | --- | --- |
| `wte.weight` | [50257, 768] | Word token embedding: one row of 768 numbers for each token in the vocabulary. |
| `wpe.weight` | [1024, 768] | Word position embedding: one row for each slot in the prompt (0 to 1023). The model's math can't tell word order without this. |
| `h.0.` … `h.11.` | 13 tensors each | The 12 transformer blocks ("hidden" layers). |
| `ln_f.weight`, `ln_f.bias` | [768] | The final layer norm, applied before the model scores the next token. |

Inside each block `h.N`:

| Name | Shape | Meaning |
| --- | --- | --- |
| `ln_1.*`, `ln_2.*` | [768] | Layer norms: rescale each token's 768 numbers before attention and before the MLP. |
| `attn.c_attn.*` | [768, 2304], [2304] | Builds each token's query, key and value vectors (3 × 768). |
| `attn.c_proj.*` | [768, 768], [768] | Combines the attention output back into 768 numbers. |
| `mlp.c_fc.*` | [768, 3072], [3072] | Widens each token's numbers to 3,072. |
| `mlp.c_proj.*` | [3072, 768], [768] | Narrows them back to 768. |
| `attn.bias` | [1, 1, 1024, 1024] | Not learned, despite the name: a triangle of 1s and 0s that stops a token from looking at later tokens. |

So the count is 2 + 12 × 13 + 2 = 160 tensors. A `weight` is a table you multiply by and a `bias` is a list you add afterwards. The `c_` prefix is a leftover from how the original GPT-2 code named its layers.

768 is GPT-2 small's width, a design choice rather than a derived number. The larger GPT-2 models use 1024, 1280 and 1600. 768 also splits evenly into 12 attention heads of 64 numbers each.

## How the loader decodes the numbers

Every supported value is 4 bytes long. The loader reads each group of 4 as a little-endian `uint32` and then:

- **F32:** `math.Float32frombits(bits)` treats the 32 bits as a float. This reinterprets the bits rather than converting a number.
- **I32:** `float32(int32(bits))` treats the bits as a signed integer and then converts its value, so `7` becomes `7.0`.

## What the loader refuses

The loader returns an error, instead of crashing or producing garbage, when:

- the file is shorter than 8 bytes, or N is larger than the rest of the file;
- the header isn't a JSON object, or an entry doesn't fit `{dtype, shape, data_offsets}`;
- the dtype is anything other than F32 or I32;
- the offsets are reversed or point outside the data section;
- the number of bytes doesn't equal the product of the shape × 4.

The last check matters because a shape that doesn't match its bytes would silently corrupt every computation in later steps.

## Memory cost

`Load` reads the whole file into memory and then decodes every tensor into a `[]float32`. For GPT-2, that's about 548 MB of file bytes plus about 500 MB of decoded floats, roughly 1 GB at peak. A later step could reuse the file bytes directly, or map the file into memory (mmap) instead of reading it.
