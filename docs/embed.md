<!--
@file docs/embed.md
@description Explains how token IDs become embedding rows: the wte and wpe tables, row lookup in flat tensors, and the checks Embed makes.
@depends_on gpt2/embed.go
@depended_on_by README.md, MEMORY.md
-->

# Step 2: from token IDs to rows of numbers

A token ID such as `4881` is just a label, like a seat number in a cinema: it tells you which seat but nothing about the person in it. Every later step in GPT-2 does arithmetic, and you can't do useful arithmetic on a label. So the first thing the model does is swap each ID for a row of 768 numbers that it learned during training. Tokens that are used in similar ways end up with similar rows.

## Two tables, two questions

Each row the engine builds answers two questions at once: *which token is this?* and *where does it sit in the prompt?*

- `wte.weight` (word token embedding) has one row per vocabulary entry, so in GPT-2 its shape is `[50257, 768]`. Row `id` answers "which token".
- `wpe.weight` (word position embedding) has one row per position, so in GPT-2 its shape is `[1024, 768]`. Row `i` answers "where". This table is also why GPT-2 can't read more than 1,024 tokens at once.

The embedding of the token at position `i` is the sum, number by number, of row `ids[i]` of `wte` and row `i` of `wpe`. The two indexes come from different places. The **ID** is the *value* stored in `ids.json`, and the **position** is the *index* of that value in the array.

```text
ids.json = [ 1 , 0 ]
  index i:   0   1      <- position: picks the wpe row
  value:     1   0      <- token ID: picks the wte row
```

If the same ID appears twice, both copies use the same `wte` row but different `wpe` rows, so their embeddings differ.

## Finding a row in a flat list

The loader stores every tensor as one flat `[]float32` plus a shape (see [safetensors.md](safetensors.md)). For a shape `[rows, columns]`, the rows are laid out one after another, so row `r` occupies indexes `r*columns` up to, but not including, `(r+1)*columns`:

```text
wte.weight, shape [2, 2]

index    0    1  |  2    3
value    1    2  |  3    4
         row 0   |  row 1       row 1 = Values[1*2 : 2*2] = [3, 4]
```

Each element of `Values` is already one decoded number; the loader turned the file's four bytes per number into a `float32` in step 1. So the stride is `columns`, never `columns * 4`, and no dtype check happens here.

In Go, `Values[a:b]` is a *view* into the same memory, not a copy. `Embed` therefore writes each sum into a fresh slice; writing into the view would quietly change the model's weights.

## Checks

`Embed` returns an error instead of crashing or reading the wrong numbers when:

- either table isn't 2-D;
- the two tables have different row widths, so the rows can't be added;
- an ID is negative or not less than the number of `wte` rows;
- there are more tokens than `wpe` has positions.

`readIDs` decodes `ids.json` into `[]int`, so fractions like `0.5` and anything that isn't an array are rejected. A missing `wte.weight` or `wpe.weight` is reported by name.

## Output

`embed` prints the rows as a single line of JSON, `[[13,24],[31,42]]`. A JSON reader treats `13` and `13.0` as the same number, so the format of the digits doesn't matter. An empty `ids.json` (`[]`) prints `[]`.
