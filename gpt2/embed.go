/**
 * @file gpt2/embed.go
 * @description Turns token IDs into embedding rows by adding each token's wte row to its position's wpe row.
 * @depends_on safetensors/safetensors.go
 * @depended_on_by main.go, gpt2/embed_test.go, docs/embed.md
 */

// Package gpt2 holds the arithmetic of GPT-2's forward pass, one step per file.
package gpt2

import (
	"fmt"

	"github.com/flcrom/inference-engine/safetensors"
)

// Embed returns one row per token. The row for the token at position i is
// row ids[i] of wte (what the token is) plus row i of wpe (where it sits).
//
// wte and wpe are 2-D tables of shape [rows, columns]. Their column counts
// must match, because the two rows are added number by number.
func Embed(wte, wpe safetensors.Tensor, ids []int) ([][]float32, error) {
	vocab, width, err := table(wte, "wte")
	if err != nil {
		return nil, err
	}
	positions, wpeWidth, err := table(wpe, "wpe")
	if err != nil {
		return nil, err
	}
	if width != wpeWidth {
		return nil, fmt.Errorf("wte rows have %d numbers but wpe rows have %d", width, wpeWidth)
	}
	if len(ids) > positions {
		return nil, fmt.Errorf("%d tokens, but wpe only has rows for %d positions", len(ids), positions)
	}

	embeddings := make([][]float32, 0, len(ids))
	for position, id := range ids {
		if id < 0 || id >= vocab {
			return nil, fmt.Errorf("token ID %d at position %d is outside the vocabulary of %d", id, position, vocab)
		}
		tokenRow := row(wte, id, width)
		positionRow := row(wpe, position, width)

		// A fresh slice: tokenRow and positionRow point into the weights
		// themselves, so writing into them would corrupt the model.
		sum := make([]float32, width)
		for j := range sum {
			sum[j] = tokenRow[j] + positionRow[j]
		}
		embeddings = append(embeddings, sum)
	}
	return embeddings, nil
}

// table checks that t is a 2-D tensor and returns its row and column counts.
func table(t safetensors.Tensor, name string) (rows, columns int, err error) {
	if len(t.Shape) != 2 {
		return 0, 0, fmt.Errorf("%s has shape %v, want [rows, columns]", name, t.Shape)
	}
	return t.Shape[0], t.Shape[1], nil
}

// row returns row r of a flat row-major table whose rows have width numbers.
// Row r starts at index r*width. The result shares memory with t.Values.
func row(t safetensors.Tensor, r, width int) []float32 {
	return t.Values[r*width : (r+1)*width]
}
