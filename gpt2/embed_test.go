/**
 * @file gpt2/embed_test.go
 * @description Verifies embedding lookup, position addition, repeated IDs, and rejection of bad shapes and IDs.
 * @depends_on gpt2/embed.go
 * @depended_on_by none
 */

package gpt2

import (
	"reflect"
	"testing"

	"github.com/flcrom/inference-engine/safetensors"
)

func tensor(shape []int, values ...float32) safetensors.Tensor {
	return safetensors.Tensor{DType: "F32", Shape: shape, Values: values}
}

func TestEmbedAddsTokenAndPositionRows(t *testing.T) {
	wte := tensor([]int{2, 2}, 1, 2, 3, 4)
	wpe := tensor([]int{2, 2}, 10, 20, 30, 40)

	got, err := Embed(wte, wpe, []int{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	// Position 0 holds ID 1: [3, 4] + [10, 20]. Position 1 holds ID 0: [1, 2] + [30, 40].
	want := [][]float32{{13, 24}, {31, 42}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestEmbedRepeatedIDDiffersByPosition(t *testing.T) {
	wte := tensor([]int{3, 4}, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2)
	wpe := tensor([]int{3, 4}, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0)

	got, err := Embed(wte, wpe, []int{2, 0, 2})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{2.5, 2, 2, 2}, {0, 0.5, 0, 0}, {2, 2, 2.5, 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	// The weights must be untouched after embedding.
	if wte.Values[8] != 2 || wpe.Values[0] != 0.5 {
		t.Fatalf("Embed modified the weights: wte=%v wpe=%v", wte.Values, wpe.Values)
	}
}

func TestEmbedNoTokens(t *testing.T) {
	got, err := Embed(tensor([]int{1, 1}, 1), tensor([]int{1, 1}, 1), nil)
	if err != nil || got == nil || len(got) != 0 {
		t.Fatalf("got %v, %v; want an empty, non-nil result", got, err)
	}
}

func TestEmbedRejectsBadInput(t *testing.T) {
	square := tensor([]int{2, 2}, 1, 2, 3, 4)
	cases := []struct {
		name     string
		wte, wpe safetensors.Tensor
		ids      []int
	}{
		{"ID past the vocabulary", square, square, []int{2}},
		{"negative ID", square, square, []int{-1}},
		{"more tokens than positions", square, square, []int{0, 1, 0}},
		{"column counts differ", square, tensor([]int{1, 4}, 1, 2, 3, 4), []int{0}},
		{"wte not 2-D", tensor([]int{4}, 1, 2, 3, 4), square, []int{0}},
	}
	for _, c := range cases {
		if _, err := Embed(c.wte, c.wpe, c.ids); err == nil {
			t.Errorf("%s: got no error", c.name)
		}
	}
}
