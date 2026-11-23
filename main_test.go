/**
 * @file main_test.go
 * @description Verifies the inspect and embed commands' JSON output and their argument and input errors.
 * @depends_on main.go, safetensors/safetensorstest/safetensorstest.go
 * @depended_on_by none
 */

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/flcrom/inference-engine/safetensors/safetensorstest"
)

// writeModel saves tensors as a temporary model file and returns its path.
func writeModel(t *testing.T, tensors []safetensorstest.Tensor) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, safetensorstest.Build(t, tensors), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// writeFile saves contents as a temporary file and returns its path.
func writeFile(t *testing.T, name, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// runInspect writes tensors to a temporary model file and returns the
// decoded output of `inspect` on it.
func runInspect(t *testing.T, tensors []safetensorstest.Tensor) []tensorSummary {
	t.Helper()
	path := writeModel(t, tensors)
	var out bytes.Buffer
	if err := run([]string{"inspect", path}, &out); err != nil {
		t.Fatalf("inspect: %v", err)
	}
	var summaries []tensorSummary
	if err := json.Unmarshal(out.Bytes(), &summaries); err != nil {
		t.Fatalf("output is not JSON: %v\n%s", err, out.String())
	}
	return summaries
}

func TestInspectSingleTensor(t *testing.T) {
	got := runInspect(t, []safetensorstest.Tensor{
		{Name: "scale", DType: "F32", Shape: []int{4}, Values: []float64{1, 0.5, 0.25, 0.125}},
	})
	want := []tensorSummary{{"scale", "F32", []int{4}, []float32{1, 0.5, 0.25, 0.125}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestInspectSortsAndTruncates(t *testing.T) {
	got := runInspect(t, []safetensorstest.Tensor{
		{Name: "zeta", DType: "F32", Shape: []int{2, 3}, Values: []float64{1, 2, 3, 4, 5, 6}},
		{Name: "alpha", DType: "I32", Shape: []int{2}, Values: []float64{-1, 9}},
	})
	want := []tensorSummary{
		{"alpha", "I32", []int{2}, []float32{-1, 9}},
		{"zeta", "F32", []int{2, 3}, []float32{1, 2, 3, 4}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestRunRejectsBadArguments(t *testing.T) {
	for _, args := range [][]string{{}, {"inspect"}, {"embed", "model"}, {"generate", "x"}} {
		if err := run(args, &bytes.Buffer{}); err == nil {
			t.Errorf("run(%q) succeeded, want an error", args)
		}
	}
}

// embedModel holds the two 2x2 tables used by the embed tests.
var embedModel = []safetensorstest.Tensor{
	{Name: "wte.weight", DType: "F32", Shape: []int{2, 2}, Values: []float64{1, 2, 3, 4}},
	{Name: "wpe.weight", DType: "F32", Shape: []int{2, 2}, Values: []float64{10, 20, 30, 40}},
}

func TestEmbedCommand(t *testing.T) {
	model := writeModel(t, embedModel)
	ids := writeFile(t, "ids.json", "[1, 0]")

	var out bytes.Buffer
	if err := run([]string{"embed", model, ids}, &out); err != nil {
		t.Fatalf("embed: %v", err)
	}
	var got [][]float64
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("output is not JSON: %v\n%s", err, out.String())
	}
	want := [][]float64{{13, 24}, {31, 42}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestEmbedCommandRejectsBadInput(t *testing.T) {
	model := writeModel(t, embedModel)
	noWPE := writeModel(t, embedModel[:1])
	cases := []struct{ name, model, ids string }{
		{"fractional ID", model, "[0.5]"},
		{"not an array", model, `{"ids": [0]}`},
		{"ID outside vocabulary", model, "[2]"},
		{"missing wpe.weight", noWPE, "[0]"},
	}
	for _, c := range cases {
		ids := writeFile(t, "ids.json", c.ids)
		if err := run([]string{"embed", c.model, ids}, &bytes.Buffer{}); err == nil {
			t.Errorf("%s: got no error", c.name)
		}
	}
}
