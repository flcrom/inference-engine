/**
 * @file main_test.go
 * @description Verifies the inspect command's JSON output, ordering, truncation, and argument errors.
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

// runInspect writes tensors to a temporary model file and returns the
// decoded output of `inspect` on it.
func runInspect(t *testing.T, tensors []safetensorstest.Tensor) []tensorSummary {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, safetensorstest.Build(t, tensors), 0o644); err != nil {
		t.Fatal(err)
	}
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
	for _, args := range [][]string{{}, {"inspect"}, {"generate", "x"}} {
		if err := run(args, &bytes.Buffer{}); err == nil {
			t.Errorf("run(%q) succeeded, want an error", args)
		}
	}
}
