/**
 * @file safetensors/safetensorstest/safetensorstest.go
 * @description Builds small, valid safetensors files in memory for tests in any package.
 * @depends_on none
 * @depended_on_by safetensors/safetensors_test.go, main_test.go
 */

// Package safetensorstest helps tests create safetensors files without
// downloading real model weights, in the same spirit as net/http/httptest.
package safetensorstest

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

// Tensor describes one tensor to write. Values become F32 bits or I32
// integers depending on DType.
type Tensor struct {
	Name   string
	DType  string
	Shape  []int
	Values []float64
}

// Build returns the bytes of a safetensors file holding tensors, plus a
// __metadata__ entry so readers are forced to skip it.
func Build(t testing.TB, tensors []Tensor) []byte {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "test"}}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		for _, v := range tensor.Values {
			var bits uint32
			if tensor.DType == "I32" {
				bits = uint32(int32(v))
			} else {
				bits = math.Float32bits(float32(v))
			}
			data = binary.LittleEndian.AppendUint32(data, bits)
		}
		header[tensor.Name] = map[string]any{
			"dtype":        tensor.DType,
			"shape":        tensor.Shape,
			"data_offsets": []int{start, len(data)},
		}
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(headerBytes)))
	file = append(file, headerBytes...)
	return append(file, data...)
}
