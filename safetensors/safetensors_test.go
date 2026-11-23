/**
 * @file safetensors/safetensors_test.go
 * @description Verifies safetensors parsing for F32 and I32 tensors, metadata skipping, and malformed-file rejection.
 * @depends_on safetensors/safetensors.go, safetensors/safetensorstest/safetensorstest.go
 * @depended_on_by none
 */

package safetensors_test

import (
	"encoding/binary"
	"reflect"
	"strings"
	"testing"

	"github.com/flcrom/inference-engine/safetensors"
	"github.com/flcrom/inference-engine/safetensors/safetensorstest"
)

func TestParseReadsF32AndI32(t *testing.T) {
	file := safetensorstest.Build(t, []safetensorstest.Tensor{
		{Name: "scale", DType: "F32", Shape: []int{4}, Values: []float64{1, 0.5, 0.25, 0.125}},
		{Name: "ids", DType: "I32", Shape: []int{2, 2}, Values: []float64{7, -3, 0, 50256}},
	})

	tensors, err := safetensors.Parse(file)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(tensors) != 2 {
		t.Fatalf("got %d tensors, want 2 (metadata must be skipped)", len(tensors))
	}

	scale := tensors["scale"]
	if scale.DType != "F32" || !reflect.DeepEqual(scale.Shape, []int{4}) ||
		!reflect.DeepEqual(scale.Values, []float32{1, 0.5, 0.25, 0.125}) {
		t.Errorf("scale = %+v", scale)
	}
	ids := tensors["ids"]
	if ids.DType != "I32" || !reflect.DeepEqual(ids.Values, []float32{7, -3, 0, 50256}) {
		t.Errorf("ids = %+v", ids)
	}
}

func TestParseRejectsMalformedFiles(t *testing.T) {
	valid := safetensorstest.Build(t, []safetensorstest.Tensor{
		{Name: "scale", DType: "F32", Shape: []int{2}, Values: []float64{1, 2}},
	})

	// withHeader wraps a hand-written header with 8 bytes of data.
	withHeader := func(header string) []byte {
		file := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
		return append(append(file, header...), make([]byte, 8)...)
	}

	cases := map[string]struct {
		file []byte
		want string
	}{
		"too short":         {[]byte{1, 2, 3}, "shorter than"},
		"header too long":   {binary.LittleEndian.AppendUint64(nil, 1<<40), "larger than"},
		"truncated data":    {valid[:len(valid)-1], "outside"},
		"header not json":   {withHeader("nope"), "not a JSON object"},
		"unsupported dtype": {withHeader(`{"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}`), "unsupported dtype"},
		"shape mismatch":    {withHeader(`{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}`), "needs 12 bytes"},
		"reversed offsets":  {withHeader(`{"x":{"dtype":"F32","shape":[0],"data_offsets":[8,0]}}`), "outside"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := safetensors.Parse(tc.file)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}
