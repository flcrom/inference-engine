/**
 * @file safetensors/safetensors.go
 * @description Parses a safetensors file into named tensors of float32 values, rejecting malformed headers and offsets.
 * @depends_on none
 * @depended_on_by main.go, safetensors/safetensors_test.go, docs/safetensors.md
 */

// Package safetensors reads model weights stored in the safetensors format.
//
// A safetensors file has three parts, back to back:
//
//	[8 bytes]  little-endian uint64 N, the length of the JSON header
//	[N bytes]  JSON header: tensor name -> {dtype, shape, data_offsets}
//	[rest]     raw tensor bytes; data_offsets are relative to this section
//
// The header describes the tensors; it never contains the numbers themselves.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
)

// Tensor is one named table of numbers from the file.
// Values are stored flat in row-major order: for shape [2, 3], the first
// three values are row 0 and the next three are row 1.
type Tensor struct {
	DType  string
	Shape  []int
	Values []float32
}

// headerEntry mirrors one tensor's description in the JSON header.
// The struct tags tell encoding/json which JSON key fills which field.
type headerEntry struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// metadataKey names the optional header entry that holds free-form strings
// (for GPT-2: {"format": "pt"}) instead of a tensor description.
const metadataKey = "__metadata__"

// bytesPerElement lists the dtypes this engine understands. Both are 4 bytes
// wide; I32 values are converted to float32 so every tensor has one Go type.
var bytesPerElement = map[string]int{
	"F32": 4,
	"I32": 4,
}

// Load reads the file at path and parses every tensor in it.
func Load(path string) (map[string]Tensor, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse decodes a complete safetensors file that is already in memory.
// It is separate from Load so tests can build files without touching disk.
func Parse(raw []byte) (map[string]Tensor, error) {
	if len(raw) < 8 {
		return nil, errors.New("file is shorter than the 8-byte header length")
	}
	headerLen := binary.LittleEndian.Uint64(raw[:8])
	// Compare as uint64 first: converting a huge headerLen to int could overflow.
	if headerLen > uint64(len(raw)-8) {
		return nil, fmt.Errorf("header length %d is larger than the %d bytes after it", headerLen, len(raw)-8)
	}
	headerEnd := 8 + int(headerLen)
	headerBytes := raw[8:headerEnd]
	data := raw[headerEnd:]

	// Decode in two steps. First, split the header into name -> undecoded JSON,
	// because __metadata__ has a different shape from the tensor entries.
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("header is not a JSON object: %w", err)
	}

	tensors := make(map[string]Tensor, len(header))
	for name, rawEntry := range header {
		if name == metadataKey {
			continue
		}
		// Second step: decode each tensor's entry into the typed struct.
		var entry headerEntry
		if err := json.Unmarshal(rawEntry, &entry); err != nil {
			return nil, fmt.Errorf("tensor %q: invalid header entry: %w", name, err)
		}
		tensor, err := decodeTensor(entry, data)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		tensors[name] = tensor
	}
	return tensors, nil
}

// decodeTensor checks one entry against the data section and converts its
// bytes into float32 values.
func decodeTensor(entry headerEntry, data []byte) (Tensor, error) {
	width, ok := bytesPerElement[entry.DType]
	if !ok {
		return Tensor{}, fmt.Errorf("unsupported dtype %q", entry.DType)
	}

	// The element count is the product of the dimensions. An empty shape is
	// a scalar: one number.
	count := 1
	for _, dim := range entry.Shape {
		if dim < 0 {
			return Tensor{}, fmt.Errorf("negative dimension in shape %v", entry.Shape)
		}
		count *= dim
	}

	start, end := entry.DataOffsets[0], entry.DataOffsets[1]
	if start < 0 || start > end || end > len(data) {
		return Tensor{}, fmt.Errorf("data offsets [%d, %d] fall outside the %d-byte data section", start, end, len(data))
	}
	if end-start != count*width {
		return Tensor{}, fmt.Errorf("shape %v needs %d bytes but offsets cover %d", entry.Shape, count*width, end-start)
	}

	raw := data[start:end]
	// make (not a nil slice) so an empty tensor still encodes as [] in JSON.
	values := make([]float32, count)
	for i := range values {
		bits := binary.LittleEndian.Uint32(raw[i*width:])
		switch entry.DType {
		case "F32":
			// Reinterpret the 32 bits as an IEEE-754 float; no numeric conversion.
			values[i] = math.Float32frombits(bits)
		case "I32":
			// Reinterpret as a signed integer, then convert its value: 7 -> 7.0.
			values[i] = float32(int32(bits))
		}
	}
	return Tensor{DType: entry.DType, Shape: entry.Shape, Values: values}, nil
}
