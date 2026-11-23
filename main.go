/**
 * @file main.go
 * @description Command-line entry point that dispatches engine commands: inspect and embed.
 * @depends_on go.mod, safetensors/safetensors.go, gpt2/embed.go
 * @depended_on_by main_test.go
 */

// Command inference-engine runs GPT-2 from its raw weights, without an ML
// framework. Usage:
//
//	inference-engine inspect <model.safetensors>
//	inference-engine embed <model.safetensors> <ids.json>
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/flcrom/inference-engine/gpt2"
	"github.com/flcrom/inference-engine/safetensors"
)

func main() {
	// Results go to stdout; errors go to stderr, so piping the output into a
	// JSON tool never mixes in an error message.
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run dispatches one command. It takes args and an output writer instead of
// reading os.Args and os.Stdout directly, so tests can call it.
func run(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: inference-engine <command> [args...]")
	}
	switch args[0] {
	case "inspect":
		if len(args) != 2 {
			return errors.New("usage: inference-engine inspect <model.safetensors>")
		}
		return inspect(args[1], stdout)
	case "embed":
		if len(args) != 3 {
			return errors.New("usage: inference-engine embed <model.safetensors> <ids.json>")
		}
		return embed(args[1], args[2], stdout)
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

// tensorSummary is one entry of inspect's output. The struct tags set the
// lowercase JSON key names.
type tensorSummary struct {
	Name  string    `json:"name"`
	DType string    `json:"dtype"`
	Shape []int     `json:"shape"`
	First []float32 `json:"first"`
}

// inspect prints a JSON array describing every tensor in the file, sorted by
// name, with each tensor's first four values (or all of them, if fewer).
func inspect(path string, stdout io.Writer) error {
	tensors, err := safetensors.Load(path)
	if err != nil {
		return err
	}

	// Go map iteration order is deliberately random, so sort the names to
	// make the output deterministic.
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	summaries := make([]tensorSummary, 0, len(names))
	for _, name := range names {
		tensor := tensors[name]
		first := tensor.Values[:min(4, len(tensor.Values))]
		summaries = append(summaries, tensorSummary{name, tensor.DType, tensor.Shape, first})
	}

	return printJSON(stdout, summaries)
}

// embed prints the embedding of every token ID in idsPath, in order, as a
// JSON array of rows.
func embed(modelPath, idsPath string, stdout io.Writer) error {
	tensors, err := safetensors.Load(modelPath)
	if err != nil {
		return err
	}
	ids, err := readIDs(idsPath)
	if err != nil {
		return err
	}
	wte, err := lookup(tensors, "wte.weight")
	if err != nil {
		return err
	}
	wpe, err := lookup(tensors, "wpe.weight")
	if err != nil {
		return err
	}
	embeddings, err := gpt2.Embed(wte, wpe, ids)
	if err != nil {
		return err
	}
	return printJSON(stdout, embeddings)
}

// readIDs decodes a JSON array of whole numbers, such as [464, 3139].
// Decoding into []int rejects fractions and anything that is not an array.
func readIDs(path string) ([]int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var ids []int
	if err := json.Unmarshal(raw, &ids); err != nil {
		return nil, fmt.Errorf("%s is not a JSON array of token IDs: %w", path, err)
	}
	return ids, nil
}

// lookup returns the named tensor, or an error naming what is missing.
func lookup(tensors map[string]safetensors.Tensor, name string) (safetensors.Tensor, error) {
	tensor, ok := tensors[name]
	if !ok {
		return safetensors.Tensor{}, fmt.Errorf("model has no tensor named %q", name)
	}
	return tensor, nil
}

// printJSON writes v as one line of JSON.
func printJSON(stdout io.Writer, v any) error {
	encoded, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(stdout, string(encoded))
	return err
}
