/**
 * @file main.go
 * @description Command-line entry point that dispatches engine commands, currently inspect.
 * @depends_on go.mod, safetensors/safetensors.go
 * @depended_on_by main_test.go
 */

// Command inference-engine runs GPT-2 from its raw weights, without an ML
// framework. Usage:
//
//	inference-engine inspect <model.safetensors>
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"

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

	encoded, err := json.Marshal(summaries)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(stdout, string(encoded))
	return err
}
