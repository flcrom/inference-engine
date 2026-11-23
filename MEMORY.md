<!--
@file MEMORY.md
@description Routes project context: file index, metadata rules, current state, and durable decisions.
@depends_on docs/safetensors.md
@depended_on_by AGENTS.md, README.md, scripts/check_metadata.py
-->

# Project memory

This file is the compact recovery point for humans and AI collaborators. It is not a work log. Source of truth, in order: code and tests, then focused docs, then this file, then chat history.

## Recovering context

Read the file index below. For candidate files, read only the metadata block in their first ten lines and follow the direct edges that matter to the task. Open full files only when they will change or constrain the change. After verifying a change, update metadata on both sides of each changed edge, update this index, and run `python3 scripts/check_metadata.py`.

## File index

| File | Responsibility | Depends on | Depended on by |
| --- | --- | --- | --- |
| `.gitignore` | Keeps weights, build output and local files out of git. | — | — |
| `AGENTS.md` | Working agreement for AI collaborators. | Memory, metadata checker | — |
| `MEMORY.md` | File index, state and durable decisions. | Safetensors doc | Agents, README, metadata checker |
| `README.md` | Human entry point, progress and commands. | Memory, safetensors doc | — |
| `go.mod` | Module path and Go version. | — | `main.go` |
| `main.go` | CLI dispatch and the `inspect` command. | `go.mod`, loader | CLI tests |
| `main_test.go` | Tests inspect output, ordering, truncation and argument errors. | `main.go`, test file builder | — |
| `safetensors/safetensors.go` | Parses and validates safetensors files into float32 tensors. | — | `main.go`, loader tests, safetensors doc |
| `safetensors/safetensors_test.go` | Tests F32/I32 decoding, metadata skipping and malformed input. | Loader, test file builder | — |
| `safetensors/safetensorstest/safetensorstest.go` | Builds small safetensors files for tests. | — | Loader tests, CLI tests |
| `docs/safetensors.md` | Explains the file layout, GPT-2 tensor names and loader checks. | Loader | README, Memory |
| `scripts/check_metadata.py` | Validates metadata headers and reciprocal edges. | Memory | Agents |

## Current state

Step 1 (inspect) is implemented and verified locally with unit tests, `go vet`, and command runs on hand-built F32 and I32 files. It has not yet been run on GPT-2's real 548 MB weights. Learner sign-off for step 1 is pending.

## Durable decisions

- **2026-11-23: Go, standard library only.** Go keeps the focus on inference rather than on learning a new language. No dependencies until a step needs one.
- **2026-11-23: original code only.** The loader was written fresh instead of reusing supplied starter code. It validates offsets and shape-versus-byte counts and returns errors instead of panicking.
- **2026-11-23: flat tensors.** Tensors are `[]float32` in row-major order with a separate shape. Index arithmetic stays visible in the code instead of being hidden in nested slices.
- **2026-11-23: testable commands.** Commands take an `io.Writer` and return errors; only `main` touches `os.Args`, `os.Stdout` and exit codes.
