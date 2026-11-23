<!--
@file AGENTS.md
@description Defines the working agreement for AI collaborators on the inference engine.
@depends_on MEMORY.md, scripts/check_metadata.py
@depended_on_by none
-->

# Working on this repository

Read [MEMORY.md](MEMORY.md) first. Use its file index to load only the files a task needs.

This project has two separate outcomes: an engine whose behavior is verified, and a learner who can explain every line of it. Passing tests never proves the second outcome. AI may write most of the code, but it works in small, inspectable steps. Before a step, explain the concept with a concrete example and a diagram where it helps. After a step, walk through the changed code and record what was verified.

The repository contains only original work. Don't copy third-party starter code or challenge text into it; write our own code and our own explanations. Keep external dependencies at zero unless a step has a concrete need, and explain any dependency before adding it.

Report only checks that actually ran, such as `go test ./...`, `go vet ./...`, command output, or comparisons with reference values. Keep verified behavior, assumptions and open questions separate.

Every new human-authored file needs the four metadata fields (`@file`, `@description`, `@depends_on`, `@depended_on_by`) within its first ten lines, with reciprocal edges. JSON and other strict formats are exempt. Before finishing a change, run:

```sh
gofmt -l . && go vet ./... && go test ./... && python3 scripts/check_metadata.py
```

All commits are authored as `flcrom <161213233+flcrom@users.noreply.github.com>` and carry no co-author trailers.
