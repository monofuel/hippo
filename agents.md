# Hippo

- Hippo is a GPU programming library for Nim.

## Organization

- src/hippo.nim is the core of the library. it uses compile time switches depending on the compiler and target.
- src/cuda.nim is for the CUDA nvcc backend of hippo.
- src/hip.nim is for the HIP backend of Hippo. This can be used for ROCM, HIP-CPU, or HIP targetting CUDA.
- src/simple.nim is a threads-only pure nim implementation.  This is useful when translating existing CPU code to GPU code.

- examples/* has example code and configuration for every single supported permutation of compiler and target.
- tests/* default to using the hip-cpu backend.
  - `nimble test` will run the hip-cpu backend tests by default on cpu.
  - `nimble test_amd` will run the tests on AMD with HIP.
  - `nimble test_cuda` will run tests for NVIDIA with CUDA.

## Nix Flakes

- there is a `flake.nix` for setting up the basic dev environment.

nix develop .#basic    # CPU-only Nim development (default)
nix develop .#nvidia   # NVIDIA/CUDA + Nim
nix develop .#amd      # AMD/ROCm + Nim  
nix develop .#all      # Everything combined, useful to run HIP targetting NVIDIA

## Function Names

- all the cuda / hip functions are conditionally included based on the target.
- cuda* cpp functions like cudaMalloc, cudaMemcpy, cudaFree can be used directly for nvidia.
- hip* cpp functions like hipMalloc, hipMemcpy, hipFree can be used directly for AMD.

- src/hippo.nim defines a set of `hippo*` functions that are Nim-friendly and work for any platform. for example, hippoMalloc() returns a struct for a device pointer that will automatically free itself when garbage collected, thus hippoFree() does not have to be called.

## Known backend quirks

- when using hip, hip platform detection ("amd" or "nvidia") may not work as expected if it cannot find a compiler, or if both compilers are present. You can set `HIP_PLATFORM` on the environment to force one or the other at compile time.

- the HIP-CPU backend requires libtbb to be installed, this can be picky on certain distros.
- the GPU compilers (hipcc, nvcc) can sometimes give lots of warnings about the cpp code that Nim produces, but this is mostly OK.
- all Hippo backends require you to compile with `nim cpp`. the only exception is the SIMPLE backend, which can compile with either `nim c` or `nim cpp`. it can even work with threads disabled, in which case it will fall back to single thread execution.

## Nim best practices

**Prefer letting errors bubble up naturally** - Nim's stack traces are excellent for debugging:

Default approach - let operations fail with full context:
```nim
# Simple and clear - if writeFile fails, we get a full stack trace
writeFile(filepath, content)

# Database operations - let them fail with complete error information
db.exec(sql"INSERT INTO users (name) VALUES (?)", username)
```

For validation and early returns, check conditions explicitly:
```nim
# Check preconditions and exit early with clear messages
if not fileExists(parentDir):
  error "Parent directory does not exist"
  quit(1)

if username.len == 0:
  error "Username cannot be empty"
  quit(1)

# Now proceed with the operation
writeFile(filepath, content)
```

This approach ensures full stack traces in CI environments and makes debugging straightforward.


## Tests

- you can use `nim check` to verify files compile, eg: `nim check src/racha_fixer.nim`
  - nim check does not return any text on success.

- you can run unit tests with `nimble test`
- make sure to run unit tests! they should always pass.

- `nimble benchmark` can be used to run benchmark tests.
- benchmark is at ./tests/bench_fraggy.nim

- do not `echo` in tests, they should not log anything when successful

## Gold Master Testing

- some of our tests like tests/test_fragments_gold.nim use a pattern of gold master testing
- they write output to a file in tests/tmp and compare it to a gold master checked into git at tests/gold/
- if the output differs, the test fails.
- Sometimes it is OK if the gold master changes in a way that we expect for a change that we make.
- if we want to update the gold master, you can run the test with `-u` like `nim c -r tests/test_fragments_gold.nim -u` to update the gold master.

## CI

- CI is ran on gitea
- .gitea/workflows/build.yml handles running tests in CI


## Nim

- Nim files should have imports at the top
- types should follow imports
- consts should follow types
- global let and vars should follow consts
- then functions should follow
- main function should be at the bottom of the file.
- comments should be complete sentences that are followed with a period.

### Nim Imports

- std imports should be first, then libraries, and then local imports
- use [] brackets to group when possible
- split imports on newlines
for example,
```
import
  std/[strformat, strutils],
  debby/[pools, postgres],
  ./[models, logs, llm] 
```

### Nim Procs

- do not put comments before functions! comments go inside functions.
- every proc should have a nimdoc comment
- nimdoc comments start with ##
- nimdoc comments should be complete sentences followed by punctuation
for example,
```
proc sumOfMultiples(limit: int): int =
  ## Calculate the sum of all multiples of 3 or 5 below the limit.
  var total = 0
  for i in 1..<limit:
    if i mod 3 == 0 or i mod 5 == 0:
      total += i
  return total
```

### Nim Properties

- if an object property is the same name as a nim keyword, you must wrap it in backticks
```
  DeleteModelResponse* = ref object
    id*: string
    `object`*: string
    deleted*: bool
```

### Variables

- please group const, let, and var variables together.
- please prefer const over let, and let over var.
- please use capitalized camelCase for consts
- use regular camelcase for var and let
- do not place 'magic variables' in the code, instead make them a const and pull them up to the top of the file
- for example:

```
const
  Version = "0.1.0"
  Model = "llama3.2:1b"
let
  embeddingModel = "nomic-embed-text"
```

## Programming

- returning in the middle of files is confusing, avoid doing it.
  - early returns at the start of the file is ok.
- try to make things as idempotent as possible. if a job runs every day, we should make sure it can be robust.
- never use booleans for 'success' or 'error'. If a function was successful, return nothing and do not throw an error. if a function failed, throw an error.