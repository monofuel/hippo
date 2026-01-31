#!/bin/bash

# monofuel note: my desktop and main server only have AMD GPUs
# Script to run a specific test file for all backends except CUDA
# Usage: ./run_test_backends.sh <test_file>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <test_file>"
    echo "Example: $0 tests/test_gpu_macro.nim"
    exit 1
fi

TEST_FILE="$1"

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file '$TEST_FILE' not found"
    exit 1
fi

echo "Running $TEST_FILE for all backends (excluding CUDA)..."
echo "=================================================="

# AMD HIP Backend
echo ""
echo "🔥 Testing AMD HIP Backend..."
echo "Command: nim cpp --cc:hipcc --define:'useMalloc' --define:'HippoRuntime:HIP' -r $TEST_FILE"
if nim cpp --cc:hipcc --define:'useMalloc' --define:'HippoRuntime:HIP' -r "$TEST_FILE"; then
    echo "✅ AMD HIP backend: PASSED"
else
    echo "❌ AMD HIP backend: FAILED"
fi

# HIP-CPU Backend
echo ""
echo "🖥️  Testing HIP-CPU Backend..."
echo "Command: nim cpp --define:'HippoRuntime:HIP_CPU' -r $TEST_FILE"
if nim cpp --define:'HippoRuntime:HIP_CPU' -r "$TEST_FILE"; then
    echo "✅ HIP-CPU backend: PASSED"
else
    echo "❌ HIP-CPU backend: FAILED"
fi

# Simple Backend (with threads)
echo ""
echo "🧵 Testing Simple Backend (with threads)..."
echo "Command: nim cpp --threads:on --define:'HippoRuntime:SIMPLE' -r $TEST_FILE"
if nim cpp --threads:on --define:'HippoRuntime:SIMPLE' -r "$TEST_FILE"; then
    echo "✅ Simple backend (threads): PASSED"
else
    echo "❌ Simple backend (threads): FAILED"
fi

# Simple Backend (without threads)
echo ""
echo "🔄 Testing Simple Backend (without threads)..."
echo "Command: nim cpp --threads:off --define:'HippoRuntime:SIMPLE' -r $TEST_FILE"
if nim cpp --threads:off --define:'HippoRuntime:SIMPLE' -r "$TEST_FILE"; then
    echo "✅ Simple backend (no threads): PASSED"
else
    echo "❌ Simple backend (no threads): FAILED"
fi

echo ""
echo "=================================================="
echo "Test run completed for all backends (excluding CUDA)"
