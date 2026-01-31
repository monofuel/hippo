import
  std/unittest,
  hippo

# Concrete types (standard Nim types to match int, long, float, double)
type
  Int = int32 # Equivalent to C int
  Float = float32 # Equivalent to C float
  Long = int64 # Equivalent to C long
  Double = float64 # Equivalent to C double

# Generate macro for Int type with map body
generateForLoopMacro(Int):
  outIt = it * 2 # Map: double

# Generate macro for Float type with map body
generateForLoopMacro(Float):
  outIt = it * 1.5f # Map: multiply by 1.5

# Generate macro for Long type with map body
generateForLoopMacro(Long):
  outIt = it + 1'i64 # Map: add 1

# Generate macro for Double type with map body
generateForLoopMacro(Double):
  outIt = it * it  # Square

suite "GPU macro map operations":
  test "map multiply by 2":
    # This test runs on: HIP, CUDA, HIP_CPU
    # This test skips on: SIMPLE and SIMPLE_NO_THREADS (both thread variants)
    let seq_int: seq[Int] = @[Int(1), Int(2), Int(3)]
    let result_int = customForLoop_Int(seq_int)
    # Verify results: each element should be doubled
    assert result_int.len == 3
    assert result_int[0] == Int(2)
    assert result_int[1] == Int(4)
    assert result_int[2] == Int(6)

  test "map with different values":
    # Test with different input values
    let seq_int: seq[Int] = @[Int(10), Int(0), Int(-5)]
    let result_int = customForLoop_Int(seq_int)
    # Verify results: each element should be doubled
    assert result_int.len == 3
    assert result_int[0] == Int(20)
    assert result_int[1] == Int(0)
    assert result_int[2] == Int(-10)

  test "map Float multiply by 1.5":
    # Test with Float values
    let seq_float: seq[Float] = @[Float(1.0), Float(2.0)]
    let result_float = customForLoop_Float(seq_float)
    # Verify results: each element should be multiplied by 1.5
    assert result_float.len == 2
    assert result_float[0] == Float(1.5)
    assert result_float[1] == Float(3.0)

  test "map Long add 1":
    # Test with Long values
    let seq_long: seq[Long] = @[Long(10), Long(0)]
    let result_long = customForLoop_Long(seq_long)
    # Verify results: each element should be incremented by 1
    assert result_long.len == 2
    assert result_long[0] == Long(11)
    assert result_long[1] == Long(1)

  test "map Double square":
    # Test with Double values
    let seq_double: seq[Double] = @[Double(4.0), Double(5.0)]
    let result_double = customForLoop_Double(seq_double)
    # Verify results: each element should be squared
    assert result_double.len == 2
    assert result_double[0] == Double(16.0)
    assert result_double[1] == Double(25.0)
