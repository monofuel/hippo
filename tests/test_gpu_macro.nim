import
  std/unittest,
  hippo

generateForLoopMacro("mapInt32", int32):
  outIt = it * 2

generateForLoopMacro("mapFloat32", float32):
  outIt = it * 1.5f

generateForLoopMacro("mapInt64", int64):
  outIt = it + 1'i64

generateForLoopMacro("mapFloat64", float64):
  outIt = it * it

suite "GPU macro map operations":
  test "map multiply by 2":
    let seq_int: seq[int32] = @[int32(1), int32(2), int32(3)]
    let result_int = mapInt32(seq_int)
    assert result_int.len == 3
    assert result_int[0] == int32(2)
    assert result_int[1] == int32(4)
    assert result_int[2] == int32(6)

  test "map with different values":
    let seq_int: seq[int32] = @[int32(10), int32(0), int32(-5)]
    let result_int = mapInt32(seq_int)
    assert result_int.len == 3
    assert result_int[0] == int32(20)
    assert result_int[1] == int32(0)
    assert result_int[2] == int32(-10)

  test "map Float multiply by 1.5":
    let seq_float: seq[float32] = @[float32(1.0), float32(2.0)]
    let result_float = mapFloat32(seq_float)
    assert result_float.len == 2
    assert result_float[0] == float32(1.5)
    assert result_float[1] == float32(3.0)

  test "map Long add 1":
    let seq_long: seq[int64] = @[int64(10), int64(0)]
    let result_long = mapInt64(seq_long)
    assert result_long.len == 2
    assert result_long[0] == int64(11)
    assert result_long[1] == int64(1)

  test "map Double square":
    let seq_double: seq[float64] = @[float64(4.0), float64(5.0)]
    let result_double = mapFloat64(seq_double)
    assert result_double.len == 2
    assert result_double[0] == float64(16.0)
    assert result_double[1] == float64(25.0)
