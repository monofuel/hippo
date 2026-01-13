import std/strformat  # Removed locks since not using atomicInc

# ──────────────────────────────────────────────────────────────────────────────
# Very minimal single-threaded "cooperative thread pool" using closure iterators
# ──────────────────────────────────────────────────────────────────────────────

type
  Task = iterator (): bool  # No {.closure.} here—it's implicit for closures

proc runAllTasks(tasks: varargs[Task]) =
  ## Very dumb round-robin scheduler
  var active = newSeq[Task]()
  for t in tasks: active.add(t)

  var round = 0

  while active.len > 0:
    round.inc
    var i = 0
    while i < active.len:
      let stillWorking = active[i]()
      if not stillWorking:
        # Task finished → remove it
        active.del(i)
      else:
        inc i

    # Optional: show progress (comment out in real code)
    # when defined(debug):
    #   echo &"round {round:3}  active tasks: {active.len}"


# ──────────────────────────────────────────────────────────────────────────────
# Our toy barrier implemented with a shared counter + generation
# ──────────────────────────────────────────────────────────────────────────────

const NUM_WORKERS = 4

var
  barrierCounter {.global.}: int = 0
  barrierGeneration {.global.}: int = 0   # like a "phase" / ticket

template syncthreads() =
  ## Simulates CUDA's __syncthreads() as a cooperative barrier.
  ## Yields control while waiting, to mimic "blocking" in single-thread coop.
  echo &"[w{threadIdxX}] reached barrier (gen={barrierGeneration})"

  let myGen = barrierGeneration
  inc barrierCounter
  let arrived = barrierCounter

  if arrived == NUM_WORKERS:
    echo "[barrier] ALL arrived → releasing everyone"
    inc barrierGeneration
    barrierCounter = 0
  else:
    while barrierGeneration == myGen:
      yield true

  echo &"[w{threadIdxX}] passed barrier (new gen={barrierGeneration})"


# ──────────────────────────────────────────────────────────────────────────────
# Worker "kernel" – now looks more like linear GPU code with syncthreads()
# ──────────────────────────────────────────────────────────────────────────────

proc worker(id: int): Task =
  return iterator (): bool =
    let threadIdxX = id  # Mimic CUDA's threadIdx.x (local to this "thread")

    echo &"[w{threadIdxX}] start"

    # ── Phase 1 (with some cooperative yields to simulate work quanta) ────────
    echo &"[w{threadIdxX}] phase 1"
    for i in 1..(3 + threadIdxX):     # different amount of "work"
      echo &"[w{threadIdxX}] phase 1 work {i}"
      yield true              # cooperative yield = "warp scheduler switch"

    # ── BARRIER (like in CUDA) ────────────────────────────────────────────────
    syncthreads()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    echo &"[w{threadIdxX}] phase 2"
    for i in 1..(5 - threadIdxX):
      echo &"[w{threadIdxX}] phase 2 work {i}"
      yield true

    echo &"[w{threadIdxX}] finished"

    yield false  # Signal done (iterator will finish after this)


# ──────────────────────────────────────────────────────────────────────────────
# Main – like launching a kernel with blockDim.x = NUM_WORKERS, gridDim=1
# ──────────────────────────────────────────────────────────────────────────────

when isMainModule:
  echo "Starting ", NUM_WORKERS, " pseudo-threads (like a CUDA block)..."

  var tasks: array[NUM_WORKERS, Task]

  for i in 0..<NUM_WORKERS:
    tasks[i] = worker(i)

  runAllTasks(tasks[0], tasks[1], tasks[2], tasks[3])

  echo "\nAll workers finished!"
