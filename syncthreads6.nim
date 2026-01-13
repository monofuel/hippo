import std/[strformat, locks]

# threads is enabled by default in nim 2.0
# nim c --mm:atomicArc -r syncthreads6.nim

# ──────────────────────────────────────────────────────────────────────────────
# Configuration constants (easy to modify for testing different scenarios)
# ──────────────────────────────────────────────────────────────────────────────

const
  NUM_WORKERS = 4
  N = 8  # Vector length (multiple of NUM_WORKERS for simplicity)
  NUM_SCHED_THREADS = 2  # Support for 2 threads as requested

  A: array[N, int] = [1, 2, 3, 4, 5, 6, 7, 8]
  B: array[N, int] = [8, 7, 6, 5, 4, 3, 2, 1]  # Dot product should be 120

# ──────────────────────────────────────────────────────────────────────────────
# Very minimal cooperative task pool using closure iterators (now per OS thread)
# ──────────────────────────────────────────────────────────────────────────────

type
  Task = iterator (): bool  # No {.closure.} here—it's implicit for closures

proc runAllTasks(tasks: seq[Task]) =
  ## Very dumb round-robin scheduler (runs on its own OS thread)
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
# Our toy barrier implemented with a shared counter + generation (now thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

var
  barrierLock: Lock
  barrierCounter: int = 0
  barrierGeneration: int = 0   # like a "phase" / ticket
  shared: array[NUM_WORKERS, int]  # "Shared memory" for partial sums
  finalResult: int = 0  # Final dot product result
  tasks: array[NUM_WORKERS, Task]  # Moved up for visibility in thread proc

template syncthreads() =
  ## Simulates CUDA's __syncthreads() as a cooperative barrier (now with locks for multi-thread safety).
  ## Polls with yields while waiting, to allow local scheduler to run other tasks.
  echo &"[w{threadIdxX}] reached barrier (gen={barrierGeneration})"

  acquire(barrierLock)
  let myGen = barrierGeneration
  inc barrierCounter
  let arrived = barrierCounter
  if arrived == NUM_WORKERS:
    echo "[barrier] ALL arrived → releasing everyone"
    inc barrierGeneration
    barrierCounter = 0
  release(barrierLock)

  while true:
    acquire(barrierLock)
    if barrierGeneration > myGen:
      release(barrierLock)
      break
    release(barrierLock)
    yield true  # Yield to local scheduler while waiting

  echo &"[w{threadIdxX}] passed barrier (new gen={barrierGeneration})"


# ──────────────────────────────────────────────────────────────────────────────
# Worker "kernel" – dot product with parallel compute + tree reduction (needs multiple syncs)
# ──────────────────────────────────────────────────────────────────────────────

proc worker(id: int): Task =
  return iterator (): bool =
    let threadIdxX = id  # Mimic CUDA's threadIdx.x (local to this "thread")
    let blockDimX = NUM_WORKERS  # Mimic blockDim.x

    echo &"[w{threadIdxX}] start"

    # ── Compute partial dot products (each thread handles N/blockDimX elements) ─
    let elementsPerThread = N div blockDimX  # 2 for N=8, blockDim=4
    var partial = 0

    for step in 0..<elementsPerThread:
      let idx = threadIdxX + step * blockDimX
      partial += A[idx] * B[idx]
      echo &"[w{threadIdxX}] computed A[{idx}] * B[{idx}] = {A[idx] * B[idx]}"
      yield true  # Cooperative yield to simulate "work quanta" / interleaving

    # ── Store partial in "shared memory" ──────────────────────────────────────
    echo &"[w{threadIdxX}] partial sum = {partial}"
    shared[threadIdxX] = partial

    # ── BARRIER 1 (sync before reduction) ─────────────────────────────────────
    syncthreads()

    # ── Parallel tree reduction (requires additional syncthreads inside loop) ──
    # This is more complicated: log(blockDim) steps, each with a syncthreads
    var stride = blockDimX div 2
    while stride > 0:
      if threadIdxX < stride:
        shared[threadIdxX] += shared[threadIdxX + stride]
        echo &"[w{threadIdxX}] reduced shared[{threadIdxX}] += shared[{threadIdxX + stride}] → {shared[threadIdxX]}"
      # ── BARRIER (sync after each reduction step) ────────────────────────────
      syncthreads()
      stride = stride div 2

    # ── Final result is in shared[0]; thread 0 copies to global ───────────────
    if threadIdxX == 0:
      finalResult = shared[0]
      echo &"[w{threadIdxX}] final dot product = {finalResult}"

    echo &"[w{threadIdxX}] finished"

    yield false  # Signal done (iterator will finish after this)


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler threads (now using 2 OS threads for the workers)
# ──────────────────────────────────────────────────────────────────────────────

proc schedulerThread(groupId: int) {.thread.} =
  ## Runs the scheduler for a group of tasks
  # NB. this gcsafe is only allowed because tasks is shared, and we are using --mm:atomicArc
  {.gcsafe.}:
    let tasksPerGroup = NUM_WORKERS div NUM_SCHED_THREADS
    let startIdx = groupId * tasksPerGroup
    let endIdx = startIdx + tasksPerGroup - 1

    var groupTasks: seq[Task]
    for i in startIdx..endIdx:
      groupTasks.add(tasks[i])

    runAllTasks(groupTasks)
  


# ──────────────────────────────────────────────────────────────────────────────
# Main – like launching a kernel with blockDim.x = NUM_WORKERS, gridDim=1
# ──────────────────────────────────────────────────────────────────────────────

var
  schedThreads: array[NUM_SCHED_THREADS, Thread[int]]

when isMainModule:
  initLock(barrierLock)  # Initialize the lock

  echo "Starting ", NUM_WORKERS, " pseudo-threads (like a CUDA block) on ", NUM_SCHED_THREADS, " OS threads..."
  echo "Computing dot product of A = ", A, " and B = ", B, "..."

  for i in 0..<NUM_WORKERS:
    tasks[i] = worker(i)

  for g in 0..<NUM_SCHED_THREADS:
    createThread(schedThreads[g], schedulerThread, g)

  joinThreads(schedThreads)

  echo "\nAll workers finished!"
  echo &"Final result: {finalResult}"
  assert finalResult == 120, &"Expected dot product to be 120, but got {finalResult}"
