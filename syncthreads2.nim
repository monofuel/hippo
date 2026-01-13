import std/strformat

# ──────────────────────────────────────────────────────────────────────────────
# Very minimal single-threaded "cooperative thread pool" using closure iterators
# ──────────────────────────────────────────────────────────────────────────────

type
  Task = proc(): bool {.closure.}   # returns true = "still working", false = "done"

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

proc worker(id: int): Task =
  var state = 0  # 0=phase1, 1=barrier, 2=phase2, 3=done
  var phase1_counter = 0
  var phase2_counter = 0
  var barrier_gen = -1  # Generation we're waiting for

  proc(): bool {.closure.} =
    case state:
      of 0:  # Phase 1
        if phase1_counter == 0:
          echo &"[w{id}] start"
          echo &"[w{id}] phase 1"
        phase1_counter += 1
        if phase1_counter <= 3 + id:
          echo &"[w{id}] phase 1 work {phase1_counter}"
          return true  # continue working
        else:
          state = 1  # move to barrier
          return true  # continue to barrier

      of 1:  # Barrier
        if barrier_gen == -1:  # First time hitting barrier
          echo &"[w{id}] reached barrier (gen={barrierGeneration})"
          barrier_gen = barrierGeneration
          atomicInc barrierCounter

          # Check if we're the last one to arrive
          if barrierCounter == NUM_WORKERS:
            echo "[barrier] ALL arrived → releasing everyone"
            atomicInc barrierGeneration
            barrierCounter = 0

        # Wait for the generation to change from what we captured
        if barrierGeneration == barrier_gen:
          return true  # still waiting

        echo &"[w{id}] passed barrier (new gen={barrierGeneration})"
        state = 2  # move to phase 2
        barrier_gen = -1  # Reset for potential future barriers
        return true

      of 2:  # Phase 2
        if phase2_counter == 0:
          echo &"[w{id}] phase 2"
        phase2_counter += 1
        if phase2_counter <= 5 - id:
          echo &"[w{id}] phase 2 work {phase2_counter}"
          return true  # continue working
        else:
          state = 3  # done
          echo &"[w{id}] finished"
          return false  # finished

      else:  # Done
        return false


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

when isMainModule:
  echo "Starting ", NUM_WORKERS, " pseudo-threads (closure iterators)..."

  var tasks: array[NUM_WORKERS, Task]

  for i in 0..<NUM_WORKERS:
    tasks[i] = worker(i)

  runAllTasks(tasks[0], tasks[1], tasks[2], tasks[3])

  echo "\nAll workers finished!"
