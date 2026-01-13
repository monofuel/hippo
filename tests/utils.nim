# test utils

import std/unittest

template testSkipPlatforms*(name: string, skipPlatforms: varargs[string], body: untyped): untyped =
  ## Test template that skips execution on specified platforms.
  ## Runs on all platforms EXCEPT the ones listed in skipPlatforms.
  ##
  ## Available platforms: "HIP", "CUDA", "HIP_CPU", "SIMPLE", "SIMPLE_NO_THREADS"
  ##
  ## Note: "SIMPLE" = SIMPLE backend with --threads:on
  ##       "SIMPLE_NO_THREADS" = SIMPLE backend with --threads:off
  ##       Specifying "SIMPLE" in skipPlatforms skips BOTH SIMPLE variants
  ##
  ## Example:
  ##   testSkipPlatforms "my test", "SIMPLE":
  ##     # test code here - runs on HIP, CUDA, and HIP_CPU but not SIMPLE or SIMPLE_NO_THREADS

  # Check if current runtime should be skipped
  const currentRuntime =
    when defined(HippoRuntime):
      when HippoRuntime == "HIP": "HIP"
      elif HippoRuntime == "CUDA": "CUDA"
      elif HippoRuntime == "HIP_CPU": "HIP_CPU"
      elif HippoRuntime == "SIMPLE":
        when compileOption("threads"): "SIMPLE"
        else: "SIMPLE_NO_THREADS"
      else: "HIP"  # default
    else:
      "HIP"  # default if not defined

  var shouldSkip = false
  for skipPlatform in skipPlatforms:
    if skipPlatform == currentRuntime:
      shouldSkip = true
      break
    # Handle "SIMPLE" matching both SIMPLE and SIMPLE_NO_THREADS
    elif skipPlatform == "SIMPLE" and (currentRuntime == "SIMPLE" or currentRuntime == "SIMPLE_NO_THREADS"):
      shouldSkip = true
      break

  if shouldSkip:
    test name:
      skip()
  else:
    test name:
      body
