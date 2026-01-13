# test utils

import std/unittest

template testSkipPlatforms*(name: string, skipPlatforms: varargs[string], body: untyped): untyped =
  ## Test template that skips execution on specified platforms.
  ## Runs on all platforms EXCEPT the ones listed in skipPlatforms.
  ##
  ## Available platforms: "HIP", "CUDA", "HIP_CPU", "SIMPLE"
  ##
  ## Example:
  ##   testSkipPlatforms "my test", "SIMPLE", "HIP_CPU":
  ##     # test code here - runs on HIP and CUDA but not SIMPLE or HIP_CPU
  ## Test template that skips execution on specified platforms.
  ## Runs on all platforms EXCEPT the ones listed in skipPlatforms.
  ##
  ## Example:
  ##   testSkipPlatforms "my test", "SIMPLE":
  ##     # test code here

  # Check if current runtime should be skipped
  const currentRuntime =
    when defined(HippoRuntime):
      when HippoRuntime == "HIP": "HIP"
      elif HippoRuntime == "CUDA": "CUDA"
      elif HippoRuntime == "HIP_CPU": "HIP_CPU"
      elif HippoRuntime == "SIMPLE": "SIMPLE"
      else: "HIP"  # default
    else:
      "HIP"  # default if not defined

  var shouldSkip = false
  for skipPlatform in skipPlatforms:
    if skipPlatform == currentRuntime:
      shouldSkip = true
      break

  if shouldSkip:
    test name:
      skip()
  else:
    test name:
      body
