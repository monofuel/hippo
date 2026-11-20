import std/os

# Keep the focused AMD invocation free of compiler flags.
if getEnv("HIPPO_APPENDIX_BACKEND") == "HIP":
  switch("cc", "hipcc")
  switch("define", "HippoRuntime:HIP")
  switch("define", "useMalloc")
  let deviceLibs = getEnv("HIP_DEVICE_LIB_PATH")
  if deviceLibs.len > 0:
    switch("passC", "--rocm-device-lib-path=" & deviceLibs)

# Exercise the synchronization with compiler optimization enabled.
switch("opt", "speed")
