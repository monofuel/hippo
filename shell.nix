{ pkgs ? import <nixpkgs> {} }:

let
  unstable = import <nixos-unstable> { config.allowUnfree = true; };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    unstable.cudatoolkit
    unstable.cudaPackages.cuda_nvcc
    unstable.cudaPackages.cuda_cudart
    linuxPackages.nvidia_x11 # needed to run cuda binaries
    unstable.nim
    unstable.nim-atlas
    unstable.nimble
    tbb # required for hip-cpu
  ];

  shellHook = ''
    export CUDA_PATH=${unstable.cudatoolkit}
    export LD_LIBRARY_PATH=${unstable.cudatoolkit}/lib:${unstable.cudaPackages.cuda_cudart}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
    export EXTRA_CFLAGS="-I${unstable.cudatoolkit}/include"
    export EXTRA_LDFLAGS="-L${unstable.cudatoolkit}/lib -L${unstable.cudaPackages.cuda_cudart}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
  '';
}