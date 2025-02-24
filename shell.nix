{ pkgs ? import <nixpkgs> {} }:

let
  unstable = import <nixos-unstable> { config.allowUnfree = true; };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    cudatoolkit
    unstable.nim
    unstable.nim-atlas
    unstable.nimble
    tbb # required for hip-cpu
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH
    export EXTRA_CFLAGS="-I${pkgs.cudatoolkit}/include"
    export EXTRA_LDFLAGS="-L${pkgs.cudatoolkit}/lib"
  '';
}