{
  description = "Hippo development environments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system} = {
        # Default to basic CPU-only environment
        default = self.devShells.${system}.basic;

        # Unified development environment with NVIDIA/CUDA + AMD/ROCm + Basic support
        all = pkgs.mkShell {
          buildInputs = with pkgs; [
            # NVIDIA/CUDA packages
            cudatoolkit
            cudaPackages.cuda_nvcc
            cudaPackages.cuda_cudart
            linuxPackages.nvidia_x11
            
            # AMD/ROCm packages
            rocmPackages.clr
            rocmPackages.hip-common
            rocmPackages.hipcc
            rocmPackages.rocm-device-libs
            rocmPackages.rocm-runtime
            
            # Basic Nim development
            nim
            nim-atlas
            nimble
            tbb
          ];

          shellHook = ''
            # NVIDIA/CUDA environment setup
            export CUDA_PATH=${pkgs.cudatoolkit}
            
            # AMD/ROCm environment setup
            export ROCM_PATH=${pkgs.rocmPackages.rocm-runtime}
            export HIP_PATH=${pkgs.rocmPackages.hip-common}
            
            # Combined library paths
            export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.rocmPackages.rocm-runtime}/lib:${pkgs.rocmPackages.hip-common}/lib:$LD_LIBRARY_PATH
            
            # Combined compiler flags
            export EXTRA_CFLAGS="-I${pkgs.cudatoolkit}/include -I${pkgs.rocmPackages.hip-common}/include -I${pkgs.rocmPackages.rocm-runtime}/include"
            export EXTRA_LDFLAGS="-L${pkgs.cudatoolkit}/lib -L${pkgs.cudaPackages.cuda_cudart}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib -L${pkgs.rocmPackages.rocm-runtime}/lib -L${pkgs.rocmPackages.hip-common}/lib"
            
            echo "🚀🔥⚡ Unified development environment ready (NVIDIA/CUDA + AMD/ROCm + Basic Nim)"
          '';
        };

        # NVIDIA/CUDA development only
        nvidia = pkgs.mkShell {
          buildInputs = with pkgs; [
            cudatoolkit
            cudaPackages.cuda_nvcc
            cudaPackages.cuda_cudart
            linuxPackages.nvidia_x11
            nim
            nim-atlas
            nimble
            tbb
          ];

          shellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
            export EXTRA_CFLAGS="-I${pkgs.cudatoolkit}/include"
            export EXTRA_LDFLAGS="-L${pkgs.cudatoolkit}/lib -L${pkgs.cudaPackages.cuda_cudart}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            echo "🚀 NVIDIA/CUDA development environment ready"
          '';
        };

        # AMD/ROCm development only
        amd = pkgs.mkShell {
          buildInputs = with pkgs; [
            rocmPackages.clr
            rocmPackages.hip-common
            rocmPackages.hipcc
            rocmPackages.rocm-device-libs
            rocmPackages.rocm-runtime
            nim
            nim-atlas
            nimble
            tbb
          ];

          shellHook = ''
            export HIP_PLATFORM=amd
            export ROCM_PATH=${pkgs.rocmPackages.rocm-runtime}
            export HIP_PATH=${pkgs.rocmPackages.hip-common}
            export LD_LIBRARY_PATH=${pkgs.rocmPackages.rocm-runtime}/lib:${pkgs.rocmPackages.hip-common}/lib:$LD_LIBRARY_PATH
            export EXTRA_CFLAGS="-I${pkgs.rocmPackages.hip-common}/include -I${pkgs.rocmPackages.rocm-runtime}/include"
            export EXTRA_LDFLAGS="-L${pkgs.rocmPackages.rocm-runtime}/lib -L${pkgs.rocmPackages.hip-common}/lib"
            echo "🔥 AMD/ROCm development environment ready"
          '';
        };

        # Basic CPU-only development
        basic = pkgs.mkShell {
          buildInputs = with pkgs; [
            nim
            nim-atlas
            nimble
            tbb
          ];

          shellHook = ''
            echo "⚡ Basic Nim development environment ready (CPU only)"
          '';
        };
      };
    };
} 