--path:"../src"

--cc:clang
--clang.cpp.exe:hipcc
--clang.cpp.linkerexe:hipcc
--passC: "--offload-arch=gfx1100"
--passC: "--offload-arch=gfx1102"