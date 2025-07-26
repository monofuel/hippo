import std/[os, strformat, strutils]
import compiler/[idents, options, modulegraphs, passes, lineinfos, sem, pathutils, ast,
                astalgo, modules, condsyms, passaux]

let file = "tests/hip/julia_2.nim"

var graph: ModuleGraph = block:
  var
    cache: IdentCache = newIdentCache()
    config: ConfigRef = newConfigRef()
  let path = getCurrentDir() / "../Nim/lib"
  config.libpath = AbsoluteDir(path)
  config.searchPaths.add config.libpath
  config.projectFull = AbsoluteFile(file)
  initDefines(config.symbols)
  newModuleGraph(cache, config)

type
  CustomContext = ref object of PPassContext
    module: PSym

proc passOpen(graph: ModuleGraph; module: PSym; idgen: IdGenerator): PPassContext =
  CustomContext(module: module)

proc annotateDeviceFuncs(n: PNode): PNode =
  result = n
  if n.kind == nkProcDef:
    let pragmas = n[pragmasPos]
    var hasInline = false
    var hasGpuAnno = false
    var hasGlobal = false
    for p in pragmas:
      if p.kind == nkIdent and p.ident.s == "inline":
        hasInline = true
      elif p.kind == nkExprColonExpr and p[0].ident.s == "codegenDecl":
        let declStr = p[1].strVal
        if declStr.contains("__host__ __device__"):
          hasGpuAnno = true
        if declStr.contains("__global__"):
          hasGlobal = true
    if hasInline and not hasGpuAnno and not hasGlobal:
      let gpuDecl = newNodeI(nkExprColonExpr, n.info)
      gpuDecl.add newIdentNode(getIdent(graph.cache, "codegenDecl"), n.info)
      gpuDecl.add newStrNode(nkStrLit, "__host__ __device__ $# $#$#")
      pragmas.add(gpuDecl)
  if n.safeLen > 0:
    for i in 0..<n.len:
      n[i] = annotateDeviceFuncs(n[i])
echo "4"
proc passNode(c: PPassContext, n: PNode): PNode =
  result = annotateDeviceFuncs(n)
  if sfMainModule in CustomContext(c).module.flags:
    # Optional: debug output
    echo "Processed node: ", n.kind

proc passClose(graph: ModuleGraph; p: PPassContext, n: PNode): PNode =
  discard

registerPass(graph, semPass)
registerPass(graph, makePass(passOpen, passNode, passClose))
compileProject(graph)