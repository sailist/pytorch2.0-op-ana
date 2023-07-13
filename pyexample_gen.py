import re
from collections import Counter
from pprint import pprint
import json

with open("core.txt") as r:
    core_names = [i.strip() for i in r.readlines()]


pattern = r"func: ([\w_0-9.]+)(\(.+\)|\(\)) -> ([\w!()\[\]]+)"

match = re.compile(pattern)


def deinputsv2(func_str):
    if func_str == "()":
        return []
    func_str = func_str.strip("()")
    args = []
    for arg in func_str.split(", "):
        if arg == "*":
            continue

        dtype, key = arg.rsplit(maxsplit=1)

        if len(key.split("=")) > 1:
            key, default = key.split("=")
        else:
            default = None

        if default is not None:
            args.append({"name": key, "dtype": dtype})
        else:
            args.append({"name": key, "dtype": dtype, "default": default})

    return args


deres = {}

with open("./funcs.md") as r:
    lines = r.readlines()
    res = [[match.findall(i), i] for i in lines]
    for i, line in res:
        # print(i,line)
        if len(i) > 0:
            i = i[0]
            # print(i[1], )
            if i[0] in core_names:
                print(i)
                deres[i[0]] = {"name": i[0], "inputs": deinputsv2(i[1]), "output": i[2]}


print(len(deres))

import_str = """
from aten_gen import *
"""



type_map = {
    "float[]": "List[float]",
    "SymInt[]": "List[SymInt]",
    "SymInt[2]": "SymInt2",
    "int[2]": "Int2",
    "int[]": "List[int]",
    "int[1]": "Int1",
}


argdoc_pattern = """        {name} ({dtype}): _description_"""
retdoc_pattern = """        {dtype}: _description_"""

func_pattern = '''
def run_PY{name}():
    PY{name}   
 
run_PY{name}()
'''


def dearg(dtype):
    optional = False
    if dtype.endswith("?"):
        optional = True
        dtype = dtype.rstrip("?")
    if dtype == "Tensor[]":
        dtype = "TensorList"
    if dtype.startswith("Tensor("):
        if "!" in dtype:
            dtype = "MutatedTensor"
        else:
            dtype = "AliasTensor"

    dtype = type_map.get(dtype, dtype)
    if optional:
        dtype = f"Optional[{dtype}]"
    return dtype


with open("aten_example_gen.py", "w") as w:
    w.write(import_str)
    for ind, core_name in enumerate(core_names, start=1):
        if core_name in deres:
            print(deres[core_name])
            funcast = deres[core_name]
            funcargs = []
            optional = False

            argdoc = []
            funcname = funcast["name"].replace(".", "_")
            for arg in funcast["inputs"]:
                dtype = dearg(arg['dtype'])
                argdoc.append(argdoc_pattern.format(name=arg["name"], dtype=dtype))
                funcargs.append(f"{arg['name']} :{dtype}")

            rettype = dearg(funcast["output"])

            retdoc = retdoc_pattern.format(dtype=rettype)

            funcstr = func_pattern.format(
                ind=ind,
                name=funcname,
                funcarg=", ".join(funcargs),
                ret=rettype.strip("()"),
                retdoc=retdoc,
                argdoc="\n".join(argdoc),
            )
            w.write(funcstr)
