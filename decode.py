import re
import json

pattern = r'func: ([\w_0-9.]+)(\(.+\)|\(\)) -> ([\w!()\[\]]+)'

match = re.compile(pattern)

def deinputs(func_str):
    if func_str == "()":
        return [{},{}]
    func_str = func_str.strip('()')
    args = {}
    res = []
    for arg in func_str.split(', '):
        if arg == '*':
            res.append(args)
            args = {}
        else:
            dtype, key = arg.rsplit(maxsplit=1)
            args[key] = dtype
                
    res.append(args)
    if len(res) == 1:
        res.append({})
    
    
    args = [f"{v} {k}" for k,v in res[0].items()]
    kwargs = [f"{v} {k}" for k,v in res[1].items()]
    
    if len(res[1]) > 0:
        defunc_str = ', '.join((args + ["*"] + kwargs))
    else:
        defunc_str = ', '.join(args)
        
    assert (defunc_str == func_str)
    return res
            

def deinputsv2(func_str):
    if func_str == "()":
        return [{},{}]
    func_str = func_str.strip('()')
    args = []
    for arg in func_str.split(', '):
        if arg == '*':
            continue
        
        dtype, key = arg.rsplit(maxsplit=1)
        
        if len(key.split('=')) > 1:
            key, default = key.split("=")
        else:
            default = None
        
        if default is not None:
            args.append({'name':key, 'dtype':dtype})
        else:
            args.append({'name':key, 'dtype':dtype, "default": default})

    return args

deres = []

with open('./funcs.md') as r:
    lines = r.readlines()
    res = [[match.findall(i),i] for i in lines]
    for i,line in res:
        # print(i,line)
        if len(i) > 0:
            i = i[0]
            # print(i[1], )
            deres.append({"name":i[0],"inputs": deinputsv2(i[1]), "output":i[2]} )

for item in deres:
    item['inputs']


with open("funcs.json",'w') as w:
    json.dump(deres, w, indent=2)