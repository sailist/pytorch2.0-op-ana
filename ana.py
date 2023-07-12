import re
from pprint import pprint
import json
with open('core.txt') as r:
    core_names = set([i.strip() for i in r.readlines()][:51])
    

pattern = r'func: ([\w_0-9.]+)(\(.+\)|\(\)) -> ([\w!()\[\]]+)'

match = re.compile(pattern)
           

def deinputsv2(func_str):
    if func_str == "()":
        return []
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
            if i[0] in core_names:
                deres.append({"name":i[0],"inputs": deinputsv2(i[1]), "output":i[2]} )

from collections import Counter

print(len(deres))

native_names =set([i["name"] for i in deres])
for core_name in core_names:
    if core_name not in native_names:
        print(core_name)


counter = Counter()
for item in deres:
    for arg in item['inputs']:
        # print(arg)
        counter.update([arg['dtype'] + " " + arg['name']])
        
pprint(counter.most_common(40))

counter = Counter()
for item in deres:
    for arg in item['inputs']:
        counter.update([arg['dtype']])
    counter.update([item['output']])
        
# pprint(counter.most_common(20))