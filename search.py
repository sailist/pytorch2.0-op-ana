import subprocess
import textwrap
import json
import yaml

with open("native_functions.yaml") as r:
    natives = yaml.safe_load(r)



with open('core.txt') as r:    
    core_names = [i.strip() for i in r.readlines()[:50]]



core_native = {item['func'].split('(')[0]:item for item in natives if item['func'].split('(')[0] in core_names}
core_native = [core_native[name] for name in core_names if name in core_native]


def post_process():
    res = []
    with open("search_result.log",'r') as r:
        lines = r.readlines()
        for line in lines:
            if "const auto" in line:
                continue
            if ");" in line:
                continue
            if "if" in line:
                continue
            if "=" in line:
                continue
            if line.strip().startswith("//"):
                continue
            
            res.append(line)
            
    with open("search_result.log",'w') as w:
        w.write(''.join(res))

def filter_output(opt):
    opt = opt.strip()
    res = []
    for line in opt.split('\n'):
        file,lno,right = line.split(':', maxsplit=2)
        if right[0] in {'/'}:
            continue
        elif right.startswith("  auto") or "return" in right:
            continue
            
        # print(line)
        res.append(line)
    opt = '\n'.join(res)
    opt = textwrap.indent(opt, "   - ")
    return opt
        

with open('search_result.log','w') as w:
    for ind, item in enumerate(core_native,start=1):
        core_func = item['func'].split('(', maxsplit=1)[0]
        # print(core_func)
        w.write(f" - {core_func}\n")
        # print(item['func'], item.keys())
        if '.' in core_func:
            core_func = core_func.split('.',maxsplit=1)[0]
        command = rf'grep -nr --include="*.cpp"  --include="*.h" --include="*.cu" {core_func}\( /home/haozhe/code/pytorch/aten'
        
        print(command)
        w.write(f'  - {core_func}\n')
        
        try:
            output = subprocess.check_output(command, shell=True, text=True)
        except:
            continue
        
        w.write(filter_output(output))
        w.write('\n')
        
        if 'dispatch' in item:
            for k,core_imp in item['dispatch'].items():
                if '.' in core_imp:
                    core_imp = core_imp.split('.',maxsplit=1)[0]
                command = rf'grep -nr --include="*.cpp"  --include="*.h" --include="*.cu" {core_imp}\( /home/haozhe/code/pytorch/aten'
                print(command)
                w.write(f'  - {core_imp}({k})\n')
                try:
                    output = subprocess.check_output(command, shell=True, text=True)
                except:
                    continue
                w.write(filter_output(output))
                w.write('\n')
        w.write('\n\n')

post_process()