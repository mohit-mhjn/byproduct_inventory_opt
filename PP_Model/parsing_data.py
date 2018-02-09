import json
import os
import glob

filename =open('imbalance_result.json')

def read_json(filename):
    print(filename)
    data = json.load(filename,encoding="utf-8")
    for k in data['CPLEXSolution']['variables']:
        print(k['name'],k['value'])

read_json(filename)

data['Solution'][1]=[]
solution = data['Solution'][1]['Variable']
HA_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'HA[' in k}
Ic_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'Ic[' in k}
x_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'x[' in k}
uc_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'uc[' in k}
uf_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'uf[' in k}
vc_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'vc[' in k}
vf_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'vf[' in k}
xc_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'xc[' in k}
xf_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'xf[' in k}
z_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'z[' in k}
ze_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'ze[' in k}