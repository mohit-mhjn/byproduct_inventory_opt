from __future__ import division

from pyomo.core.base import ConcreteModel, Param, Set, Var, Constraint, Objective,\
    NonNegativeIntegers, NonNegativeReals, maximize
import json
import re
from docloud.job import JobClient
from pyomo.environ import *
import pandas as pd
import os
import numpy as np
import sys
import itertools
# MIP formulation from A Mixed Integer Linear Program for Operational Planning Meat Planing Plant
# AVIOTA
# Rishikesh

model = ConcreteModel()

# os.chdir('/home/rishi/GIT/imbalanced_parts/PP_Model/')
if len(sys.argv) > 1:
    filename = sys.argv[-1]
else:
    filename = '/home/rishi/GIT/imbalanced_parts/PP_Model/Model_Input_Final.xlsm'

demand_fresh = pd.read_excel(filename, sheetname="Demand_Fresh")
demand_fresh = demand_fresh.dropna(axis='columns', how='all')
# demand_fresh_df = pd.DataFrame(0, index=range(1, len(demand_fresh)+1), columns=range(1, len(demand_fresh.loc[0])))
# for i in range(0, len(demand_fresh)):
#     for j in range(1, len(demand_fresh.loc[0])):
#         demand_fresh_df.loc[i+1][j]=demand_fresh.loc[i][j]


def df(model, i, j):
    try:
        return np.nan_to_num(demand_fresh[demand_fresh['item code']==i][j].iloc[0])
    except IndexError:
        return 0


demand_frozen = pd.read_excel(filename, sheetname="Demand_Frozen")
demand_frozen = demand_frozen.dropna(axis='columns', how = 'all')
# demand_frozen_dc = pd.DataFrame(0, index=range(1, len(demand_frozen)+1), columns=range(1, len(demand_frozen.loc[0])))
# for i in range(0, len(demand_frozen)):
#     for j in range(1, len(demand_frozen.loc[0])):
#         demand_frozen_dc.loc[i+1][j] = demand_frozen.loc[i][j]


def dc(model, i, j):
    try:
        return np.nan_to_num(demand_frozen[demand_frozen['item code'] == i][j].iloc[0])
    except IndexError:
        return 0

ff = pd.read_excel('Model_Input_Final.xlsm', sheetname="Fresh_Frozen_Relation")
ff = ff.dropna(axis='columns', how='all')
ff =ff.astype(str)
for i in demand_frozen['item code'].astype(str).unique():
    if i in ff['Frozen SKU'].astype(str).unique():
        demand_frozen['item code'][demand_frozen.index[demand_frozen['item code']==i]]=ff['Fresh SKU'][ff.index[ff['Frozen SKU']==i]]

# products = list(set(list(demand_fresh['item code'])+list(demand_frozen['item code'])))
selling_price_fresh = pd.read_excel(filename, sheetname='Selling_Price_Fresh')
selling_price_fresh = selling_price_fresh.dropna(axis='columns', how='all')

def pf(model,i):
    try:
        return np.nan_to_num(float(selling_price_fresh[selling_price_fresh['item code']==i]['Selling Price Fresh (in RM/kg)'].iloc[0]))
    except IndexError:
        return 10


selling_price_frozen = pd.read_excel(filename, sheetname='Selling_Price_Frozen')
selling_price_frozen = selling_price_frozen.dropna(axis='columns', how='all')
for i in selling_price_frozen['item code'].astype(str).unique():
    if i in ff['Frozen SKU'].astype(str).unique():
        selling_price_frozen['item code'][selling_price_frozen.index[selling_price_frozen['item code']==i]]=ff['Fresh SKU'][ff.index[ff['Frozen SKU']==i]]


def pc(model, i):
    try:
        return np.nan_to_num(float(selling_price_frozen[selling_price_frozen['item code']==i]['Selling Price Frozen (in RM/kg)'].iloc[0]))
    except IndexError:
        return 10


holding_price_fresh = pd.read_excel(filename, sheetname='Holding_Cost_Fresh')
holding_price_fresh = holding_price_fresh.dropna(axis='columns', how='all')


def hf(model, i):
    try:
        return np.nan_to_num(float(
            holding_price_fresh[holding_price_fresh['item code'] == i]['Holding_Cost_Fresh (in RM/kg)'].iloc[0]))
    except IndexError:
        return 10

holding_price_frozen = pd.read_excel(filename, sheetname='Holding_Cost_Frozen')
holding_price_frozen = holding_price_frozen.dropna(axis='columns', how='all')
for i in holding_price_frozen['item code'].astype(str).unique():
    if i in ff['Frozen SKU'].astype(str).unique():
        holding_price_frozen['item code'][holding_price_frozen.index[holding_price_frozen['item code']==i]]=ff['Fresh SKU'][ff.index[ff['Frozen SKU']==i]]


def hc(model, i):
    try:
        return np.nan_to_num(float(
            holding_price_frozen[holding_price_frozen['item code'] == i]['Holding_Cost_Frozen(In RM/kg)'].iloc[0]))
    except IndexError:
        return 10


penalty_fresh = pd.read_excel(filename,sheetname='Penalty_Fresh')
penalty_fresh = penalty_fresh.dropna(axis='columns', how='all')


def sf(model, i):
    try:
        return np.nan_to_num(float(
            penalty_fresh[penalty_fresh['item code'] == i]['Penalty_Cost_Fresh(in RM/kg)'].iloc[0]))
    except IndexError:
        return 10


penalty_frozen = pd.read_excel(filename, sheetname='Penalty_Frozen')
penalty_frozen = penalty_frozen.dropna(axis='columns', how='all')

for i in penalty_frozen['item code'].astype(str).unique():
    if i in ff['Frozen SKU'].astype(str).unique():
        penalty_frozen['item code'][penalty_frozen.index[penalty_frozen['item code']==i]]=ff['Fresh SKU'][ff.index[ff['Frozen SKU']==i]]


def sc(model,i):
    try:
        return np.nan_to_num(float(
            penalty_frozen[penalty_frozen['item code'] == i]['Penalty_Cost_Frozen(in RM/kg)'].iloc[0]))
    except IndexError:
        return 10


operational_cost = pd.read_excel(filename,sheetname='Operational_Cost')
operational_cost = operational_cost.dropna(axis ='columns', how = 'all')


def c(model,j):
    return np.nan_to_num(float(operational_cost.iloc[j-1][1]))


operational_cost_overtime = pd.read_excel(filename, sheetname='Operational_Cost_Overtime')
operational_cost_overtime = operational_cost_overtime.dropna(axis='columns', how = 'all')


def ce(model,j):
    return np.nan_to_num(float(operational_cost_overtime.iloc[j-1][1]))


available_hour_regular = pd.read_excel(filename, sheetname='Available_Hour_Regular')
available_hour_regular = available_hour_regular.dropna(axis ='columns', how = 'all')


def Ta(model,t):
    return np.nan_to_num(float(available_hour_regular[available_hour_regular['Day']==t]['Available Hours'].iloc[0]))


available_hour_overtime = pd.read_excel(filename,sheetname='Available_Hour_Overtime')
available_hour_overtime = available_hour_overtime.dropna(axis ='columns', how = 'all')


def Te(model,t):
    return np.nan_to_num(float(available_hour_overtime[available_hour_overtime['Day']==t]['Available Hours'].iloc[0]))


common_data = pd.read_excel(filename,sheetname='Common_Data')
common_data = common_data.dropna(axis ='columns', how = 'all')


def Wf(model):
    return common_data["Value"][0]


def Wc(model):
    return common_data["Value"][1]
# def Ta(model,t):
#     return common_data["Value"][2]
# def Te(model):
#     return common_data["Value"][3]


def b(model):
    return common_data["Value"][5]


def F(model):
    print(float(common_data["Value"][6]))
    return float(common_data["Value"][6])


def tau(model):
    return int(common_data["Value"][7])


def delta(model):
    return common_data["Value"][7]
# def H(model):
#     return int(common_data["Value"][9])


shelf_lif = pd.read_excel(filename, sheetname='Shelf_Life')
shelf_life = int(max(shelf_lif['Shelf_Life']))

def L(model,i):
    try:
        L = list(j for j in range(int(shelf_lif[shelf_lif['item code']==i]['Shelf_Life'].iloc[0])))
    except IndexError:
        L = [0, 1]
    # L = L[:]
    return L
# Ld = list(i for i in range(shelf_life + 1))
# L = Ld[1:]


def Ld(model,i):
    try:
        Ld = list(j for j in range(int(shelf_lif[shelf_lif['item code']==i]['Shelf_Life'].iloc[0])))
        Ld = Ld[:]
    except IndexError:
        Ld = [0, 1]
    return Ld


operational_time = pd.read_excel(filename,sheetname='Operation_Time')
operational_time = operational_time.dropna(axis ='columns', how = 'all')


def t(model,j):
    return float(operational_time.iloc[j-1][1])


y = pd.read_excel(filename, sheetname='Yield',)
y = y.dropna(axis ='columns', how='all')
bird_type = y.columns[2:]
cut_pattern = list(set(y['Cutting Pattern']))
# products = list(i for i in range(1, len(y['item code'].unique())+1))
# products = np.repeat(products, len(y['Cutting Pattern'].unique()))
# cut_pattern = list(i for i in range(1, len(y['Cutting Pattern'].unique())+1))
# cut_pattern = cut_pattern*len(y['item code'].unique())

# b_t = list(i for i in range(1, len(y.columns)+1))
# y.columns = b_t
products = list(set(demand_fresh['item code']).intersection(set(y['item code'])).union(set(demand_frozen['item code']).intersection(set(y['item code']))))
# y = y.set_index([products, cut_pattern])
# del y[1]
# del y[2]
# y.columns = range(1, len(y.columns)+1)

# bird_type = y.columns


def yd(model, i, j, r):
    try:
        return np.nan_to_num(float(y[(y['item code']==i) & (y['Cutting Pattern']==j)][r].iloc[0]))
    except IndexError:
        return 0


bird_type_ratio = pd.read_excel(filename, sheetname='Bird_Type_Ratio')
bird_type_ratio = bird_type_ratio.dropna(axis='columns', how='all')


def alpha(model, r):
    return np.nan_to_num(float(bird_type_ratio[bird_type_ratio['Bird Type']==r]['Ratio'].iloc[0]))


days = list(demand_fresh.columns[1:])
availability = pd.read_excel(filename ,sheetname='Availability')
availability = availability.dropna(axis='columns', how='all')


def H(model,r):
    return availability[availability['Days']==r]['Availability'].iloc[0]


cutting_pattern = pd.read_excel(filename,sheetname='Cutting_Pattern')
cutting_pattern = cutting_pattern.dropna(axis='columns', how='all')
cutting_pattern=cutting_pattern.fillna(0)
cp=[]
for i in cutting_pattern.itertuples():
    cp.append((list(i[2:])))
for i in cp:
    if 0 in i:
        i.remove(0)
cp = [[int(j) for j in i] for i in cp]
sections = list(s for s in range(1, len(cp)+1))


def Jk(model, k):
    return cp[int(k-1)]
#  F = Fresh and C = Frozen
last_day = days[-1]

model.T = Set(initialize=days, ordered=True)     # planning horizon
model.J = Set(initialize=list(set(cut_pattern)), ordered = True)     # cutting patterns
model.P = Set(initialize=products, ordered = True)     # products
model.L = Set(model.P, initialize=L, ordered=True)     # shell life for fresh products
model.H = Param(model.T, initialize = H)         # min no of avaiable carcasses of type r in all planning horizon
model.K = Set(initialize = sections)             # no of sections
model.Jk = Set(model.K, initialize = Jk)         # cutting pattern
model.R = Set(initialize=bird_type, ordered = True)     # type of carcasses
model.alpha = Param(model.R, initialize = alpha)    # proportion of carcasses of type r
model.yd = Param(model.P, model.J, model.R, initialize = yd)   # yield of product i in pattern j on carcasses of type r
model.pf = Param(model.P,initialize = pf)    # SP of fresh product
model.pc = Param(model.P,initialize = pc)    # SP of frozen product
model.c = Param(model.J, initialize = c)     # operational cost of pattern j
model.ce = Param(model.J, initialize = ce)     # operational cost of pattern j in OT
model.b = Param(initialize = b)     # freezing cost /kg
model.hf = Param(model.P, initialize = hf)    # HC F
model.hc = Param(model.P, initialize = hc)    # HC Frozen
model.sf = Param(model.P, initialize = sf)     # penality for unstatisfed fresh
model.sc = Param(model.P, initialize = sc)     # penality for unstatisfed frozen
model.F = Param(initialize = F)    # Freezing tunnel capacity at t
model.df = Param(model.P, model.T, initialize = df) # Demand of fresh products
model.dc = Param(model.P, model.T, initialize = dc ) # Demand of frozen products
model.tau = Param(initialize = 0)         # Freezing Process duration
model.Wf = Param(initialize = Wf)     # warehouse capacity
model.Wc = Param(initialize = Wc)     # warehouse capacity
model.t = Param(model.J, initialize = t)    #cutting operation for pattern j
model.Ta = Param(model.T, initialize = Ta)    #avaiable work hours in each time period
model.Te = Param(model.T, initialize = Te)       # avaiable overtime hours
# model.delta = Param(initialize = delta)         # auxilary pattern for better control the available carcass
model.Ld = Set(model.P, initialize = Ld, ordered = True)
# Extra Parameters to handle computed indices
# td = list(range((days[0]-shelf_life), 1)) + days + list(range(days[-1]+1, days[-1]+shelf_life+1))
td = [0]+days+[5]
model.Td = Set(initialize=td, ordered=True)
model.x = Var(model.P, model.T, domain=NonNegativeReals)   # total quantity of product i to be processed in period t
model.xf = Var(model.P, model.Td, model.Td, domain=NonNegativeReals)        # quantity of fresh product i to process in period t to be sold at t'
model.z = Var(model.J, model.R, model.T, domain=NonNegativeIntegers)   # no of times to use pattern j on carcass r in period t in RT
model.ze = Var(model.J, model.R, model.T, domain=NonNegativeIntegers)    # no of times to use pattern j on carcass r in period t in OT
model.xc = Var(model.P, model.Td, domain=NonNegativeReals)        # quantity of frozen product i to process in t
model.vf = Var(model.P, model.T, domain=NonNegativeReals)        # quantity of fresh product i to be sold in t
model.vc = Var(model.P, model.T, domain=NonNegativeReals)        # quantity of frozen product i to be sold in t
model.Ic = Var(model.P, model.Td, domain=NonNegativeReals)        # quantity of frozen product i to hold in t
model.uf = Var(model.P, model.T, domain=NonNegativeReals)        # unsatisfied demand of fresh product i in t
model.uc = Var(model.P, model.T, domain=NonNegativeReals)        # unsatisfied demand of frozen product i in t
model.HA = Var(model.T, domain=NonNegativeIntegers)         # No of carcasses to be processed in period t
model.z.pprint()
print("model input done")
def objective_function(model):
    return sum(sum(model.pf[i]*model.vf[i, t] + model.pc[i]*model.vc[i, t] for i in model.P) for t in model.T) - sum(sum(model.b*model.xc[i, t] for i in model.P) for t in model.T) - sum(sum(model.hc[i]*model.Ic[i, t] for i in model.P) for t in model.T) - sum(sum(sum(model.hf[i]*model.xf[i, t, t+l] for t in model.T) for l in model.L[i] if l >= last_day) for i in model.P) - sum(sum(model.sf[i]*model.uf[i, t] - model.sc[i]*model.uc[i, t] for i in model.P) for t in model.T) - sum(sum(sum(model.c[j]*model.z[j, r, t] + model.ce[j]*model.ze[j, r, t] for j in model.J) for r in model.R) for t in model.T)
    # return sum(sum(model.Ic[i, t] for i in model.P)for t in model.Td)

model.OBJ = Objective(rule = objective_function, sense = maximize)


def carcass_availability(model,r, t,k):
        return sum(model.z[j,r,t] + model.ze[j, r, t] for j in model.Jk[k]) == model.alpha[r]*model.HA[t]

model.A2Constraint = Constraint(model.R, model.T, model.K,  rule = carcass_availability)

def carcass_limit(model, t):
        return model.HA[t] <= model.H[t]
model.A3Constraint = Constraint( model.T,rule = carcass_limit)

def cutting_pattern(model,i,t):
        return model.x[i,t] == sum(sum(model.yd[i,j,r] * (model.z[j,r,t] + model.ze[j,r,t]) for j in model.J ) for r in model.R )
model.A4Constraint = Constraint(model.P, model.T, rule = cutting_pattern)

def available_daily_wh (model,t):
    return sum(sum(model.z[j,r,t]* model.t[j] for j in model.J) for r in model.R) <= model.Ta[t]

model.A5Constraint = Constraint(model.T, rule = available_daily_wh)

def available_daily_ot(model,t):
    return sum(sum(model.ze[j,r,t]* model.t[j] for j in model.J) for r in model.R) <= model.Te[t]

model.A6Constraint = Constraint(model.T, rule = available_daily_ot)

def fresh_frozen_balance(model,i,t):
        # print(model.x[i,t] == sum(model.xf[i,t,t+l] for l in model.Ld[i]) + model.xc[i,t])
        return model.x[i,t] == sum(model.xf[i,t,t+l] for l in model.Ld[i] if t+l <=last_day) + model.xc[i,t]

model.A7Constraint = Constraint(model.P, model.T, rule = fresh_frozen_balance)

def fresh_product_sold(model,i,t):
        return model.vf[i,t] == sum(model.xf[i,t-l,t] for l in model.Ld[i] if t-l > 0 )

model.A8Constraint = Constraint(model.P, model.T, rule = fresh_product_sold)

def frozen_product_sold(model,i,t):
    if t > 0:
        return model.vc[i,t] == model.Ic[i,t-1] + model.xc[i,t-model.tau] - model.Ic[i,t]

model.A9Constraint = Constraint(model.P,model.T, rule = frozen_product_sold)

def demand_frozen_product(model, i, t):
        return model.vc[i,t] + model.uc[i,t] == model.dc[i,t]
model.A10Constraint = Constraint(model.P, model.T, rule = demand_frozen_product)

def demand_fresh_product (model,i, t):
        return model.vf[i,t] + model.uf[i,t] == model.df[i,t]
model.A11Constraint = Constraint(model.P, model.T,  rule = demand_fresh_product)

def freezing_capacity(model, t):
    return sum(sum(model.xc[i,t] for t in range(t-model.tau, t+1) if t >0 )for i in model.P) <= model.F

model.A12Constraint = Constraint(model.T, rule = freezing_capacity)

def fresh_product_wh_cap(model, t):
    return sum(sum(sum(model.xf[i,t-l,td] for td in range(t, t-l+len(L(model,i))) if td <=last_day) for l in model.L[i] if t-l > 0) for i in model.P) <= model.Wf

model.A13Constraint = Constraint(model.T, rule = fresh_product_wh_cap)

def frozen_product_wh_cap(model, t):
    if t < 0:
        return model.Ic[i,t] == 0
    else:
        return sum(model.Ic[i,t] for i in model.P) <= model.Wc

model.A15Constraint = Constraint(model.T, rule = frozen_product_wh_cap)

initial_inv_frozen = pd.read_excel(filename, sheetname='Initial_Inventory')
def initial_inventory(model,i):
    try:
        return model.Ic[i,0] ==np.nan_to_num(float(initial_inv_frozen[initial_inv_frozen['item code']==i]['Initial Inventory'].iloc[0]))
    except IndexError:
        return Constraint.Skip
# model.initial_inv_Constraint = Constraint(model.P, rule = initial_inventory)


# def initial_inventory_frozen_product(model,i):
for i in model.P:
    try:
        print(np.nan_to_num(float(initial_inv_frozen[initial_inv_frozen['item code']==i]['Initial Inventory'].iloc[0])))
        model.xc[i,0].fix(np.nan_to_num(float(initial_inv_frozen[initial_inv_frozen['item code']==i]['Initial Inventory'].iloc[0])))
    except:
        model.xc[i,0].fix(0)
#
# model.initial_inv_frozen_Constraint = Constraint(model.P, rule = initial_inventory_frozen_product)
# key = 'api_32f781e7-5704-40a4-94ea-e99e314945cf'
# base_url = 'https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/'
# doc = DOcloud(base_url, key, verbose=True)
# model.symbolic_solver_labels = True
# model = model.write('imbalanced_part.lp',io_options={"symbolic_solver_labels": True})
#
# import glob
# file = glob.glob('imbalanced_part.lp')
# client = JobClient(base_url, key)
# resp = client.execute(input=file,output="results.json")
# model.solutions.store_to(results)
# resp.write(filename='results.json',format = 'json')
#model.pprint()
#print ('end****')
# resp = client.execute(input=file,output="imbalance_result.json", load_solution = True)
# result = json.loads(resp.solution.decode("utf-8"))
# print(result)
# for i,v in result.items():
#     for l,j in v.items():
#         for p in j:
#             print(p)
print("solver running")
# model.pprint()
# opt = SolverFactory("cplex")
#
# solver_manager = SolverManagerFactory('neos')
#
# results = solver_manager.solve(model, opt=opt)

# results.write()
opt = SolverFactory('cplex')
# opt.options["threads"] = 4
opt.options["mip_tolerance_mipgap"] = 0.05
opt.options["log"] = 1
results = opt.solve(model, tee=True, keepfiles = True)
model.load(results)
#
model.solutions.store_to(results)
results.write(filename='results.json',format = 'json')
# print(results)
# objective_value = results['Solution'][0]['Objective']['OBJ']['Value']
df = pd.read_excel(filename, sheetname="Demand_Fresh")
df = df.dropna(axis='columns', how='all')
column_len = len(df.loc[0])
products = products
with open('results.json') as data_file:
    data = json.load(data_file)
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

days = ['item code']+ [0] + days
# for i in range(1, column_len):
#     days.append('Day_'+str(i))
# print(xc_sol)
d_x = pd.DataFrame(0, index=products, columns=days[1:])
for i, j in list(itertools.product(model.P, model.T)):
    d_x[j][i] = model.x[i, j].value

d_x.columns = days[1:]
d_x.index = products
print(d_x, 'Production')
d_x.to_csv('Production.csv')

d_xc = pd.DataFrame(0, index=products, columns=days[1:])
for i, j in list(itertools.product(model.P, model.T)):
    d_xc[j][i] = model.xc[i, j].value

days_with_initial = days[1:]
d_xc.columns = days_with_initial
d_xc.index = products
d_xc.index.name = 'item code'
d_xc.to_csv('Frozen_Product_Production.csv')

d_vf = pd.DataFrame(0, index=products, columns=days[1:])
for i, j in list(itertools.product(model.P, model.T)):
    d_vf[j][i] = model.vf[i, j].value

d_vf.columns = days[1:]
d_vf.index = products
d_vf.index.name = 'item code'
d_vf.to_csv('Fresh_Product_Sold.csv')

d_vc = pd.DataFrame(0, index=products, columns=days[1:])
for i, j in list(itertools.product(model.P, model.T)):
    d_vc[j][i] = model.vc[i, j].value

d_vc.columns = days[1:]
d_vc.index = products
d_vc.index.name = 'item code'
d_vc.to_csv('Frozen_Product_Sold.csv')

d_Ic = pd.DataFrame(0, index=products, columns=days[1:])
for i, j in list(itertools.product(model.P, model.T)):
    d_Ic[j][i] = model.Ic[i, j].value

d_Ic.columns = days_with_initial
d_Ic.index = products
d_Ic.index.name = 'item code'
d_Ic.to_csv('Frozen_Product_Inventory.csv')

d_uf = pd.DataFrame(0, index=products, columns=days[1:])
for i, j in list(itertools.product(model.P, model.T)):
    d_uf[j][i] = model.uf[i, j].value

d_uf.columns = days[1:]
d_uf.index = products
d_uf.index.name = 'item code'
d_uf.to_csv('Unsatisfied_Fresh.csv')

d_uc = pd.DataFrame(0, index=products, columns=days[1:])
for i, j in list(itertools.product(model.P, model.T)):
    d_uc[j][i] = model.uc[i, j].value

d_uc.columns = days[1:]
d_uc.index = products
d_uc.to_csv('Unsatisfied_Frozen.csv')

df_two = pd.read_excel(filename, sheetname="Yield")
df_two = df_two.dropna(axis='columns', how='all')
products_two = products
cut_pattern = cut_pattern
bird_type = bird_type
with open('results.json') as data_file:
    data = json.load(data_file)
    solution = data['Solution'][1]['Variable']
    z_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'z[' in k}

z_df = pd.Series(z_sol, name='Order_Quantity')
z_df = pd.DataFrame(z_df)
z_df = z_df.reset_index()
i_1 = []
i_2 = []
i_3 = []
for i in z_df['index']:
    i_1.append(i[1])
    i_2.append(i[3])
    i_3.append(i[5])
del z_df['index']
z_df['I_1'] = i_1
z_df['I_2'] = i_2
z_df['I_3'] = i_3
z_df = pd.pivot_table(z_df, values='Order_Quantity', index=['I_2', 'I_1'], columns='I_3',aggfunc=sum).reset_index()

# z_df.columns = ['Carcass_Type', 'Cut_Type']+list(days)
z_df.to_csv('Pattern_Count_RT.csv')

ze_df = pd.Series(ze_sol, name='Order_Quantity')
ze_df = pd.DataFrame(ze_df)
ze_df = ze_df.reset_index()
i_1 = []
i_2 = []
i_3 = []
for i in ze_df['index']:
    i_1.append(i[2])
    i_2.append(i[5])
    i_3.append(i[6])
del ze_df['index']
ze_df['I_1'] = i_1
ze_df['I_2'] = i_2
ze_df['I_3'] = i_3
ze_df = pd.pivot_table(ze_df, values='Order_Quantity', index=['I_2', 'I_1'], columns='I_3', aggfunc=sum).reset_index()

ze_df.to_csv('Pattern_Count_OT.csv')
xf_df = pd.Series(x_sol, name='Order_Quantity')
xf_df = pd.DataFrame(xf_df)
xf_df = xf_df.reset_index()
i_1 = []
i_2 = []
i_3 = []
for i in xf_df['index']:
    i_1.append(i[2])
    i_2.append(i[4])
    i_3.append(i[6])
del xf_df['index']
xf_df['I_1'] = i_1
xf_df['I_2'] = i_2
xf_df['I_3'] = i_3
xf_df = pd.pivot_table(xf_df, values='Order_Quantity', index=['I_1', 'I_2'], columns='I_3', aggfunc=sum).reset_index()
xf_df.to_csv('Process_Fresh_OverHorizon.csv')
HA = pd.DataFrame({'Number':HA_sol})
HA.to_csv("Bird_Count.csv")
print(HA)

arr = []
for i,j,k in itertools.product(model.J, model.R, model.T):
    val = value(model.z[i,j,k])
    if val > 0:
        arr.append({'Cutting_pattern':i, 'Bird_Type':j, 'Day':k, 'count':round(val,2), 'keyfigure':'Pattern_Count_RT'})

for i,j,k in itertools.product(model.J, model.R, model.T):
    val = value(model.ze[i,j,k])
    if val > 0:
        arr.append({'Cutting_pattern':i, 'Bird_Type':j, 'Day':k, 'count':round(val,2), 'keyfigure':'Pattern_Count_OT'})
data_required = pd.DataFrame(arr)
print(data_required)
data_required.to_csv('production_count'+str(last_day)+'.csv',index = False)
profit = {}
selling = {}
freezing_cost = {}
holding_fresh = {}
holding_frozen = {}
penalty = {}
operating_cost ={}
cost = []
for t in model.T:
    cost.append({'Day':t,'Cost':sum(model.pf[i]*model.vf[i, t].value + model.pc[i]*model.vc[i, t].value for i in model.P),'keyfigure':'Selling_Price'})
    cost.append({'Day':t,'Cost':sum(model.b*model.xc[i, t].value for i in model.P),'keyfigure':'freezing_cost'})
    cost.append({'Day':t,'Cost':sum(model.hc[i]*model.Ic[i, t].value for i in model.P),'keyfigure':'holding_frozen'})
    cost.append({'Day':t,'Cost':sum(sum(model.hf[i]*model.xf[i, t, t+l].value for l in model.L[i] if l >= last_day) for i in model.P),'keyfigure':'holding_fresh'})
    cost.append({'Day':t,'Cost':sum(model.sf[i]*model.uf[i, t].value - model.sc[i]*model.uc[i, t].value for i in model.P),'keyfigure':'penalty'})
    cost.append({'Day':t,'Cost':sum(sum(model.c[j]*model.z[j, r, t].value + model.ce[j]*model.ze[j, r, t].value for j in model.J) for r in model.R),'keyfigure':'operating_cost'})
    selling[t] = sum(model.pf[i]*model.vf[i, t].value + model.pc[i]*model.vc[i, t].value for i in model.P)
    freezing_cost[t] =sum(model.b*model.xc[i, t].value for i in model.P)
    holding_frozen[t] = sum(model.hc[i]*model.Ic[i, t].value for i in model.P)
    holding_fresh[t] = sum(sum(model.hf[i]*model.xf[i, t, t+l].value for l in model.L[i] if l >= last_day) for i in model.P)
    penalty[t]=sum(model.sf[i]*model.uf[i, t].value - model.sc[i]*model.uc[i, t].value for i in model.P)
    operating_cost[t]=sum(sum(model.c[j]*model.z[j, r, t].value + model.ce[j]*model.ze[j, r, t].value for j in model.J) for r in model.R)
    profit[t] = selling[t]-freezing_cost[t]-holding_fresh[t]-holding_frozen[t]-penalty[t]-operating_cost[t]
    cost.append({'Day':t,'Cost':profit[t],'keyfigure':'profit'})
data_required = pd.DataFrame(cost)
data_required.to_csv('cost'+str(last_day)+'.csv',index = False)