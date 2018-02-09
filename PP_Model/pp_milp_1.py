from __future__ import division

from pyomo.core.base import ConcreteModel, Param, Set, Var, Constraint, Objective, \
    NonNegativeIntegers, NonNegativeReals, maximize
from pyomo.core.base import IntegerSet
from pyomo.environ import *
import pandas as pd
import os

# MIP formulation from A Mixed Integer Linear Program for Operational Planning Meat Planing Plant
# AVIOTA
# Rishikesh
# modified for different shelf life's
model = ConcreteModel()

os.chdir('/home/rishi/Downloads/temp_projects/SNO/PP_Model/')

demand_fresh = pd.read_excel('Model_Input.xlsm', sheetname="Demand_Fresh")
demand_fresh_df = pd.DataFrame(0, index=range(1, len(demand_fresh) + 1), columns=range(1, len(demand_fresh.loc[0])))
for i in range(0, len(demand_fresh)):
    for j in range(1, len(demand_fresh.loc[0])):
        demand_fresh_df.loc[i + 1][j] = demand_fresh.loc[i][j]


def df(model, i, j):
    return demand_fresh_df[int(j)][int(i)]


demand_frozen = pd.read_excel('Model_Input.xlsm', sheetname="Demand_Frozen")
demand_frozen_dc = pd.DataFrame(0, index=range(1, len(demand_frozen) + 1), columns=range(1, len(demand_frozen.loc[0])))
for i in range(0, len(demand_frozen)):
    for j in range(1, len(demand_frozen.loc[0])):
        demand_frozen_dc.loc[i + 1][j] = demand_frozen.loc[i][j]


def dc(model, i, j):
    return demand_frozen_dc[int(j)][int(i)]


##pattern_info = pd.read_excel('Model_Input.xlsm',sheetname="Pattern_Information")
selling_price_fresh = pd.read_excel('Model_Input.xlsm', sheetname='Selling_Price_Fresh')


def pf(model, i):
    return float(selling_price_fresh.iloc[i - 1][1])


selling_price_frozen = pd.read_excel('Model_Input.xlsm', sheetname='Selling_Price_Frozen')


def pc(model, i):
    return float(selling_price_frozen.iloc[i - 1][1])


holding_price_fresh = pd.read_excel('Model_Input.xlsm', sheetname='Holding_Cost_Fresh')


def hf(model, i):
    return float(selling_price_fresh.iloc[i - 1][1])


holding_price_frozen = pd.read_excel('Model_Input.xlsm', sheetname='Holding_Cost_Frozen')


def hc(model, i):
    return float(holding_price_frozen.iloc[i - 1][1])


penalty_fresh = pd.read_excel('Model_Input.xlsm', sheetname='Penalty_Fresh')


def sf(model, i):
    return float(penalty_fresh.iloc[i - 1][1])


penalty_frozen = pd.read_excel('Model_Input.xlsm', sheetname='Penalty_Frozen')


def sc(model, i):
    return float(penalty_frozen.iloc[i - 1][1])


operational_cost = pd.read_excel('Model_Input.xlsm', sheetname='Operational_Cost')


def c(model, j):
    return float(operational_cost.iloc[j - 1][1])


operational_cost_overtime = pd.read_excel('Model_Input.xlsm', sheetname='Operational_Cost_Overtime')


def ce(model, j):
    return float(operational_cost_overtime.iloc[j - 1][1])


available_hour_regular = pd.read_excel('Model_Input.xlsm', sheetname='Available_Hour_Regular')


def Ta(model, t):
    return float(available_hour_regular.iloc[t - 1][1])


available_hour_overtime = pd.read_excel('Model_Input.xlsm', sheetname='Available_Hour_Overtime')


def Te(model, t):
    return float(available_hour_overtime.iloc[t - 1][1])


common_data = pd.read_excel('Model_Input.xlsm', sheetname='Common_Data')


def Wf(model):
    return common_data["Value"][0]


def Wc(model):
    return common_data["Value"][1]


# def Ta(model,t):
#     return common_data["Value"][2]
# def Te(model):
#     return common_data["Value"][3]
def b(model):
    return common_data["Value"][4]


def F(model):
    return int(common_data["Value"][5])


def tau(model):
    return int(common_data["Value"][6])


def delta(model):
    return common_data["Value"][7]


# def H(model):
#    return int(common_data["Value"][9])

# shelf_life = int(common_data["Value"][8])

shelf_life = pd.read_excel('Model_Input.xlsm', sheetname='Shelf_Life')
shelf_lif = max(shelf_life['Shelf_Life'])

def L(model,i):
    L = list(j for j in range(shelf_life.iloc[i-1][1]))
    L = L[1:]
    return L
# Ld = list(i for i in range(shelf_life + 1))
# L = Ld[1:]
def Ld(model,i):
    Ld = list(j for j in range(shelf_life.iloc[i-1][1]))
    Ld = Ld[1:-1]
    return Ld


operational_time = pd.read_excel('Model_Input.xlsm', sheetname='Operation_Time')


def t(model, j):
    return float(operational_time.iloc[j - 1][1])


y = pd.read_excel('Model_Input.xlsm', sheetname='Yield', )
products = list(i for i in range(1,len(y['Product'].unique())+1))
cut_pattern= list(i for i in range(1,len(y['Cutting Pattern'].unique())+1))
bird_type = list(i for i in range(1, len(y.columns) + 1))
y.columns = bird_type
# y.index.set_levels([products, cut_pattern], inplace=True)


def yd(model, i, j, r):
    print(y[r][i][j])
    return float(y[r][i][j])


bird_type_ratio = pd.read_excel('Model_Input.xlsm', sheetname='Bird_Type_Ratio')


def alpha(model, r):
    return float(bird_type_ratio.iloc[r - 1][1])


days = list(i for i in range(1, len(demand_frozen.loc[0])))
print(days)
availability = pd.read_excel('Model_Input.xlsm', sheetname='Availability')


def H(model, r):
    return int(availability.iloc[r - 1][1])


cutting_pattern = pd.read_excel('Model_Input.xlsm', sheetname='Cutting_Pattern')
cutting_pattern = cutting_pattern.fillna(0)
cp = []
for i in cutting_pattern.itertuples():
    cp.append((list(i[2:])))
for i in cp:
    if 0 in i:
        i.remove(0)
cp = [[int(j) for j in i] for i in cp]
sections = list(s for s in range(1, len(cp) + 1))


def Jk(model, k):
    return cp[int(k - 1)]


#  F = Fresh and C = Frozen
# print(sections)
model.T = Set(initialize=days, ordered=True)  # planning horizon
model.J = Set(initialize=cut_pattern, ordered=True)  # cutting patterns
model.L = Set(model.J, initialize=Ld, ordered=True)  # shell life for fresh products
model.H = Param(model.T, initialize=H)  # min no of avaiable carcasses of type r in all planning horizon
model.K = Set(initialize=sections)  # no of sections
model.Jk = Set(model.K, initialize=Jk)  # cutting pattern
model.R = Set(initialize=bird_type, ordered=True)  # type of carcasses
model.P = Set(initialize=products, ordered=True)  # products
# model.alpha = Param(model.R, initialize=alpha)  # proportion of carcasses of type r
model.yd = Param(model.P, model.J, model.R, initialize=yd)  # yield of product i in pattern j on carcasses of type r
model.pf = Param(model.P, initialize=pf)  # SP of fresh product
model.pc = Param(model.P, initialize=pc)  # SP of frozen product
model.c = Param(model.J, initialize=c)  # operational cost of pattern j
model.ce = Param(model.J, initialize=ce)  # operational cost of pattern j in OT
model.b = Param(initialize=b)  # freezing cost /kg
model.hf = Param(model.P, initialize=hf)  # HC F
model.hc = Param(model.P, initialize=hc)  # HC Frozen
model.sf = Param(model.P, initialize=sf)  # penality for unstatisfed fresh
model.sc = Param(model.P, initialize=sc)  # penality for unstatisfed frozen
model.F = Param(initialize=F)  # Freezing tunnel capacity at t
model.df = Param(model.P, model.T, initialize=df)  # Demand of fresh products
model.dc = Param(model.P, model.T, initialize=dc)  # Demand of frozen products
model.tau = Param(initialize=tau)  # Freezing Process duration
model.Wf = Param(initialize=Wf)  # warehouse capacity
model.Wc = Param(initialize=Wc)  # warehouse capacity
model.t = Param(model.J, initialize=t)  # cutting operation for pattern j
model.Ta = Param(model.T, initialize=Ta)  # avaiable work hours in each time period
model.Te = Param(model.T, initialize=Te)  # avaiable overtime hours
model.delta = Param(initialize=delta)  # auxilary pattern for better control the available carcass
model.Ld = Set(model.J,initialize=Ld, ordered=True)
# Extra Parameters to handle computed indices
td = list(range((days[0] - shelf_life), 1)) + days + list(range(days[-1] + 1, days[-1] + shelf_life + 1))
print(Ld)
model.Td = Set(initialize=td, ordered=True)

model.x = Var(model.P, model.T, domain=NonNegativeReals)  # total quantity of product i to be processed in period t
model.xf = Var(model.P, model.Td, model.Td,
               domain=NonNegativeReals)  # quantity of fresh product i to process in period t to be sold at t'
model.z = Var(model.J, model.R, model.T,
              domain=NonNegativeIntegers)  # no of times to use pattern j on carcass r in period t in RT
model.ze = Var(model.J, model.R, model.T,
               domain=NonNegativeIntegers)  # no of times to use pattern j on carcass r in period t in OT
model.xc = Var(model.P, model.Td, domain=NonNegativeReals)  # quantity of frozen product i to process in t
model.vf = Var(model.P, model.T, domain=NonNegativeReals)  # quantity of fresh product i to be sold in t
model.vc = Var(model.P, model.T, domain=NonNegativeReals)  # quantity of frozen product i to be sold in t
model.Ic = Var(model.P, model.Td, domain=NonNegativeReals)  # quantity of frozen product i to be hold in t
model.uf = Var(model.P, model.T, domain=NonNegativeReals)  # unsatisfied demand of fresh product i in t
model.uc = Var(model.P, model.T, domain=NonNegativeReals)  # unsatisfied demand of frozen product i in t
model.HA = Var(model.T, domain=NonNegativeIntegers)  # No of carcasses to be processed in period t


def objective_function(model):
    return sum(
        sum(model.pf[i] * model.vf[i, t] + model.pc[i] * model.vc[i, t] for i in model.P) for t in model.T) - sum(
        sum(model.b * model.xc[i, t] for i in model.P) for t in model.T) - sum(
        sum(model.hc[i] * model.Ic[i, t] for i in model.P) for t in model.T) - sum(
        sum(sum(model.hf[i] * model.xf[i, t, t + l] for i in model.P) for t in model.T) for l in model.L) - sum(
        sum(model.sf[i] * model.uf[i, t] - model.sc[i] * model.uc[i, t] for i in model.P) for t in model.T) - sum(
        sum(sum(model.c[j] * model.z[j, r, t] + model.ce[j] * model.ze[j, r, t] for j in model.J) for r in model.R) for
        t in model.T)


model.OBJ = Objective(rule=objective_function, sense=maximize)


def carcass_availabiltiy(model, r, t, k):
    return sum(model.z[j, r, t] + model.ze[j, r, t] for j in model.Jk[k]) == model.alpha[r] * model.HA[t]


model.A2Constraint = Constraint(model.R, model.T, model.K, rule=carcass_availabiltiy)


def carcass_limit(model, t):
    return model.HA[t] <= model.H[t]


model.A3Constraint = Constraint(model.T, rule=carcass_limit)


def cutting_pattern(model, i, t):
    return model.x[i, t] == sum(
        sum(model.yd[i, j, r] * (model.z[j, r, t] + model.ze[j, r, t]) for j in model.J) for r in model.R)


model.A4Constraint = Constraint(model.P, model.T, rule=cutting_pattern)


def avaiable_daily_wh(model, t):
    return sum(sum(model.z[j, r, t] * model.t[j] for j in model.J) for r in model.R) <= model.Ta[t]


model.A5Constraint = Constraint(model.T, rule=avaiable_daily_wh)


def avaiable_daily_ot(model, t):
    return sum(sum(model.ze[j, r, t] * model.t[j] for j in model.J) for r in model.R) <= model.Te[t]


model.A6Constraint = Constraint(model.T, rule=avaiable_daily_ot)


def fresh_frozne_balance(model, i, t):
    return model.x[i, t] == sum(model.xf[i, t, t + l] for l in model.Ld[i]) + model.xc[i, t]


model.A7Constraint = Constraint(model.P, model.T, rule=fresh_frozne_balance)


def fresh_product_sold(model, i, t):
    return model.vf[i, t] == sum(model.xf[i, t - l, t] for l in model.Ld if t - l > 0)


model.A8Constraint = Constraint(model.P, model.T, rule=fresh_product_sold)


def frozen_product_sold(model, i, t):
    if t > 0:
        return model.vc[i, t] == model.Ic[i, t - 1] + model.xc[i, t - model.tau] - model.Ic[i, t]


model.A9Constraint = Constraint(model.P, model.T, rule=frozen_product_sold)


def demand_frozen_product(model, i, t):
    return model.vc[i, t] + model.uc[i, t] == model.dc[i, t]


model.A10Constraint = Constraint(model.P, model.T, rule=demand_frozen_product)


def demand_fresh_product(model, i, t):
    return model.vf[i, t] + model.uf[i, t] == model.df[i, t]


model.A11Constraint = Constraint(model.P, model.T, rule=demand_fresh_product)


def freezing_capacity(model, t):
    return sum(sum(model.xc[i, t] for t in range(t - model.tau, t + 1) if t > 0) for i in model.P) <= model.F


model.A12Constraint = Constraint(model.T, rule=freezing_capacity)


def fresh_product_wh_cap(model, t):
    return sum(
        sum(sum(model.xf[i, t - l, td] for td in range(t, t - l + len(L))) for l in model.L if t - l > 0) for i in
        model.P) <= model.Wf


model.A13Constraint = Constraint(model.T, rule=fresh_product_wh_cap)


def frozen_product_wh_cap(model, t):
    if t <= 0:
        return model.Ic[i, t] == 0
    else:
        return sum(model.Ic[i, t] for i in model.P) <= model.Wc


model.A14Constraint = Constraint(model.T, rule=frozen_product_wh_cap)


def initial_inventory(model, i):
    return model.Ic[i, 0] == 0


model.initial_inv_Constraint = Constraint(model.P, rule=initial_inventory)

initial_inv_frozen = pd.read_excel('Model_Input.xlsm', sheetname='Initial_Inventory')


def initial_inventory_frozen_product(model, i):
    return model.xc[i, 0] == int(initial_inv_frozen.iloc[i - 1][1])


model.initial_inv_frozen_Constraint = Constraint(model.P, rule=initial_inventory_frozen_product)

# model.pprint()
opt = SolverFactory('cbc')
##opt.options["sec"] = 10
opt.options["log"] = 1
results = opt.solve(model)
# model.load(results)

model.solutions.store_to(results)
results.write(filename='results.json', format='json')

import csv
import json
import re

df = pd.read_excel('Model_Input.xlsm', sheetname="Demand_Fresh")
column_len = len(df.loc[0])
products = df['days']
with open('results.json') as data_file:
    data = json.load(data_file)
    solution = data['Solution'][1]['Variable']
    HA_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'HA' in k}
    Ic_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'Ic' in k}
    x_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'x[' in k}
    uc_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'uc' in k}
    uf_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'uf' in k}
    vc_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'vc' in k}
    vf_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'vf' in k}
    xc_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'xc' in k}
    xf_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'xf' in k}
    z_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'z[' in k}
    ze_sol = {k[1:]: v['Value'] for k, v in solution.items() if 'ze' in k}

days = []
for i in range(1, column_len):
    days.append('Day_' + str(i))

import pandas as pd

d_x = pd.DataFrame(0, index=range(1, 8), columns=range(1, column_len))
for x, v in x_sol.items():
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    outputString = " ".join((words))
    d_x[int(words[0][2])][int(words[0][0])] = v

d_x.columns = days
d_x.index = products
d_x.to_csv('Production.csv')
d_xc = pd.DataFrame(0, index=range(1, 8), columns=range(0, column_len))
for x, v in xc_sol.items():
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    outputString = " ".join((words))
    d_xc[int(words[0][2])][int(words[0][0])] = v

days_with_initial = ['Day_0'] + days
d_xc.columns = days_with_initial
d_xc.index = products

d_xc.to_csv('Frozen_Product_Production.csv')

d_vf = pd.DataFrame(0, index=range(1, 8), columns=range(1, column_len))
for x, v in vf_sol.items():
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    outputString = " ".join((words))
    d_vf[int(words[0][2])][int(words[0][0])] = v

d_vf.columns = days
d_vf.index = products

d_vf.to_csv('Fresh_Product_Sold.csv')

d_vc = pd.DataFrame(0, index=range(1, 8), columns=range(1, column_len))
for x, v in vc_sol.items():
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    outputString = " ".join((words))
    d_vc[int(words[0][2])][int(words[0][0])] = v

d_vc.columns = days
d_vc.index = products

d_vc.to_csv('Frozen_Product_Sold.csv')

d_Ic = pd.DataFrame(0, index=range(1, 8), columns=range(0, column_len))
for x, v in Ic_sol.items():
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    outputString = " ".join((words))
    d_Ic[int(words[0][2])][int(words[0][0])] = v

d_Ic.columns = days_with_initial
d_Ic.index = products

d_Ic.to_csv('Frozen_Product_Hold.csv')

d_uf = pd.DataFrame(0, index=range(1, 8), columns=range(1, column_len))
for x, v in uf_sol.items():
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    outputString = " ".join((words))
    d_uf[int(words[0][2])][int(words[0][0])] = v

d_uf.columns = days
d_uf.index = products

d_uf.to_csv('Unsatisfied_Fresh.csv')

d_uc = pd.DataFrame(0, index=range(1, 8), columns=range(1, column_len))
for x, v in uc_sol.items():
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    outputString = " ".join((words))
    d_uc[int(words[0][2])][int(words[0][0])] = v

d_uc.columns = days
d_uc.index = products

d_uc.to_csv('Unsatisfied_Frozen.csv')

from pprint import pprint

df_two = pd.read_excel('Model_Input.xlsm', sheetname="Yield")
# print(df_two.columns[0])
products_two = df_two.index.levels[0]
cut_pattern = df_two.index.levels[1]
bird_type = df_two.columns
print('product', products_two, 'cut', cut_pattern, 'days', days)
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
z_df = pd.pivot_table(z_df, values='Order_Quantity', index=['I_2', 'I_1'], columns='I_3', aggfunc=sum).reset_index()

z_df.columns = ['Carcass_Type', 'Cut_Type'] + list(days)
print(z_df['Carcass_Type'])
z_df.to_csv('Pattern_Count_RT.csv')

ze_df = pd.Series(ze_sol, name='Order_Quantity')
ze_df = pd.DataFrame(ze_df)
ze_df = ze_df.reset_index()
i_1 = []
i_2 = []
i_3 = []
for i in ze_df['index']:
    i_1.append(i[2])
    i_2.append(i[4])
    i_3.append(i[6])
del ze_df['index']
ze_df['I_1'] = i_1
ze_df['I_2'] = i_2
ze_df['I_3'] = i_3
ze_df = pd.pivot_table(ze_df, values='Order_Quantity', index=['I_2', 'I_1'], columns='I_3', aggfunc=sum).reset_index()

z_df.columns = ['Carcass_Type', 'Cut_Type'] + list(days)
ze_df.to_csv('Pattern_Count_OT.csv')

xf_df = pd.Series(xf_sol, name='Order_Quantity')
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

z_df.columns = ['Carcass_Type', 'Day'] + list(days)
xf_df.to_csv('Production_Fresh_OverHorizon.csv')

print(HA_sol)
HA = pd.Series(HA_sol)
HA.to_csv("Bird_Count.csv")
