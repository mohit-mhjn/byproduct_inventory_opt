from __future__ import division
import pandas as pd
from pyomo.environ import *
import os
## MILP model for planning at operative level in meat packing plant
## Authors : Victor M. Albornoz, Marcela G-Araya, Matias C Gripe and Sara V. Radriguez

os.chdir('D:/SNO/PP_Model/')
df = pd.read_excel('Model_Input.xlsm',sheetname="Demand")
demand = pd.DataFrame(0, index=range(1,len(df)+1), columns=range(1,len(df.loc[0])))

for i in range(0,len(df)):
    for j in range(1,len(df.loc[0])):
        demand.loc[i+1][j]=df.loc[i][j]
print("Demand Matrix [in columns = Days(Time Horizon) and rows = Products]")
print(demand)
model = ConcreteModel()
set_I = list(range(len(df)+1))
set_K = list(range(len(df.loc[0])))
print(set_K)
set_T = set_K[1:]
print("Set of Time Horizon :", set_T)
set_P = set_I[1:]
print("Set of Products :",set_P)
model.T = Set(initialize =set_T, ordered = True)     # Days for Demand
model.P = Set(initialize =set_P, ordered = True)     # No. of SKUs
model.I = Set(initialize = set_I, ordered = True)
model.K = Set(initialize = set_K, ordered = True)
def demand1(model,i,j):
    return demand[int(j)][int(i)]
model.d = Param(model.S, model.T, initialize=demand1)  # demand sku wise and day wise

model.i = Var(model.S, model.K, domain=NonNegativeIntegers)      # inventory SKUs wise and day wise
model.R = RangeSet(1,4)
model.x = Var(model.R, model.T, domain=NonNegativeIntegers)      # no of birds per cut-type per day wise


#pp= pd.read_csv('processing_cost.csv')
# def cost(model,r):
# 	return int(pp.iloc[r-1][1])
# model.c = Param(model.R,initialize = cost)
# model.a = Param(model.R, model.T)

def objective_function(model):
    return summation(model.i)
#	return sum(sum(model.x[r,t] * model.c[r] for r in model.R) for t in model.T)

model.OBJ = Objective(rule = objective_function, sense = minimize)

def initial_inventory(model,s):
     return model.i[s,0] ==0
model.initial_inv_Constraint = Constraint(model.S, rule = initial_inventory)

# def initial_demand(model,s):
     # return model.d[s,0] ==0
# model.initial_demand_const = Constraint(model.S, rule = initial_demand)

def inventory_balance_ribs(model,t):
   return model.i[1,t-1] + 2*model.x[3,t] + 2*model.x[4,t] - model.d[1,t] == model.i[1,t]
model.inv_ribs = Constraint(model.T, rule= inventory_balance_ribs)

def inventory_balance_keel(model,t):
   return model.i[2,t-1] + model.x[3,t] + model.x[4,t] - model.d[2,t] == model.i[2,t]
model.inv_keel = Constraint(model.T,  rule= inventory_balance_keel)

def inventory_balance_legs(model,t):
   return model.i[3,t-1] +  2*model.x[1,t] + 2*model.x[3,t] - model.d[3,t] == model.i[3,t]
model.inv_legs = Constraint(model.T, rule= inventory_balance_legs)

def inventory_balance_thighs(model,t):
   return model.i[4,t-1] + 2*model.x[2,t] + 2*model.x[4,t] - model.d[4,t] == model.i[4,t]
model.inv_thighs = Constraint(model.T,  rule= inventory_balance_thighs)

def inventory_balance_drumsticks(model,t):
   return model.i[5,t-1] + 2*model.x[2,t] + 2*model.x[4,t] - model.d[5,t] == model.i[5,t]
model.inv_drumsticks = Constraint(model.T, rule= inventory_balance_drumsticks)

def inventory_balance_wings(model,t):
   return model.i[6,t-1] +  2*model.x[1,t] + 2*model.x[2,t] + 2*model.x[3,t] + 2*model.x[4,t] - model.d[6,t] == model.i[6,t]
model.inv_wings = Constraint(model.T, rule= inventory_balance_wings)

def inventory_balance_breasts(model,t):
   return model.i[7,t-1] + 2*model.x[1,t] + 2*model.x[2,t] - model.d[7,t] == model.i[7,t]
model.inv_breasts = Constraint(model.T, rule= inventory_balance_breasts)

# def supply(model,r,t):
	# return model.x[r,t] <= model.a[r,t]
# model.supply_cons = Constraint(model.R,model.T, rule = supply)
