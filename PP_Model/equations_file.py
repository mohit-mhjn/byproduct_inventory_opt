from pyomo.core.base import ConcreteModel, Param, Set, Var, Constraint, Objective,\
    NonNegativeIntegers, NonNegativeReals, maximize
import json
import time
start_time = time.time()
from docloud.job import JobClient
from pyomo.environ import *
import pandas as pd
import os
import numpy as np
import sys
import itertools
from solutionmethod import solve_model

directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

filename = "table_yield.xlsx"

demand_fresh = pd.read_excel(filename,sheet_name="Demand_Fresh")
demand_fresh = demand_fresh.dropna(axis='columns', how='all')
demand_fresh = demand_fresh.round(2)
products = list(set(demand_fresh['item_code']))
days = list(demand_fresh.columns[1:])

y = pd.read_excel(filename, sheet_name='yield2')
y = y[(y.yield_p > 0)]
y['yield_p']= y['yield_p'].fillna(value=0)
y = y[y['item_code'].isin(products)]
y["yield_p"] = y["yield_p"].apply(lambda x: round(x,2))
y["carcass"] = y["carcass"].apply(lambda x: round(x,2))    # >> Temporary Fix
y["section"] = y["section"].apply(lambda x: [int(i) for i in str(x)])

bird_type = list(set(y["carcass"]))
cut_pattern = list(set(y['cutting_pattern']))
prod_group = list(set(y['product_group']))

cp = pd.read_excel(filename, sheet_name = 'cutting_pattern')
cp = cp.filter(items = ['section', 'cutting_pattern'])
cp = cp[cp['cutting_pattern'].isin(cut_pattern)]
sections = list(set(cp['section']))

bird_type_ratio = pd.read_excel(filename, sheet_name='bird_type_ratio')
bird_type_ratio = bird_type_ratio.dropna(axis='columns', how='all')

# print (products)
# print (bird_type)
# print (cut_pattern)
# print (sections)
# print (prod_group)
# print (days)

model = ConcreteModel()

model.T = Set(initialize= days, ordered=True)     # planning horizon
model.J = Set(initialize= cut_pattern, ordered = True) # cutting patterns
model.P = Set(initialize = products, ordered = True)
model.PG = Set(initialize= prod_group, ordered = True)     # products
model.H = Param(model.T, initialize = 100000)         # max no of avaiable carcasses in all planning horizon
model.K = Set(initialize = sections, ordered = True)             # no of sections
model.R = Set(initialize=bird_type, ordered = True)     # type of carcasses

def cp_to_sec(model,j):
    global cp
    df_temp = cp[(cp.cutting_pattern == j)]
    return list(set(df_temp['section']))
model.Jk = Set(model.J, initialize = cp_to_sec)

def sec_to_cp(model,k):
    global cp
    df_temp = cp[(cp.section == k)]
    return list(set(df_temp['cutting_pattern']))
model.Kj = Set(model.K, initialize = sec_to_cp)

def indx_k_j_gen(model):
    my_set = set()
    for k in model.K:
        for j in model.Kj[k]:
            my_set.add((k,j))
    return my_set
model.indx_k_j = Set(dimen = 2, initialize = indx_k_j_gen)

def cp_sec_to_p(model,k,j):
    global y
    df_tmp = y[y.section.map(set([k]).issubset)]
    df_tmp = df_tmp[(df_tmp.cutting_pattern == j)]
    if not df_tmp.empty:
        return list(set(df_tmp['item_code']))
    else:
        return []
model.Jp = Set(model.indx_k_j, initialize = cp_sec_to_p)

section_pg_set = set()
def cp_sec_to_pg1(model,k,j):
    global section_pg_set
    global y
    df_tmp = y[y.section.map(set([k]).issuperset)]
    df_tmp = df_tmp[(df_tmp.cutting_pattern == j)]
    if not df_tmp.empty:
        lst = set(df_tmp['product_group'])
        section_pg_set = section_pg_set.union(lst)
        return lst
    else:
        return ()
model.Jpg1 = Set(model.indx_k_j, initialize = cp_sec_to_pg1)

def cp_sec_to_pg2(model,k,j):
    global section_pg_set
    global y
    df_tmp = y[y.section.map(set([k]).issubset)]
    df_tmp = df_tmp[(df_tmp.cutting_pattern == j)]
    if not df_tmp.empty:
        lst = set(df_tmp['product_group']) - set(model.Jpg1[k,j])
        return lst
    else:
        return ()
model.Jpg2 = Set(model.indx_k_j, initialize = cp_sec_to_pg2)

def pgroup_p(model,i):
    global y
    df_temp = y[(y.product_group == i)]
    return set(df_temp['item_code'])
model.Pn = Set(model.PG, initialize = pgroup_p)

def p_pgroup(model,i):
    global y
    df_temp = y[(y.item_code == i)]
    return [df_temp.iloc[0]['product_group']]
model.PGn = Set(model.P, initialize = p_pgroup)

def section_p(model,k):         # Collects SKU's Falling under each section
    global y
    df_tmp = y[y.section.map(set([k]).issubset)]
    return set(df_tmp['item_code'])
model.Kp = Set(model.K, initialize = section_p)

def alpha(model, r):
    df_temp = bird_type_ratio[(bird_type_ratio.bird_type == r)]
    v = float(df_temp.iloc[0]['ratio'])
    return round(v,2)
model.alpha = Param(model.R, initialize = alpha)    # proportion of carcasses of type r

def indx_i_j_r_gen(model):
    global y
    my_set = set()
    for indx,row in y.iterrows():
        my_set.add((row["item_code"],row["cutting_pattern"],row["carcass"]))
    return my_set
model.indx_i_j_r = Set(dimen = 3, initialize = indx_i_j_r_gen)

def yd_gen(model,i,j,r):
    global y
    df_temp = y[(y.item_code == i) & (y.cutting_pattern == j) & (y.carcass == r)]
    return df_temp.iloc[0]['yield_p']*df_temp.iloc[0]['n_parts']
model.yd = Param(model.indx_i_j_r, initialize = yd_gen)

def demand_fresh_gen(model,t,i):
    global demand_fresh
    df_temp = float(demand_fresh[(demand_fresh.item_code) == i].iloc[0][t])
    return df_temp
model.df = Param(model.T,model.P, initialize = demand_fresh_gen) #Demand of fresh products

sku_to_jr_map = {i:set() for i in model.P}
jr_to_sku_map = {(j,r):set() for j in model.J for r in model.R}

def indx_r_j_gen(model):
    my_set = set()
    for i in model.indx_i_j_r:
        my_set.add((i[2],i[1]))
        jr_to_sku_map[(i[1],i[2])].add(i[0])
        sku_to_jr_map[i[0]].add((i[1],i[2]))
        #j_to_rsku_map[i[1]].add((i[0],i[2]))
    return my_set
model.indx_r_j = Set(dimen = 2, initialize = indx_r_j_gen)

def indx_r_k_j_gen(model):
    my_set = set()
    for k,j0 in model.indx_k_j:
        lst = [r for r,j1 in model.indx_r_j if j1 == j0]
        for rn in lst:
            my_set.add((rn,k,j0))
    return my_set
model.indx_r_k_j = Set(dimen = 3, initialize = indx_r_k_j_gen)

############# Variable Definition  ######################################

model.z = Var(model.T, model.R, domain=NonNegativeIntegers)   # no of carcass r processed in period T
model.zk = Var(model.T, model.R, model.K, domain = NonNegativeIntegers) # Number of k section produced by carcass type R
model.zkj = Var(model.T, model.indx_r_k_j, domain = NonNegativeIntegers) # Number of times cutting pattern j is applied on section k of carcass type R
model.zj = Var(model.T, model.indx_r_j, domain = NonNegativeIntegers)

model.x = Var(model.T, model.P, domain=NonNegativeReals)   # total quantity of product i to be processed in period t
model.xjr = Var(model.T, model.P, model.indx_r_j, domain = NonNegativeReals)
#model.xjkr = Var(model.T, model.P, model.R, indx_k_j, domain = NonNegativeReals) # total quantiy of sku i produced by applying cutting pattern J on section K of carcass type R

##########################################################################################

def carcass_availability(model,t,r):
    return model.z[t,r] <= model.H[t]*model.alpha[r]
model.A0Constraint = Constraint(model.T, model.R, rule = carcass_availability)

def carcass_to_section(model,t,r,k):
    return model.zk[t,r,k] == model.z[t,r]
model.A1Constraint = Constraint(model.T, model.R, model.K, rule = carcass_to_section)

def carcass_in_cutpattern(model,t,r,k):
    lst = [j0 for r0,k0,j0 in model.indx_r_k_j if r0 == r and k0 == k and j0 in model.Kj[k]]
    return sum(model.zkj[t,r,k,j] for j in lst) == model.zk[t,r,k]
model.A2Constraint = Constraint(model.T, model.R, model.K, rule = carcass_in_cutpattern)

def cutting_pattern_count_gen(model,t,r,k,j):
    return model.zj[t,r,j] >= model.zkj[t,r,k,j]
model.A3Constraint = Constraint(model.T, model.indx_r_k_j, rule = cutting_pattern_count_gen)

def cutting_pattern_count_limiter(model,t,r,j):  # Will become redundant if z is in min(obj)
    return model.zj[t,r,j] <= sum(model.zkj[t,r,k,j] for k in model.Jk[j])
model.A4Constraint = Constraint(model.T, model.indx_r_j, rule = cutting_pattern_count_limiter)

def cutting_pattern_balancer(model,t,r,k,j):
    return model.zkj[t,r,k,j] == model.zj[t,r,j]
model.A5Constraint = Constraint(model.T, model.indx_r_k_j, rule = cutting_pattern_balancer)

## map p >> to >> K

## If carcass r is cut by cutting pattern J it will produce SKU i with yield 'y' and collection of section of SKU
# y >>
#x[i,j,r,k,t]
# Prepare Yield Sheet for J<K<R Index defining an SKU f=coming form section A
#what if sku i is mapped with multiple k
# X1 = [1]
# X2 = [1,2,3]

def sku_prod_total(model,t,i):
    global sku_to_jr_map
    return model.x[t,i] == sum(model.xjr[t,i,r,j] for j,r in sku_to_jr_map[i])
model.A6Constraint = Constraint(model.T, model.P, rule = sku_prod_total)

def map_k_j_to_pg(model):
    my_set = set()
    for r,k,j in model.indx_r_k_j:
        in_PG = model.Jpg1[k,j]
        if in_PG == []:
            my_set.add((r,k,j,"WholeB"))
        else:
            for grp in in_PG:
                my_set.add((r,k,j,grp))
    return my_set
model.indx_r_k_j_pg = Set(dimen = 4, initialize = map_k_j_to_pg)

# model.Jpg1.pprint()
# model.Jpg2.pprint()
# model.Pn.pprint()

def sku_conversion(model,t,r,k,j,pg):
    all_items = jr_to_sku_map[(j,r)]
    items1 = set()
    if not pg == "WholeB":
        items1 = all_items.intersection(model.Pn[pg])
    pg2 = model.Jpg2[k,j]
    my_set = set()
    for grp in pg2:
        my_set = my_set.union(model.Pn[grp])
    items2 = set()
    if my_set:
        items2 = all_items.intersection(my_set)
    items_req = items1.union(items2)
    if not items_req:
        # print ("skipped for:", (r,k,j,pg))
        return Constraint.Skip
    return model.zkj[t,r,k,j] == sum(model.xjr[t,i,r,j]/model.yd[i,j,r] for i in items_req)
model.A7Constraint = Constraint(model.T, model.indx_r_k_j_pg, rule = sku_conversion)

def obj(model):
    return sum(model.z[t,r] for t in model.T for r in model.R)
model.objctve = Objective(rule = obj, sense = minimize)

def demand_satisfaction(model,t,i):
    return model.x[t,i] >= model.df[t,i]
model.cons = Constraint(model.T, model.P ,rule = demand_satisfaction)

#C3JW07100K02DC , Cutting Pattern 1

solution = solve_model(model)
model = solution[0]
result = solution[1]
print (result)
model.solutions.store_to(result)

print ("************************************************************************")
print ("SKU Production Values \n")

for i in model.P:
    for t in model.T:
        val = round(value(model.x[t,i]),0)
        d = value(model.df[t,i])
        if val > 0:
            print ("Day:", t, "   SKU:",i,"   Group:",model.PGn[i].value,"   quantity:",val,"   demand:",d,"   Inventory Remaining:",val-d)

print ("************************************************************************")
print ("Bird Processing \n")

# for t in model.T:
#     for r,k,j in model.indx_r_k_j:
#         if model.x[r,k]


exit(0)
