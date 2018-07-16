"""
Note: Compatible with Python 3

This file handles the main pyomo MILP that comprises of required Sets, Constraints, Expression, Param and Model objects.
Concrete model is used for instance creation

Loading Data in concrete model:
To load the static data cached data is imported from the respective reader modules
To load the dynamic data input files are read and transformed using the function of the respective readers (sales,inventory)

General Ref:
combination_gen : function used to load pyomo indexed sets in the model. The combinations are preprocessed and cached
expression_gen : fucntion used to generate pyomo expressions that are mentioned against
A [i] Constraint : Name used to indicate constraint with index i
In Comments, Product is reffered as product group

Assumptions:
1. If a cutting pattern is applied on whole bird it applies on all the sections of the birds in equal number of times
2. Inventory transfer if FIFO
3. Frozen product doesn't age (long shelf life in comparison with planning horizon)

To Do:
1. cache r,j,k,p combinations
2. planning horizon include in indexes
3. Data Post processing : Map indexes to their description
"""

# Setting Up Environment
import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)
import datetime
from pyomo.environ import *
import itertools
import pandas
# Importing Input Data
from sales_order_reader import get_orders
from inventory_reader import get_birds, get_parts
from index_reader import read_masters
from BOM_reader import read_combinations
from coef_param import read_coef
from ageing_param import read_inv_life

# Importing Solver Fucntion
from solutionmethod import solve_model

## Parsing Input Data  ###################################################
## Static Data : Cached
indexes = read_masters()
bom_data = read_combinations()
cc_data = read_coef()
cost_data = cc_data['cost']
capacity_data = cc_data['capacity']
inv_life_data = read_inv_life()
shelf_life_fresh = inv_life_data['shelf_life_fresh']
age_comb_fresh = inv_life_data['age_combinations_fresh']

## Variable Data
horizon = [datetime.date(2018,6,28),datetime.date(2018,6,29),datetime.date(2018,6,30)] # To Do: Get dates through Index
orders = get_orders(indexes,horizon)  # Function to get sales orders
birds_inv = get_birds(indexes,horizon) # Function to get birds availability at date,bird_type level
parts_inv = get_parts(indexes,horizon) # Function to get initial parts inventory at part,bird_type and age level (note :age only for fresh)
fresh_inv = parts_inv['Fresh']        # Separating fresh inventory
frozen_inv = parts_inv['Frozen']            # Separating frozen inventory

## MILP Model Initialization #############################################
model = ConcreteModel()

## Index Definition #####################################################
model.T = Set(initialize= list(range(len(horizon))), ordered = True) # planning horizon* Temporary >> To include initialize in indx
model.J = Set(initialize= indexes['cutting_pattern'].keys())   # cutting patterns
model.P = Set(initialize= indexes['product_group'].keys())     # products
model.K = Set(initialize= indexes['section'].keys())          # no of sections
model.R = Set(initialize= indexes['bird_type'].keys())         # type of carcasses
model.P_type = Set(initialize = indexes['product_typ'])          # Type of Products
model.M = Set(initialize = indexes['marination'])                # Marination Indicator
model.C = Set(initialize = indexes['c_priority'])                # Customer Priority Indicator

## Generating combinations ###############################################
def combination_gen1(model,j):
    global indexes
    return indexes['cutting_pattern'][j]['section']
model.Jk = Set(model.J, initialize = combination_gen1)       # Cutting Pattern >> set(Sections)

def combination_gen2(model,k):
    global indexes
    return indexes['section'][k]['cutting_pattern']
model.Kj = Set(model.K, initialize = combination_gen2)       # Section vs set(Cutting Patterns)  (Inverse of previous)

def combination_gen3(model):
    global bom_data
    return bom_data['iter_combs']['sec_cp']
model.indx_kj = Set(dimen = 2, initialize = combination_gen3)    # Combinations of (section,cutting_pattern)

def combination_gen4(model,k,j):
    global bom_data
    return bom_data['sec_nwb_pg'][(k,j)]
model.KJp1 = Set(model.indx_kj, initialize = combination_gen4)   # Non Whole bird products coming from a (section,cutting_pattern)

def combination_gen5(model,k,j):
    global bom_data
    return bom_data['sec_wb_pg'][(k,j)]
model.KJp2 = Set(model.indx_kj, initialize = combination_gen5)   # Whole bird products coming from a (section,cutting_pattern)

def combination_gen6(model,k):
    global indexes
    return indexes['section'][k]['product_group']
model.Kp = Set(model.K, initialize = combination_gen6)     # Products from coming section

def combination_gen7(model):
    global bom_data
    return bom_data['iter_combs']['pgcptyp']
model.indx_pjr = Set(dimen = 3, initialize = combination_gen7) # Combinations of (product,cutting_pattern,bird_type)

def combination_gen8(model):
    global bom_data
    return bom_data['iter_combs']['typ_cp']
model.indx_rj = Set(dimen = 2, initialize = combination_gen8)    # Combinations of (bird_type, cutting_pattern)

def combination_gen9(model):
    global bom_data
    return bom_data['iter_combs']['typseccp']
model.indx_rkj = Set(dimen = 3, initialize = combination_gen9)   # Combinations of (bird_type,section,cutting_pattern)

def combination_gen10(model,p):
    global bom_data
    return bom_data['iter_combs']['pg_cptyp'][p]
model.Pjr = Set(model.P,dimen=2,initialize = combination_gen10)    # Combinations of (cutting_pattern, bird_type) that yield product P

def combination_gen11(model,r,j):
    global bom_data
    return bom_data['iter_combs']['cptyp_pg'][(j,r)]
model.RJp = Set(model.indx_rj,initialize = combination_gen11)  # Products yielded by a particular (bird_type,cutting_pattern) (Inverse of previous)

def combination_gen12(model):
    global age_comb_fresh
    return age_comb_fresh
model.INV_Fresh = Set(dimen = 3, initialize=combination_gen12)   # For fresh products inventory index with age (product,bird_type,age) where age lies in range [1,shelf_life]

def combination_gen13(model):   # This can be cached >>> To Do
    my_set = set()
    for r,k,j in model.indx_rkj:
        in_PG = set(model.KJp1[k,j])
        if in_PG == set():
            my_set.add((r,k,j,-1))
        else:
            for grp in in_PG:
                my_set.add((r,k,j,grp))
    return my_set
model.indx_rkjp = Set(dimen = 4, initialize = combination_gen13)     # Combination of (bird type,section,cutting_pattern,product)

## Loading Input Parameters #############################################

def inv_gen1(model,t,r):
    global birds_inv
    global horizon
    dt = str(horizon[t])
    return birds_inv[(dt,r)]
model.H = Param(model.T, model.R, initialize = inv_gen1)         # availability of birds of type r at time t

def inv_gen2(model,p,r,l):  # Opening Inventory
    global fresh_inv
    if (p,r,l) in fresh_inv.keys():
        return fresh_inv[(p,r,l)]
    else:
        return 0
model.initial_inv_fresh = Param(model.INV_Fresh, initialize = inv_gen2)     # Initial inventory of fresh products p of bird type r  with age l (in days)

def inv_gen3(model,p,r):
    global frozen_inv
    if (p,r) in frozen_inv.keys():
        return frozen_inv[(p,r)]
    else:
        return 0
model.initial_inv_frozen = Param(model.P,model.R, initialize = inv_gen3)     # Initial inventory of frozen product p of bird type r

def order_gen1(model,t,cp,p,r,pt,m):
    global orders
    global horizon
    dt = str(horizon[t])
    if (dt,cp,p,r,pt,m) in orders.keys():
        return orders[(dt,cp,p,r,pt,m)]
    else:
        return 0
model.sales_order = Param(model.T,model.C,model.P,model.R,model.P_type,model.M, initialize = order_gen1)   # Sales order (day,customer_priority, product,bird_type, fresh/frozen, marination (1/0))

def order_gen2(model,t):
    return 10
model.further_proessing = Param(model.T, initialize = order_gen2)   # Demand of fresh products for further processing

def yield_gen(model,p,j,r):
    global bom_data
    return bom_data['yield_data'][(p,j,r)]['yld']
model.yd = Param(model.indx_pjr, initialize = yield_gen)    # Yield of product p from bird type r at cutting pattern j

def shelflife_gen(model,p,r): # Only Fresh
    global shelf_life_fresh
    return shelf_life_fresh[(p,r)]
model.L = Param(model.P,model.R,initialize = shelflife_gen)       # Shelf Life of fresh product p of bird type r

def capacity_gen1(model,process):
    global capacity_data
    return capacity_data[process]
model.process_capacity = Param(['freezing','marination'],initialize= capacity_gen1)   # Hard Coded process steps | Marination and freezing Capacity

def capacity_gen2(model,j):
    global capacity_data
    return capacity_data['cutting_pattern'][j]
model.cutting_capacity = Param(model.J, initialize = capacity_gen2)       # Cutting Capacity at cutting pattern J (birds/sections per hour)

# def cost_gen1(model,process):
#     global cost
#     return cost_data[process]
model.unit_freezing_cost = Param(initialize= cost_data['freezing_cost'])   # Operations Cost Coef. Freezing Cost

# def cost_gen2(model,process):
#     global cost
#     return cost_data[process]
model.unit_marination_cost = Param(initialize= cost_data['marination_cost'])   # Operations Cost Coef. Marination Cost

def cost_gen3(model,j):
    global cost_data
    return cost_data['cutting_cost'][j]
model.unit_cutting_cost = Param(model.J,initialize = cost_gen3)    # Operations Cost Cost of Cutting >> Related with Cutting Pattern

def cost_gen4(model,p,r,t):
    global cost_data
    return cost_data['holding_cost'][(p,r,t)]
model.inventory_holding_cost = Param(model.P,model.R,model.P_type, initialize = cost_gen4) # Cost coef of Inventroy Holding >> Fresh and Frozen product 'P' of bird type 'R'

def cost_gen5(model,p,r,t,m):
    global cost_data
    return cost_data['selling_price'][(p,r,t,m)]
model.selling_price = Param(model.P,model.R,model.P_type,model.M, initialize = cost_gen5)  # Selling Price (profit Cost Coef) selling price of product p bird type R against fresh/frozen and marination 1/0

## Variable Objects #######################################

model.z = Var(model.T, model.R, domain= NonNegativeIntegers)               # no of carcass r processed in period T
model.zk = Var(model.T, model.R, model.K, domain = NonNegativeIntegers)   # Number of k section produced by carcass type R
model.zkj = Var(model.T, model.indx_rkj, domain = NonNegativeIntegers)  # Number of times cutting pattern j is applied on section k of carcass type R
model.zj = Var(model.T, model.indx_rj, domain = NonNegativeIntegers)     # Number of times cutting pattern j is applied on bird of size R

model.xpr = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Quantity of product P of bird type R produced in time slot 't'
model.ifs = Var(model.T, model.INV_Fresh, domain = NonNegativeReals)     # Auxillary variable to check Inv Fresh
model.ifz = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Auxillary variable to check Inv Frozen (Aeging Diff not considered)
model.xpjr = Var(model.T, model.indx_pjr, domain = NonNegativeReals)     # Quantity of product P of bird type R produced in time slot 't' with cutting pattern J

model.x_freezing = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Quantity of Product P of bird type R converted from Fresh to Frozen
model.x_marination = Var(model.T,model.P,model.R, domain = NonNegativeReals)    # Quantity of Product P of bird type R converted from Fresh to Fresh Marinated

model.u_fresh = Var(model.T, model.P, model.R, domain = NonNegativeReals)       # Demand Satisfied Fresh
model.v_fresh = Var(model.T, model.P, model.R, domain = NonNegativeReals)       # Demand Unsatisfied Fresh
model.um_fresh = Var(model.T,model.P, model.R, domain = NonNegativeReals)       # Demand Satisfied marinated fresh
model.vm_fresh = Var(model.T, model.P, model.R, domain = NonNegativeReals)      # Demand Unsatisfied marinated fresh
model.u_frozen = Var(model.T, model.P, model.R, domain = NonNegativeReals)      # Demand Satisfied Frozen
model.v_frozen = Var(model.T, model.P, model.R, domain = NonNegativeReals)      # Demand Unsatisfied Frozen

## Constraints ##############################################

def carcass_availability(model,t,r):
    return model.z[t,r] <= model.H[t,r]
model.A0Constraint = Constraint(model.T, model.R, rule = carcass_availability)      # Total Number of Birds Cut < Bird Available

def carcass_to_section(model,t,r,k):
    return model.zk[t,r,k] == model.z[t,r]
model.A1Constraint = Constraint(model.T, model.R, model.K, rule = carcass_to_section)         # All sections of birds are cut in equal number (no inventroy at section level)

def carcass_in_cutpattern(model,t,r,k):
    lst = [j0 for r0,k0,j0 in model.indx_rkj if r0 == r and k0 == k and j0 in model.Kj[k]]
    return sum(model.zkj[t,r,k,j] for j in lst) == model.zk[t,r,k]
model.A2Constraint = Constraint(model.T, model.R, model.K, rule = carcass_in_cutpattern)   # Total number of cuts on section k of bird type r is the sum of total applicable cutting pattenrs on (r,k) combinations

def cutting_pattern_count_gen(model,t,r,k,j):
    return model.zj[t,r,j] >= model.zkj[t,r,k,j]
model.A3Constraint = Constraint(model.T, model.indx_rkj, rule = cutting_pattern_count_gen)  # Determining number of times cutting pattern J is applied (min value)

def cutting_pattern_count_limiter(model,t,r,j):  # Will become redundant if z is in min(obj)
    return model.zj[t,r,j] <= sum(model.zkj[t,r,k,j] for k in model.Jk[j])
model.A4Constraint = Constraint(model.T, model.indx_rj, rule = cutting_pattern_count_limiter) # Determining number of times cutting pattern J is applied (max value)

def cutting_pattern_balancer(model,t,r,k,j):
    return model.zkj[t,r,k,j] == model.zj[t,r,j]
model.A5Constraint = Constraint(model.T, model.indx_rkj, rule = cutting_pattern_balancer)   # Cutting pattern of whole bird is equally applied on all the sections

def product_yield_eqn(model,t,r,k,j,p):
    all_items = set(model.RJp[r,j])       # >> All items possible from r,j
    products1 = set([p])                  # >> All sectional(non-whole bird) bird entities from cutting pattenr j
    products2 = set(model.KJp2[k,j])      # >> All whole bird entities from cutting pattern j
    products = products1.union(products2) # >> Combine two sets of possible products
    products = all_items.intersection(products)  # >> This step will any remove any p = -1 coming from indx_rjkp
    if not products:
        return Constraint.Skip
    else:
        return model.zkj[t,r,k,j] == sum(model.xpjr[t,p,j,r]/model.yd[p,j,r] for p in products)
model.A7Constraint = Constraint(model.T, model.indx_rkjp, rule = product_yield_eqn)        # Conversion of a cut-up into a product group

## Expressions ##########################################

def expression_gen1(model,t,p,r):
    return sum(model.xpjr[t,p,j1,r] for p1,j1,r1 in model.indx_pjr if p1 == p and r1 == r)
model.fresh_part_production = Expression(model.T, model.P, model.R, rule = expression_gen1)

def expression_gen2(model,t,p,r):
    return model.x_freezing[t,p,r] + model.x_marination[t,p,r] + model.u_fresh[t,p,r]
model.inv_usage = Expression(model.T, model.P, model.R, rule = expression_gen2)

def expression_gen3(model,t,p,r,l):
    if t == 0:
        return model.initial_inv_fresh[p,r,l]
    else:
        if l == 1:
            return model.fresh_part_production[t-1,p,r] - (model.inv_usage[t-1,p,r] + sum(model.inv_fresh[t-1,p,r,l1] for l1 in range(1,int(model.L[p,r]))))
        else:
            return model.inv_fresh[t-1,p,r,l-1]
model.inv_fresh = Expression(model.T, model.INV_Fresh, rule = expression_gen3)

def expression_gen4(model,t,p,r):
    return sum(model.inv_fresh[t,p,r,l1] for l1 in range(1,int(model.L[p,r])))
model.total_inv_fresh = Expression(model.T,model.P,model.R, rule = expression_gen4)

def expression_gen5(model,t,p,r):
    if t == 0:
        return model.initial_inv_frozen[p,r]
    else:
        return model.inv_frozen[t-1,p,r] + model.x_freezing[t-1,p,r] - model.u_frozen[t-1,p,r]
model.inv_frozen = Expression(model.T, model.P, model.R, rule = expression_gen5)

##### Constraints on top of Expressions:

def limit_inv_usage(model,t,p,r):
    return model.total_inv_fresh[t,p,r] + model.fresh_part_production[t,p,r] >= model.inv_usage[t,p,r]
model.A8Constraint = Constraint(model.T,model.P,model.R, rule = limit_inv_usage)
# inv_fresh for all ages + model.fresh_part_production[t,p,r] =< model.inv_usage[t,p,r])

def fresh_requirement_balance(model,t,p,r):
    return model.u_fresh[t,p,r] + model.v_fresh[t,p,r] == sum(model.sales_order[t,c,p,r,'Fresh',0] for c in model.C)
model.requirement_balance1 = Constraint(model.T, model.P, model.R, rule = fresh_requirement_balance)

def fresh_m_requirement_balance1(model,t,p,r):
    return model.um_fresh[t,p,r] + model.vm_fresh[t,p,r] == sum(model.sales_order[t,c,p,r,'Fresh',1] for c in model.C)
model.requirement_balance2 = Constraint(model.T, model.P, model.R, rule = fresh_m_requirement_balance1)

def fresh_m_requirement_balance2(model,t,p,r):
    return model.um_fresh[t,p,r] == model.x_marination[t,p,r]
model.requirement_balance3 = Constraint(model.T, model.P, model.R, rule = fresh_m_requirement_balance2)

def frozen_requirement_balance(model,t,p,r):
    return model.u_frozen[t,p,r] + model.v_frozen[t,p,r] == sum(model.sales_order[t,c,p,r,'Frozen',m] for c in model.C for m in model.M)
model.requirement_balance4 = Constraint(model.T, model.P, model.R, rule = frozen_requirement_balance)

def inv_requirement_balance1(model,t,p,r,l):
    return model.ifs[t,p,r,l] == model.inv_fresh[t,p,r,l]
model.requirement_balance5 = Constraint(model.T, model.INV_Fresh, rule = inv_requirement_balance1)

def inv_requirement_balance2(model,t,p,r):
    return model.ifz[t,p,r] == model.inv_frozen[t,p,r]
model.requirement_balance6 = Constraint(model.T, model.P, model.R, rule = inv_requirement_balance2)

# Freeze Inv at the age = shelf life  >> need to check 'l' or 'l-1'
def freeze_expiring_inventory(model,t,p,r,l):
    if model.L[p,r] == l:
        return model.x_freezing[t,p,r] >= model.inv_fresh[t,p,r,l]
    else:
        return Constraint.Skip
model.A9Constraint = Constraint(model.T, model.INV_Fresh, rule = freeze_expiring_inventory)

def capacity_gen1(model,t,p,r):
    return model.x_freezing[t,p,r] <= model.process_capacity['freezing']*24
model.A10Constraint = Constraint(model.T, model.P, model.R, rule = capacity_gen1)

def capacity_gen2(model,t,p,r):
    return model.x_marination[t,p,r] <= model.process_capacity['marination']*24
model.A11Constraint = Constraint(model.T, model.P, model.R, rule = capacity_gen2)

def capacity_gen3(model,t,j):
    return sum(model.zkj[t,r,k,j] for r,k,j1 in model.indx_rkj if j == j1) <= 24*model.cutting_capacity[j]
model.A12Constraint = Constraint(model.T, model.J, rule  = capacity_gen3)

# Costing Expressions : selling_gains - Op Cost - Inv holding

def expression_gen6(model,t):
    return sum(model.u_fresh[t,p,r]*model.selling_price[p,r,'Fresh',0] + model.um_fresh[t,p,r]*model.selling_price[p,r,'Fresh',1] + model.u_frozen[t,p,r]*model.selling_price[p,r,'Frozen',0] for p in model.P for r in model.R)
model.selling_gains = Expression(model.T, rule = expression_gen6)   # Calculated selling gains obtained from satisfied demand of SKU's

def expression_gen7(model,t):
    return sum(sum(model.zkj[t,r,k,j] for r,k,j1 in model.indx_rkj if j == j1)*model.unit_cutting_cost[j] for j in model.J) + sum(model.x_freezing[t,p,r]*model.unit_freezing_cost + model.x_marination[t,p,r]*model.unit_marination_cost for p in model.P for r in model.R)
model.operations_cost = Expression(model.T, rule = expression_gen7)

def expression_gen8(model,t):
    return sum(model.total_inv_fresh[t,p,r]*model.inventory_holding_cost[p,r,'Fresh'] + model.ifz[t,p,r]*model.inventory_holding_cost[p,r,'Frozen'] for p in model.P for r in model.R)
model.holding_cost = Expression(model.T, rule = expression_gen8)

###################################  Temporary #########################
# def force1(model):
#     return model.zk['2018-06-28',1,1] >= 10
# model.F1Constraint = Constraint(rule = force1)

# def force11(model):
#     return model.x_freezing[0,18,1] == 10
# model.F11Constraint = Constraint(rule = force11)

# def force2(model):
#     return model.xpjr[0,18,1,1] >= 10
# model.F2Constraint = Constraint(rule = force2)

def obj(model):
    return sum(model.z[t,r] for t in model.T for r in model.R)
model.objctve = Objective(rule = obj, sense = minimize)

solution = solve_model(model)
model = solution[0]
result = solution[1]
# print(result)

############################################################## post processing to print result tables >>

# Bird Type Requirement
bird_req_data = []
for t,r in itertools.product(model.T,model.R):
    bird_req_data.append({'date':str(horizon[t]),'bird_size':indexes['bird_type'][r]['bird_type'],'req_number':model.z[t,r].value})
bird_type_requirement = pandas.DataFrame(bird_req_data)
bird_type_requirement = bird_type_requirement[(bird_type_requirement.req_number > 0)]


#Cutting Pattern Plan
cutting_pattern_data = []
for t,(r,k,j) in itertools.product(model.T,model.indx_rkj):
    cutting_pattern_data.append({'date':str(horizon[t]),'bird_type':indexes['bird_type'][r]['bird_type'],'section':indexes['section'][k]['description'],'cutting_pattern':j,'line':indexes['cutting_pattern'][j]['line'], 'pattern_count':model.zkj[t,r,k,j].value })
cutting_pattern_plan = pandas.DataFrame(cutting_pattern_data)
cutting_pattern_plan = cutting_pattern_plan[(cutting_pattern_plan.pattern_count > 0)]


#Output Production and Processing
production_data1 = []
production_data2 = []
production_data3 = []

for t,p,r in itertools.product(model.T, model.P, model.R):
    production_data1.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_produced':value(model.fresh_part_production[t,p,r]),'UOM':'KG'})
    production_data2.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_produced':model.x_freezing[t,p,r].value,'UOM':'KG'})
    production_data3.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_produced':model.x_marination[t,p,r].value,'UOM':'KG'})

fresh_production = pandas.DataFrame(production_data1)
fresh_production = fresh_production[(fresh_production.quantity_produced > 0)]

freezing_lots = pandas.DataFrame(production_data2)
freezing_lots = freezing_lots[(freezing_lots.quantity_produced > 0)]

marination_lots = pandas.DataFrame(production_data3)
marination_lots = marination_lots[(marination_lots.quantity_produced > 0)]


inventory_report1 = []
for t,(p,r,l) in itertools.product(model.T,model.INV_Fresh):
    inventory_report1.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'age_days':l,'quantity_on_hand':model.ifs[t,p,r,l].value,'UOM':'KG'})
fresh_inventory_report = pandas.DataFrame(inventory_report1)
fresh_inventory_report = fresh_inventory_report[(fresh_inventory_report.quantity_on_hand > 0)]

inventory_report2 = []
for t,p,r in itertools.product(model.T,model.P, model.R):
    inventory_report2.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_on_hand':model.ifz[t,p,r].value,'UOM':'KG'})
frozen_inventory_report = pandas.DataFrame(inventory_report2)
frozen_inventory_report = frozen_inventory_report[(frozen_inventory_report.quantity_on_hand > 0)]

sales_cost_report1 = []
sales_cost_report2 = []
sales_cost_report3 = []

for t,p,r in itertools.product(model.T,model.P,model.R):
    fresh_satisfied_q = model.u_fresh[t,p,r].value
    fresh_unsatisfied_q = model.v_fresh[t,p,r].value
    orders1 = sum(model.sales_order[t,c,p,r,'Fresh',0] for c in model.C)
    selling_gains1 = model.selling_price[p,r,'Fresh',0]*fresh_satisfied_q
    sales_cost_report1.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'orders':value(orders1),'satisfied':fresh_satisfied_q,'unsatisfied':fresh_unsatisfied_q,'selling_gains':value(selling_gains1)})

    marinated_satisfied_q = model.um_fresh[t,p,r].value
    marinated_unsatisfied_q = model.vm_fresh[t,p,r].value
    orders2 = sum(model.sales_order[t,c,p,r,'Fresh',1] for c in model.C)
    selling_gains2 = model.selling_price[p,r,'Fresh',1]*marinated_satisfied_q
    sales_cost_report2.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'orders':value(orders2),'satisfied':marinated_satisfied_q,'unsatisfied':marinated_unsatisfied_q,'selling_gains':value(selling_gains2)})

    frozen_satisfied_q = model.u_frozen[t,p,r].value
    frozen_unsatisfied_q = model.v_frozen[t,p,r].value
    orders3 = sum(model.sales_order[t,c,p,r,'Frozen',0] for c in model.C)
    selling_gains3 = model.selling_price[p,r,'Frozen',0]*frozen_satisfied_q
    sales_cost_report3.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'orders':value(orders3),'satisfied':frozen_satisfied_q,'unsatisfied':frozen_unsatisfied_q,'selling_gains':value(selling_gains3)})


sales_fresh_sku = pandas.DataFrame(sales_cost_report1)
sales_fresh_sku = sales_fresh_sku[(sales_fresh_sku.orders > 0)]

sales_marination_sku = pandas.DataFrame(sales_cost_report2)
sales_marination_sku = sales_marination_sku[(sales_marination_sku.orders > 0)]

sales_frozen_sku = pandas.DataFrame(sales_cost_report3)
sales_frozen_sku = sales_frozen_sku[(sales_frozen_sku.orders > 0)]


cost_report1 = []
for t in model.T:
    cost_report1.append({'date':str(horizon[t]),'COGS':value(model.operations_cost[t]),'HoldingCost':value(model.holding_cost[t]),'Revenue':value(model.selling_gains[t])})
cost_summary = pandas.DataFrame(cost_report1)


print (bird_type_requirement)
print (cutting_pattern_plan)
print (fresh_production)
print (freezing_lots)
print (marination_lots)
print (fresh_inventory_report)
print (frozen_inventory_report)
print (sales_fresh_sku)
print (sales_frozen_sku)
print (sales_marination_sku)
print (cost_summary)
