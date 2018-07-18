print ("Start")
print ("\n\t<<<< May the force be with you \m/ >>>>\n")
"""
Note: Compatible with Python 3

This file handles the main pyomo MILP that comprises of required Sets, Constraints, Expression, Param and Model objects.
Concrete model is used for pyomo instance creation

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
1. process logger
2. planning horizon include in indexes
3. Data Post processing : Map indexes to their description >> Partially Done : Need to move the piece to separate program + imporve result formatting >>
4. Warehouse Capacity
5.
"""

# Setting Up Environment
import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)
import datetime
from pyomo.environ import *

# Importing Data processing modules
from sales_order_reader import get_orders
from inventory_reader import get_birds, get_parts
from index_reader import read_masters
from BOM_reader import read_combinations
from coef_param import read_coef
from ageing_param import read_inv_life
from postprocessor import summarize_results

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
    global bom_data
    return bom_data['iter_combs']['typseccpp']
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

model.il = Var(model.T, model.INV_Fresh, domain = NonNegativeReals)     # Fresh Inventory of Age L used

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

def fresh_part_production(model,t,p,r):
    return model.xpr[t,p,r] == sum(model.xpjr[t,p,j1,r] for p1,j1,r1 in model.indx_pjr if p1 == p and r1 == r)
model.A8Constraint = Constraint(model.T, model.P, model.R, rule = fresh_part_production)           # Summation of product belonging to a bird type obtained from agreegation of all freshly produced lots

## Inventroy Balance Equations and Constraints ##########################################

def expression_gen1(model,t,p,r):
    return model.x_freezing[t,p,r] + model.x_marination[t,p,r] + model.u_fresh[t,p,r]
model.fresh_inv_used = Expression(model.T, model.P, model.R, rule = expression_gen1)        # Quantity of Fresh Inventory used is equal to inv q used in freezing + q used in marination + q sold in fresh form

def inventory_used_for_age(model,t,p,r):                                                    # Quantity of Fresh Inv used = Qunaity of Fresh Inv used for all ages
    return sum(model.il[t,p,r,l] for l in range(int(model.L[p,r]) + 1)) == model.fresh_inv_used[t,p,r]
model.A9Constraint = Constraint(model.T, model.P, model.R, rule = inventory_used_for_age)

def expression_gen2(model,t,p,r,l):
    if t == 0 or l == 0:                                                              # opening inv == 0 at age 0 (initial value without production)
        return model.initial_inv_fresh[p,r,l]
    else:
        if l == 1:
            return model.xpr[t-1,p,r] - model.il[t-1,p,r,0]       # Inventroy produced at t is aged == 0
        else:
            return model.fresh_inv_qoh[t-1,p,r,l-1] - model.il[t-1,p,r,l-1]
model.fresh_inv_qoh = Expression(model.T, model.INV_Fresh, rule = expression_gen2)       # Expression to derive Fresh Inventory(opening) quantity_on_hand(qoh) by age

def expression_gen3(model,t,p,r):
    if t == 0:
        return model.initial_inv_frozen[p,r]
    else:
        return model.inv_frozen[t-1,p,r] + model.x_freezing[t-1,p,r] - model.u_frozen[t-1,p,r]
model.inv_frozen = Expression(model.T, model.P, model.R, rule = expression_gen3)            # Expression to derive Frozen Inventroy(opening) quantity on hand total without age

def inv_requirement_balance1(model,t,p,r,l):
    return model.ifs[t,p,r,l] == model.fresh_inv_qoh[t,p,r,l]
model.A10Constraint = Constraint(model.T, model.INV_Fresh, rule = inv_requirement_balance1)   # Fresh Inventory Variable which makes fresh_inv_qoh >= 0

def inv_requirement_balance2(model,t,p,r):
    return model.ifz[t,p,r] == model.inv_frozen[t,p,r]
model.A11Constraint = Constraint(model.T, model.P, model.R, rule = inv_requirement_balance2)   # Frozen Inventory Variable which makes inv_frozen >= 0

def expression_gen4(model,t,p,r):
    return sum(model.fresh_inv_qoh[t,p,r,l1] for l1 in range(int(model.L[p,r])+1))         # Total Inventory(opening) Fresh == sum of inventory at all ages
model.total_inv_fresh = Expression(model.T,model.P,model.R, rule = expression_gen4)

def limit_inv_usage(model,t,p,r):                                                           # Inventory Usage <= Inital Available + Usage
    return model.total_inv_fresh[t,p,r] + model.xpr[t,p,r] >= model.fresh_inv_used[t,p,r]
model.A12Constraint = Constraint(model.T, model.P, model.R, rule = limit_inv_usage)

def balance_inv_usage(model,t,p,r):                                                        # Conserve Inventory Quantity between two consecutive days
    if t == 0:                                                                              # inv_fresh for all ages + model.fresh_part_production[t,p,r] =< model.inv_usage[t,p,r])
        return Constraint.Skip
    else:
        return model.total_inv_fresh[t-1,p,r] + model.xpr[t-1,p,r] - model.fresh_inv_used[t-1,p,r] == model.total_inv_fresh[t,p,r]
model.A12_1Constraint = Constraint(model.T, model.P, model.R, rule = balance_inv_usage)

def freeze_expiring_inventory(model,t,p,r):                                         # Freeze Inv at the age = shelf life  >> need to check 'l' or 'l-1'
    max_life = model.L[p,r]
    return model.fresh_inv_qoh[t,p,r,max_life] - model.fresh_inv_used[t,p,r] <= 0
model.A13Constraint = Constraint(model.T, model.P, model.R, rule = freeze_expiring_inventory)

### Demand Satisfaction Constraints

def fresh_requirement_balance(model,t,p,r):
    return model.u_fresh[t,p,r] + model.v_fresh[t,p,r] == sum(model.sales_order[t,c,p,r,'Fresh',0] for c in model.C)
model.requirement_balance1 = Constraint(model.T, model.P, model.R, rule = fresh_requirement_balance)    # Sold + Unsold Fresh Without Marination

def fresh_m_requirement_balance1(model,t,p,r):
    return model.um_fresh[t,p,r] + model.vm_fresh[t,p,r] == sum(model.sales_order[t,c,p,r,'Fresh',1] for c in model.C)
model.requirement_balance2 = Constraint(model.T, model.P, model.R, rule = fresh_m_requirement_balance1)   # Sold + Unsold Fresh with Marination

def fresh_m_requirement_balance2(model,t,p,r):
    return model.um_fresh[t,p,r] == model.x_marination[t,p,r]
model.requirement_balance3 = Constraint(model.T, model.P, model.R, rule = fresh_m_requirement_balance2)    # Marination process is Make to Order

def frozen_requirement_balance(model,t,p,r):
    return model.u_frozen[t,p,r] + model.v_frozen[t,p,r] == sum(model.sales_order[t,c,p,r,'Frozen',m] for c in model.C for m in model.M)
model.requirement_balance4 = Constraint(model.T, model.P, model.R, rule = frozen_requirement_balance)    # Sold + Unsold Frozen Products without Marination

## Capacity Constraints ###################################  (Checking on UOM Pending)

def capacity_gen1(model,t,p,r):
    return model.x_freezing[t,p,r] <= model.process_capacity['freezing']*24
model.A14Constraint = Constraint(model.T, model.P, model.R, rule = capacity_gen1)

def capacity_gen2(model,t,p,r):
    return model.x_marination[t,p,r] <= model.process_capacity['marination']*24
model.A15Constraint = Constraint(model.T, model.P, model.R, rule = capacity_gen2)

def capacity_gen3(model,t,j):
    return sum(model.zkj[t,r,k,j] for r,k,j1 in model.indx_rkj if j == j1) <= 24*model.cutting_capacity[j]
model.A16Constraint = Constraint(model.T, model.J, rule  = capacity_gen3)

# Costing Expressions : selling_gains - Op Cost - Inv holding ##########################

def expression_gen6(model,t):
    return sum(model.u_fresh[t,p,r]*model.selling_price[p,r,'Fresh',0] + model.um_fresh[t,p,r]*model.selling_price[p,r,'Fresh',1] + model.u_frozen[t,p,r]*model.selling_price[p,r,'Frozen',0] for p in model.P for r in model.R)
model.selling_gains = Expression(model.T, rule = expression_gen6)   # Calculated selling gains obtained from satisfied demand of SKU's

def expression_gen7(model,t):
    return sum(sum(model.zkj[t,r,k,j] for r,k,j1 in model.indx_rkj if j == j1)*model.unit_cutting_cost[j] for j in model.J) + sum(model.x_freezing[t,p,r]*model.unit_freezing_cost + model.x_marination[t,p,r]*model.unit_marination_cost for p in model.P for r in model.R)
model.operations_cost = Expression(model.T, rule = expression_gen7)   # Calculating total cost incurred in processing in cutup + freezing + marination process

def expression_gen8(model,t):
    return sum(model.total_inv_fresh[t,p,r]*model.inventory_holding_cost[p,r,'Fresh'] + model.ifz[t,p,r]*model.inventory_holding_cost[p,r,'Frozen'] for p in model.P for r in model.R)
model.holding_cost = Expression(model.T, rule = expression_gen8)        # Calculation total Cost Incurred to hold the imbalance inventory

## Temporary Forcing Constraints (Testing Purpose) #########################

# def force1(model):
#     return model.zk['2018-06-28',1,1] >= 10
# model.F1Constraint = Constraint(rule = force1)

def force11(model):
    return model.x_freezing[0,7,1] == 10
model.F11Constraint = Constraint(rule = force11)

# def force12(model):
#     return sum(model.x_marination[t,18,1] for t in model.T) == 10
# model.F111Constraint = Constraint(rule = force12)

# def stop_freezing(model):
#     return sum(model.x_freezing[t,p,r] for t in model.T for p in model.P for r in model.R) == 0
# model.F12Constraint = Constraint(rule = stop_freezing)

## Objective Function and Scenario Selection ############################################

def obj(model):
    return sum(model.z[t,r] for t in model.T for r in model.R) + sum((3-t)*model.x_freezing[t,p,r] for t in model.T for p in model.P for r in model.R)
model.objctve = Objective(rule = obj, sense = minimize)

## Using Solver Method ##################################################

solution = solve_model(model)
model = solution[0]
result = solution[1]
#print(result)

## post processing to print result tables #######################################################
summarize_results(model,horizon,indexes,print_tables=False,keep_files = False)

print ("End")
exit()
