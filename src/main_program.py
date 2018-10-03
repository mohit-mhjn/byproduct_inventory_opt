print("\n\t<<<< \m/ may the force be with you \m/ >>>>\n")

# Setting Up Environment
print("Start")
import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)
import datetime
todayDate = datetime.date.today()
str_todayDate = datetime.datetime.strftime(todayDate, "%y%m%d")
import pickle

# Get Config
import configparser
config = configparser.ConfigParser()
config.read('../start_config.ini')

## Initializing Logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
file_handler = logging.FileHandler('../logs/main_program%s.log'%(todayDate))
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

## Checking Scenario to Execute
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--scenario_id", help="selection of scenario_id", type=int)
args = parser.parse_args()
scenario_id = args.scenario_id
if scenario_id == None:
    logger.warning("scenario selection not found \n\tUse argument \"--scenario_id n\" to define scenario number \n\tValid options for n = [1,2])")
    scenario_id = 2   # Scenario id set : [1,2] >> 1 is working 2 is infeasible
    logger.info("default scenario = %d"%(scenario_id))
logger.info("selected scenario id == %d"%(scenario_id))

# Importing Data processing modules
from inputs import *
from sales_order_reader import get_orders
from inventory_reader import get_birds, get_parts
from postprocessor import summarize_results

# from index_reader import read_masters
# from BOM_reader import read_combinations
# from coef_param import read_coef
# from ageing_param import read_inv_life
# from flex_typ import read_flex_typ

# Importing Solver Fucntion
from solutionmethod import solve_model

# Parsing Input Data
# Static Data : Cached
with open("../cache/master_data","rb") as fp:
    master = pickle.load(fp)

# print (master)
# indexes = read_masters()
# bom_data = read_combinations()
# cc_data = read_coef()
# cost_data = cc_data['cost']
# capacity_data = cc_data['capacity']
# inv_life_data = read_inv_life()
# shelf_life_fresh = inv_life_data['shelf_life_fresh']
# age_comb_fresh = inv_life_data['age_combinations_fresh']
# flex_ranges = read_flex_typ()

## Variable Data
horizon = [datetime.date(2018,6,28),datetime.date(2018,6,29),datetime.date(2018,6,30)] # To Do: Get dates through Index
orders = get_orders(indexes, horizon)  # Function to get sales orders
birds_inv = get_birds(indexes, horizon) # Function to get birds availability at date,bird_type level
parts_inv = get_parts(indexes, horizon) # Function to get initial parts inventory at part,bird_type and age level (note :age only for fresh)
fresh_inv = parts_inv['Fresh']        # Separating fresh inventory
frozen_inv = parts_inv['Frozen']            # Separating frozen inventory

orders_aggregate = orders["strict"]["aggregate"]      # Orders aggregated by C_Priority
order_breakup = orders["strict"]["breakup"]           # Individual Orders
order_grouped = orders["strict"]["grouped_by_product"] # Orders belonging to a SKU key at time t

flex_orders_aggregate = orders["flexible"]["aggregate"]        # Orders aggregated by C_Priority
flex_order_breakup = orders["flexible"]["breakup"]             # Individual Orders
flex_order_grouped = orders["flexible"]["grouped_by_product"]  # Orders belonging to a SKU key at time t

## MILP Model Initialization #############################################
from pyomo.environ import *
model = ConcreteModel()

## Index Definition #####################################################
model.T = Set(initialize= list(range(len(horizon))), ordered = True) # planning horizon* Temporary >> To include initialize in indx
model.J = Set(initialize= indexes['cutting_pattern'].keys())   # cutting patterns
model.P = Set(initialize= indexes['product_group'].keys())     # products
model.K = Set(initialize= indexes['section'].keys())          # no of sections
model.R = Set(initialize= indexes['bird_type'].keys())         # type of carcasses
model.P_type = Set(initialize = [i+1 for i in range(len(indexes['product_typ']))])       # Type of Products
model.M = Set(initialize = indexes['marination'])                # Marination Indicator
model.C_priority = Set(initialize = indexes['c_priority'])       # Customer Priority Indicator
model.O = Set(initialize = order_breakup.keys())                 # Order Id's for strict bird type products
model.RNG = Set(initialize = indexes['weight_range'].keys())      # Distinct range sets of flexible size range
model.flex_O = Set(initialize = flex_order_breakup.keys())      # Order Id's for flexible bird type products

## Generating combinations ###############################################
def combination_gen1(model,j):
    global indexes
    return indexes['cutting_pattern'][j]['section_id']
model.Jk = Set(model.J, initialize = combination_gen1)       # Cutting Pattern >> set(Sections)

def combination_gen2(model,k):
    global indexes
    return indexes['section'][k]['cp_id']
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

order_iter_set = set()
def combination_gen15(model,t,c,p,r,typ,m):       # Time Consuming >> Temporarily Fine
    global order_grouped
    dt = str(horizon[t])
    if (dt,c,p,r,typ,m) in order_grouped.keys():
        order_set = order_grouped[(dt,c,p,r,typ,m)]
        for o in order_set:
            order_iter_set.add((t,c,p,r,typ,m,o))
        return order_set
    else:
        return []
model.order_group = Set(model.T,model.C_priority,model.P, model.R, model.P_type, model.M, initialize = combination_gen15)

def combination_gen16(model):                                        #  Combination of date,c_priority, product, bird_type, fresh/frozen, marination, order_no
    global order_iter_set
    return list(order_iter_set)
model.indx_o_filling = Set(dimen = 7, initialize = combination_gen16)

def combination_gen17(model,rng):                                        # Bird Types for several flexible weight ranges
    global flex_ranges
    return flex_ranges["rng_to_type"][rng]
model.wt_set1 = Set(model.RNG, initialize = combination_gen17)

def combination_gen18(model,r):                                        #  Flexible weight ranges for weight R
    global flex_ranges
    return flex_ranges["type_to_rng"][r]
model.wt_set2 = Set(model.R, initialize = combination_gen18)

def combination_gen19(model):                                   #  Flexible weight range v/s weight combinations
    global flex_ranges
    return flex_ranges["flex_rng_comb"]
model.wt_set3 = Set(dimen = 2, initialize = combination_gen19)

#####  +++++++++++++ FLEX Orders Grouped ++++++

flex_order_iter_set = set()
def combination_gen20(model,t,c,p,r,typ,m):       # Time Consuming >> Temporarily Fine
    global flex_order_grouped
    dt = str(horizon[t])
    if (dt,c,p,r,typ,m) in flex_order_grouped.keys():
        print (dt)
        order_set = flex_order_grouped[(dt,c,p,r,typ,m)]
        for o in order_set:
            flex_order_iter_set.add((t,c,p,r,typ,m,o))
        return order_set
    else:
        return []
model.flex_order_group = Set(model.T,model.C_priority,model.P, model.RNG, model.P_type, model.M, initialize = combination_gen20)

## Loading Input Parameters #############################################
model.BigM = Param(initialize = 9999999)
# Highly Controversial Value >> Careful with this while tuning the real dataset
# Solvers might sometime fail because of big M values, If in case reduce the value by a factor of 10 and try. It shall work otherwise repeat this step again

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
    global orders_aggregate
    global horizon
    dt = str(horizon[t])
    if (dt,cp,p,r,pt,m) in orders_aggregate.keys():
        return orders_aggregate[(dt,cp,p,r,pt,m)]
    else:
        return 0
model.sales_order = Param(model.T,model.C_priority,model.P,model.R,model.P_type,model.M, initialize = order_gen1)   # Sales order (day,customer_priority, product,bird_type, fresh/frozen, marination (1/0))

def order_gen2(model,t):
    return 10
model.further_proessing = Param(model.T, initialize = order_gen2)   # Demand of fresh products for further processing

def yield_gen(model,p,j,r):
    global bom_data
    return bom_data['yield_data'][(p,j,r)]['yld']
model.yd = Param(model.indx_pjr, initialize = yield_gen)    # Yield of product p from bird type r at cutting pattern j

def shelflife_gen(model,p,r): # Only Fresh
    global shelf_life_fresh
    return shelf_life_fresh[(p, r)]
model.L = Param(model.P,model.R,initialize = shelflife_gen)       # Shelf Life of fresh product p of bird type r

def capacity_gen1(model,process):
    global capacity_data
    return capacity_data[process]
model.process_capacity = Param(['freezing','marination'],initialize= capacity_gen1)   # Hard Coded process steps | Marination and freezing Capacity

def capacity_gen2(model,j):
    global capacity_data
    return capacity_data['cp_id'][j]
model.cutting_capacity = Param(model.J, initialize = capacity_gen2)       # Cutting Capacity at cutting pattern J (birds/sections per hour)

# def cost_gen1(model,process):
#     global cost
#     return cost_data[process]
model.unit_freezing_cost = Param(initialize= cost_data['freezing_cost'])   # Operations Cost Coef. Freezing Cost

# def cost_gen2(model,process):
#     global cost
#     return cost_data[process]
model.unit_marination_cost = Param(initialize= cost_data['marination_cost'])   # Operations Cost Coef. Marination Cost

# def cost_gen3(model,j):
#     global cost_data
#     print(cost_data)
#     return cost_data['ops_cost'][j]
# model.unit_cutting_cost = Param(model.J,initialize = cost_gen3)    # Operations Cost Cost of Cutting >> Related with Cutting Pattern

def cost_gen4(model,p,r,t):
    global cost_data
    return cost_data['holding_cost'][(p,r,t)]
model.inventory_holding_cost = Param(model.P,model.R,model.P_type, initialize = cost_gen4) # Cost coef of Inventroy Holding >> Fresh and Frozen product 'P' of bird type 'R'

def cost_gen5(model,p,r,t,m):
    global cost_data
    if t == 1 and m == 1: # >> Removed from Consideration frozen marinated sku
        return 0
    else:
        return cost_data['selling_price'][(p,r,t,m)]

model.selling_price = Param(model.P,model.R,model.P_type,model.M, initialize = cost_gen5)  # Selling Price (profit Cost Coef) selling price of product p bird type R against fresh/frozen and marination 1/0

def sla_fulfillment(model,o):
    global order_breakup
    return order_breakup[o]["customer_sla"]*order_breakup[o]["order_qty"]
model.order_sla = Param(model.O, initialize = sla_fulfillment)                        # Service Level Agreement against each order line item

def order_selling_price_gen(model,o):
    global order_breakup
    return order_breakup[o]["selling_price"]
model.order_sp = Param(model.O, initialize = order_selling_price_gen)                # Selling price of order

def order_qty_gen(model,o):
    global order_breakup
    return order_breakup[o]["order_qty"]
model.order_qty = Param(model.O, initialize = order_qty_gen)                         # Quantity contained in a order

def order_date_gen(model,o):
    global order_breakup
    dt = horizon.index(datetime.datetime.strptime(order_breakup[o]["date"],"%Y-%m-%d").date())
    return dt
model.order_date = Param(model.O, initialize = order_date_gen)                        # date of an order

def order_priority_gen(model,o):
    global order_breakup
    return order_breakup[o]["priority"]
model.order_priority = Param(model.O, initialize = order_priority_gen)             # C_Priority of an oder

## Following is for flexible >> Under Testing

def order_gen3(model,t,c,p,rng,typ,m):                                                     # Sales order for Flex type SKU
    global flex_orders_aggregate
    global horizon
    dt = str(horizon[t])
    if (dt,c,p,rng,typ,m) in flex_orders_aggregate.keys():
        return flex_orders_aggregate[(dt,c,p,rng,typ,m)]
    else:
        return 0
model.flex_sales_order = Param(model.T,model.C_priority,model.P,model.RNG,model.P_type,[0],initialize = order_gen3)   # Only m = 0 >> No Marination for flex (not in scope)

def flex_sla_fulfillment(model,o):
    global flex_order_breakup
    return flex_order_breakup[o]["customer_sla"]*flex_order_breakup[o]["order_qty"]
model.flex_order_sla = Param(model.flex_O, initialize = flex_sla_fulfillment)                        # Service Level Agreement against each order line item

def flex_order_qty_gen(model,o):
    global flex_order_breakup
    return flex_order_breakup[o]["order_qty"]
model.flex_order_qty = Param(model.flex_O, initialize = flex_order_qty_gen)                         # Quantity contained in a order

## Variable Objects #######################################

model.z = Var(model.T, model.R, domain= NonNegativeIntegers)             # no of carcass r processed in period T
model.zk = Var(model.T, model.R, model.K, domain = NonNegativeIntegers)  # Number of k section produced by carcass type R
model.zkj = Var(model.T, model.indx_rkj, domain = NonNegativeIntegers)   # Number of times cutting pattern j is applied on section k of carcass type R
model.zj = Var(model.T, model.indx_rj, domain = NonNegativeIntegers)     # Number of times cutting pattern j is applied on bird of size R

model.xpr = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Quantity of product P of bird type R produced in time slot 't'
model.ifs = Var(model.T, model.INV_Fresh, domain = NonNegativeReals)     # Auxillary variable to check Inv Fresh
model.ifz = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Auxillary variable to check Inv Frozen (Aeging Diff not considered)
model.xpjr = Var(model.T, model.indx_pjr, domain = NonNegativeReals)     # Quantity of product P of bird type R produced in time slot 't' with cutting pattern J

model.il = Var(model.T, model.INV_Fresh, domain = NonNegativeReals)      # Fresh Inventory of Age L used
=======
# horizon = [datetime.date(2018,6,28),datetime.date(2018,6,29),datetime.date(2018,6,30)] # To Do: Get dates through Index
var_data = decision_input(datetime.date(2018,6,28),3)
>>>>>>> 32a3c28106adb722f3fc94b0f00fff970557e6c8

var_data = get_orders(master,var_data,config)  # Function to get sales orders
var_data = get_birds(master,var_data,config) # Function to get birds availability at date,bird_type level
var_data = get_parts(master,var_data,config) # Function to get initial parts inventory at part,bird_type and age level (note :age only for fresh)

#fresh_inv = parts_inv[1]        # Separating fresh inventory
#frozen_inv = parts_inv[2]            # Separating frozen inventory

# orders_aggregate = orders["strict"]["aggregate"]      # Oders agreegated by C_Priority
# order_breakup = orders["strict"]["breakup"]           # Individual Orders
# order_grouped = orders["strict"]["grouped_by_product"] # Orders belonging to a SKU key at time t

# flex_orders_aggregate = orders["flexible"]["aggregate"]        # Oders agreegated by C_Priority
# flex_order_breakup = orders["flexible"]["breakup"]             # Individual Orders
# flex_order_grouped = orders["flexible"]["grouped_by_product"]  # Orders belonging to a SKU key at time t

from equations import create_instance
model = create_instance(master,var_data, scenario = scenario_id)

## Using Solver Method ##################################################
solution = solve_model(model, p_summary = bool(int(config['solver']['p_summary'])))
model = solution[0]
result = solution[1]
if bool(int(config['solver']['print_solution'])):
    print (result)


## post processing to print result tables ########################################
summarize_results(model,var_data,master,
                    print_tables= bool(int(config['results']['print_tables'])),
                    keep_files = bool(int(config['results']['keep_files'])))

print ("End")
exit(0)

"""
## SOME CODE PIECES REMOVED ################

## Objective function Scenario 2 found not to be important for customers

elif scenario_id == 2:

    def produce_fresh_for_p1(model,t,p,r):
        return model.u_fresh[t,p,r] >= model.sales_order[t,1,p,r,1,0]
    model.SC2_Constraint1 = Constraint(model.T, model.P, model.R, rule = produce_fresh_for_p1)

    def produce_fresh_m_for_p1(model,t,p,r):
        return model.um_fresh[t,p,r] >= model.sales_order[t,1,p,r,1,1]
    model.SC2_Constraint2 = Constraint(model.T, model.P, model.R, rule = produce_fresh_m_for_p1)

    def produce_frozen_for_p1(model,t,p,r):
        return model.u_frozen[t,p,r] >= model.sales_order[t,1,p,r,2,0]
    model.SC2_Constraint3 = Constraint(model.T, model.P, model.R, rule = produce_frozen_for_p1)

    def production_constraint_for_p1(model,t,p,r):
        req_fresh = model.sales_order[t,1,p,r,1,0] + model.sales_order[t,1,p,r,1,1] - model.total_inv_fresh[t,p,r]
        req_frozen = model.sales_order[t,1,p,r,2,0] - model.inv_frozen[t,p,r]
        return model.xpr[t,p,r] <= req_fresh + req_frozen
    model.SC2_Constraint4 = Constraint(model.T,model.P, model.R, rule = production_constraint_for_p1)
    # return sum(model.profit_projected[t] for t in model.T)
    return sum(model.z[t,r] for t in model.T for r in model.R) + sum((3-t)*model.x_freezing[t,p,r] for t in model.T for p in model.P for r in model.R)

# def pref_use_older_inv(model,t,p,r):  ## not Required on Priority >>  Inv age at 3 must be finished before starting to use inv at age 2
#     return
# model.A131Constraint = Constraint(model.T,model.P,model.R)
"""
