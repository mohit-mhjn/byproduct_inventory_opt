# Setting Up Environment
import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)
import datetime
from pyomo.environ import *

# Importing Input Data
from sales_order_reader import get_orders
from inventory_reader import get_birds, get_parts
from index_reader import read_masters
from BOM_reader import read_combinations
from coef_param import read_coef

# Importing Solver Fucntion
from solutionmethod import solve_model

# Parsing Input Data
orders = get_orders()
birds_inv = get_birds()
parts_inv = get_parts()
indexes = read_masters()
bom_data = read_combinations()
cc_data = read_coef()
cost_data = cc_data['cost']
capacity_data = cc_data['capacity']

dt_all = orders['date'].unique()
dt_all.sort()

# MILP Model Initialization
model = ConcreteModel()

# Index Definition
model.T = Set(initialize= list(map(lambda x: str(x),t_all)), ordered = True) # planning horizon* Temporary >> To include initialize in indx
model.J = Set(initialize= indexes['cutting_pattern'].keys())   # cutting patterns
model.P = Set(initialize= indexes['product_group'].keys())     # products
model.K = Set(initialize= indexes['section'].keys())          # no of sections
model.R = Set(initialize= indexes['bird_type'].keys())         # type of carcasses
model.P_type = Set(initialize = ['Fresh','Frozen'])          # Type of Products
model.M = Set(initialize = [1,0])                           # Marination Indicator
model.C = Set(initialize = [1,2])                           # Customer Priority Indicator

# Generating combinations
def combination_gen1(model,j):
    global indexes
    return indexes['cutting_pattern'][j]['section']
model.Jk = Set(model.J, initialize = combination_gen1)

def combination_gen2(model,k):
    global indexes
    return indexes['section'][k]['cutting_pattern']
model.Kj = Set(model.K, initialize = combination_gen2)

def combination_gen3(model):
    global bom_data
    return bom_data['iter_combs']['sec_cp']
model.indx_kj = Set(dimen = 2, initialize = combination_gen3)

def combination_gen4(model,k,j):
    global bom_data
    return bom_data['sec_nwb_pg'][(k,j)]
model.Jp1 = Set(model.indx_kj, initialize = combination_gen4)

def combination_gen5(model,k,j):
    global bom_data
    return bom_data['sec_wb_pg'][(k,j)]
model.Jp2 = Set(model.indx_kj, initialize = combination_gen5)

def combination_gen6(model,k):
    global indexes
    return indexes['section'][k]['product_group']
model.Kp = Set(model.K, initialize = combination_gen6)

def combination_gen7(model):
    global bom_data
    return bom_data['iter_combs']['pgcptyp']
model.indx_pjr = Set(dimen = 3, initialize = combination_gen7)

def combination_gen8(model):
    global bom_data
    return bom_data['iter_combs']['typ_cp']
model.indx_rj = Set(dimen = 2, initialize = combination_gen8)

def combination_gen9(model):
    global bom_data
    return bom_data['iter_combs']['typseccp']
model.index_rkj = Set(dimen = 3, initialize = combination_gen9)

def combination_gen10(model,p):
    global bom_data
    return bom_data['iter_combs']['pg_cptyp'][p]
model.Pjr = Set(model.P,dimen=2,initialize = combination_gen10)

def combination_gen11(model,r,j):
    global bom_data
    return bom_data['iter_combs']['cptyp_pg'][(j,r)]
model.RJp = Set(model.indx_rj,initialize = combination_gen11)

# Loading Input Parameters
def inv_gen1(model,t,r):
    global birds_inv
    return birds_inv[(t,r)]
model.H = Param(model.T, model.R, initialize = inv_gen1)

def yield_gen(model,p,j,r):
    global bom_data
    return bom_data['yield_data'][(p,j,r)]['yld']
model.yd = Param(model.indx_pjr, initialize = yield_gen)

def capacity_gen1(model,process):
    global capacity_data
    return capacity_data[process]
model.process_capacity = Param(['freezing','marination'],initialize= capacity_gen1)

def capacity_gen2(model,j):
    global capacity_data
    return capacity_data['cutting_pattern'][j]
model.cutting_capacity = Param(model.J, initialize = capacity_gen2)

# def cost_gen1(model,process):
#     global cost
#     return cost_data[process]
model.unit_freezing_cost = Param(initialize= cost_data['freezing_cost'])

# def cost_gen2(model,process):
#     global cost
#     return cost_data[process]
model.unit_marination_cost = Param(initialize= cost_data['marination_cost'])

def cost_gen3(model,j):
    global cost_data
    return cost_data['cutting_cost'][j]
model.unit_cutting_cost = Param(model.J,initialize = cost_gen3)

def cost_gen4(model,p,r,t):
    global cost_data
    return cost_data['holding_cost'][(p,r,t)]
model.inventory_holding_cost = Param(model.P,model.R,model.P_type, initialize = cost_gen4)

def cost_gen5(model,p,r,t,m):
    global cost_data
    return cost_data['selling_price'][(p,r,t,m)]
model.selling_price = Param(model.P,model.R,model.P_type,model.M, initialize = cost_gen5)


"""
To check : Effect of Freezing on weight
To check : Effect of Marination on weight

bom_data = {'yield_data':yd,
            'sec_nwb_pg':sc_pg1,
            'sec_wb_pg':sc_pg2,
            'iter_combs':{
            'sec_cp':sec_cp_comb,
            'typ_cp':typ_cp_comb,
            'pgcptyp':pgcptyp_comb,
            'typseccp':typseccp_comb,
            'cptyp_pg':cptyp_pg_comb,
            'pg_cptyp':pg_cptyp_comb}}

indexes = {'bird_type':typ_dct,
           'cutting_pattern':cp_dct,
           'section':sec_dct,
           'product_group':pg_dct}

Flow Chart for Demand Matrix Generation and Inventory Update
"""
