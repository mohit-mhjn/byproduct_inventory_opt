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
# horizon = [datetime.date(2018,6,28),datetime.date(2018,6,29),datetime.date(2018,6,30)] # To Do: Get dates through Index
var_data = decision_input(datetime.date(2018,6,28),3)

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
