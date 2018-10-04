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

# Importing Solver Fucntion
from solutionmethod import solve_model

# Parsing Input Data
# Static Data : Cached
with open("../cache/master_data","rb") as fp:
    master = pickle.load(fp)

# Variable Data >> READING FROM RELATED FILES
var_data = decision_input(datetime.date(2018,6,28),3)
var_data = get_orders(master,var_data,config)  # Function to get sales orders
var_data = get_birds(master,var_data,config) # Function to get birds availability at date,bird_type level
var_data = get_parts(master,var_data,config) # Function to get initial parts inventory at part,bird_type and age level (note :age only for fresh)

## MILP Model Initialization #############################################
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
