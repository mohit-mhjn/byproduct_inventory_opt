"""
To Extract inventroy data from the source files

function: get_birds() will import the inventory of live birds by date by size to the processing unit
function: get_parts() will import the initial available inventory of parts by product group by size at T=0
            get_parts returned as dataframe for preprocessing with sales order >> main_program will convert to dictionary

Direct Execution of this file will just read the data and print
Indirect Execution will be requir calling the corresponding functions
"""

import pandas
import datetime
import json

def get_birds():
    # Long Term Input from Farm Data/Harvest Data will be connected here >>
    # More Clarification required for bird Inventroy
    tbl = pandas.read_csv("input_files/birds_available.csv")
    tbl.reset_index(inplace=True,drop=True)
    tbl_dct = tbl.set_index(['date','bird_type']).to_dict(orient='dict')['inventory']
    return tbl_dct

def get_parts():
    # Preprocess Inventroy data table from ERP in this function
    tbl = pandas.read_csv("input_files/inventory.csv")
    i_master = pandas.read_csv("input_files/sku_master.csv") # Getting Inventory Shelf Life
    i_master = i_master.filter(items =['prod_group_index','bird_type_index','product_type','shelf_life'])
    i_master.dropna(inplace=True)
    tbl = tbl.merge(i_master, on = ['prod_group_index','bird_type_index','product_type']) # Join Schelf Life with Inv File
    tbl = tbl[(tbl.inv_age < tbl.shelf_life)]    # Usable Inventory

    # Checking Inventory Update Status (as of which date the inventory data is available)
    with open('input_files/update_status.json') as jsonfile:
        us = dict(json.load(jsonfile))
        dt = us['inventory']

    tbl['date'] = datetime.datetime.strptime(dt,"%Y-%m-%d %H:%M:%S").date()
    tbl.reset_index(inplace=True,drop =True)
    return tbl

if __name__=='__main__':
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    print ("Bird Inventory >>>")
    print (get_birds())
    print ("\nPart Inventory >>>")
    print (get_parts())
