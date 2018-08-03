"""
To update and import cost and capacity parameters related to the process
Cost coef and capacity data is static unless operational changes
hence the data is cached into 'cost_coef' and 'capacity_coef' files

Direct execution will update the coef files
Indriectly read_coef is called to read the cached file

fucntion : update_coef is self explanatory
function : read_coef is also self explanatory

Ref:
capacity_dct is the object that stores the capacity data
cost_dct is the object that sotres the costing data

capacity_dct : { "marination": marination_capacity,
                "cutting_pattern": {"cutting_pattern_index":capacity},
                "freezing": freezing_capacity},

cost_dct : { "selling_price":{(product_group,bird_type,fresh/frozen,marination >>  0/1): selling price value,
            "cutting_cost": {'cutting_pattern_index': unit_cutting_cost_value},
            "holding_cost": {(product_group,bird_type,fresh/frozen): holding cost value},
            "freezing_cost": unit_freezing_cost
            "marination_cost": unit_marination_cost}

##############################################################################################


While updating the data, the timestamp of the update event is stored in update_status file

To Do:
1. Generalize Process Masters (presently hard coded)
        freezing cost and marination
        similarly to be done in a piece for cutting pattern as well

Idea:
Maintenance schedules on cutup/freezing/marination lines to be incorporated
"""
import pandas
import json
import pickle
import datetime

def update_coef():

    capacity_dct = {'cutting_pattern':None,'freezing':None, 'marination':None}
    cost_dct = {'selling_price':None, 'cutting_cost':None, 'freezing_cost':None, 'holding_cost':None, 'marination_cost':None}

    # Selling Price
    i_master = pandas.read_csv("input_files/sku_master.csv")
    i_master.drop(labels = ['sku_index','active'], axis = 1,inplace=True)
    cost_dct['selling_price'] = i_master.set_index(['prod_group_index','bird_type_index','product_type','marination']).to_dict(orient = 'dict')['selling_price']
    # print (sp_dct)  #Selling Price Dictionary

    # Operational Cost
    cp_master = pandas.read_csv("input_files/cutting_pattern.csv")
    cp_master = cp_master.filter(items = ['cutting_pattern','capacity_kgph','operational_cost'])
    cp_master.drop_duplicates(inplace = True)
    cost_dct['cutting_cost'] = cp_master.set_index('cutting_pattern').to_dict(orient='dict')['operational_cost']
    capacity_dct['cutting_pattern'] = cp_master.set_index('cutting_pattern').to_dict(orient = 'dict')['capacity_kgph']

    # Feezing Cost + Capacity
    # Marination Cost + Capacity
    # p_master = pandas.read_csv("input_files/processing.csv")
    cost_dct['freezing_cost'] = 3333
    cost_dct['marination_cost'] = 3
    capacity_dct['freezing'] = 414
    capacity_dct['marination'] = 200

    # Inventory Holding Cost
    i_master.drop(labels = ['marination'],axis=1,inplace = True)
    i_master.dropna(inplace=True)
    cost_dct['holding_cost'] = i_master.set_index(['prod_group_index','bird_type_index','product_type']).to_dict(orient = 'dict')['holding_cost']
    # print (hc_dct) #Holding Cost Dictionary

    #Cacheing the objects
    with open("input_files/cost_coef","wb") as fp:
        pickle.dump(cost_dct,fp)

    with open("input_files/capacity_coef","wb") as fp:
        pickle.dump(capacity_dct,fp)

    # Recording Event in the status file
    with open("input_files/update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['cost_coef'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")
        us['capacity_coef'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("input_files/update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print("SUCCESS : cost_coef updated!")
    print("SUCCESS : capacity_coef updated!")
    return None

def read_coef():
    # Loading the cached files
    with open("input_files/cost_coef","rb") as fp:
        cost_dct = pickle.load(fp)
    with open("input_files/capacity_coef","rb") as fp:
        capacity_dct = pickle.load(fp)
    return {'cost':cost_dct,'capacity':capacity_dct}

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    update_coef()
    # read_coef()
