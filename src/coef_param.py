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

    if bool(int(config['input_source']['mysql'])):
        import MySQLdb
        db = MySQLdb.connect(host=config['db']['host'], database=config['db']['db_name'], user=config['db']['user'],
                             password=config['db']['password'])
        db_cursor = db.cursor()

        # Index of Bird Types
        query_1 = "select * from inventory"
        db_cursor.execute(query_1)
        i_master = pandas.DataFrame(list(db_cursor.fetchall()), columns=['pgroup_id','bird_type_id','product_type'
            ,'inv_age','q_on_hand'])

        query_2 = "select * from cutting_pattern"
        db_cursor.execute(query_2)
        cp_master = pandas.DataFrame(list(db_cursor.fetchall()), columns=['cp_id', 'cp_line', 'section_id', 'description',
                                                                   'capacity', 'ops_cost'])

        query_3 = "select * from post_processing"
        db_cursor.execute(query_3)
        process_master = pandas.DataFrame(list(db_cursor.fetchall()), columns=['machine_id','machine_type','capacity','ops_cost'])

    else:

        # Selling Price
        i_master = pandas.read_csv("../input_files/sku_master.csv")
        cp_master = pandas.read_csv("../input_files/cutting_pattern.csv")
        process_master = pandas.read_csv("../input_files/post_processing.csv")

    i_master.drop(labels = ['sku_id','active'], axis=1,inplace=True)
    cost_dct['selling_price'] = i_master.set_index(['pgroup_id' ,'bird_type_id', 'product_type', 'marination']).to_dict(orient = 'dict')['selling_price']

    # Operational Cost
    cp_master = cp_master.filter(items=['cp_id','capacity','ops_cost'])
    cp_master.drop_duplicates(inplace=True)
    cost_dct['cutting_cost'] = cp_master.set_index('cp_id').to_dict(orient='dict')['ops_cost']
    capacity_dct['cutting_pattern'] = cp_master.set_index('cp_id').to_dict(orient='dict')['capacity']

    # Freezing Cost + Capacity
    # Marination Cost + Capacity
    process_master = process_master.groupby(["machine_type"]).agg({'machine_id':'size','capacity':'sum','ops_cost':'mean'})
    process_dct1 = process_master.to_dict(orient = 'dict')["capacity"]   ## >> dct_1 gives capacity
    process_dct2 = process_master.to_dict(orient = 'dict')["ops_cost"]   ## >> dct_2 gives cost

    cost_dct['freezing_cost'] = process_dct2["Freezer"]
    cost_dct['marination_cost'] = process_dct2["Marinator"]
    capacity_dct['freezing'] = process_dct1["Freezer"]
    capacity_dct['marination'] = process_dct1["Marinator"]

    # Inventory Holding Cost
    i_master.drop(labels = ['marination'],axis=1,inplace = True)
    i_master.dropna(inplace=True)
    cost_dct['holding_cost'] = i_master.set_index(['pgroup_id','bird_type_id','product_type']).to_dict(orient = 'dict')['holding_cost']
    # print (hc_dct) #Holding Cost Dictionary

    #Cacheing the objects
    with open("../cache/master_data","rb") as fp:
        master = pickle.load(fp)

    master.cost_dct = cost_dct
    master.capacity_dct = capacity_dct

    with open("../cache/master_data","wb") as fp:
        pickle.dump(master,fp)

    # Recording Event in the status file
    with open("../update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['cost_coef'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")
        us['capacity_coef'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("../update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print("SUCCESS : cost_coef updated!")
    print("SUCCESS : capacity_coef updated!")
    return None

# def read_coef():
#     # Loading the cached files
#     with open("../cache/capacity_coef","rb") as fp:
#         capacity_dct = pickle.load(fp)
#     return {'cost':cost_dct,'capacity':capacity_dct}

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    from inputs import *
    import configparser
    config = configparser.ConfigParser()
    config.read('../start_config.ini')
    update_coef()
