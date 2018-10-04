"""
To Extract inventroy data from the source files

function: get_birds() will import the inventory of live birds by date by size to the processing unit

function: get_parts() will import the initial available inventory of parts by product group by size at T=0
            get_parts returned as two separate entities for fresh and frozen inventory
            Age is not critical in frozen inventory hence, presently considered only for fresh product
            Keys for fresh Inventory : product_group,bird_type,age
            Keys for frozen Inventory : product_group,bird_type,age

Direct Execution of this file will just read the data and print
Indirect Execution will be requir calling the corresponding functions
"""

import pandas
import datetime
import json
import warnings

def get_birds(master,var_data,config):
    horizon = var_data.horizon
    global us
    # print(int(config['input_source']['mySQL']))
    # Long Term Input from Farm Data/Harvest Data will be connected here >>
    # More Clarification required for bird Inventroy
    if bool(int(config['input_source']['mysql'])):
        import MySQLdb
        db = MySQLdb.connect(host=config['db']['host'], database=config['db']['db_name'], user=config['db']['user'],
                             password=config['db']['password'])
        db_cursor = db.cursor()

        # Index of Bird Types
        query_1 = "select * from birds_available"
        db_cursor.execute(query_1)
        tbl = pandas.DataFrame(list(db_cursor.fetchall()), columns=['date','bird_type_id','available'])
    else:
        tbl = pandas.read_csv("../input_files/birds_available.csv")

    if tbl.empty:
        raise ImportError("No inventory found; error code: 101A")

    with open('../update_status.json') as jsonfile:
        us = dict(json.load(jsonfile)) # us == update status
        dt = datetime.datetime.strptime(us['bird_avail'],"%Y-%m-%d %H:%M:%S").date()

    if horizon[0] > dt:  # Depends on Batch Job Schedule
        # Using bird inventory as updated at datetime = dt
        # raise AssertionError("T=0 Parts Bird availablility data has not been updated, using as is")
        warnings.warn("T=0 Parts Bird availablility data has not been updated, using as is")

    tbl['date'] = tbl['date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").date())
    tbl = tbl[tbl['date'].isin(horizon)]
    tbl['date'] = tbl['date'].apply(lambda x: str(x))

    if tbl.empty:
        raise ImportError("No inventory found; error code: 101B")

    tbl.reset_index(inplace=True,drop=True)
    tbl_dct = tbl.set_index(['date','bird_type_id']).to_dict(orient='dict')['available']

    var_data.bird_availability = tbl_dct
    # return tbl_dct
    return var_data

def get_parts(master,var_data,config):
    global us
    horizon = var_data.horizon
    horizon.sort()

    # Checking Inventory Update Status (as of which date the inventory data is available)
    with open('../update_status.json') as jsonfile:
        us = dict(json.load(jsonfile))
        dt = datetime.datetime.strptime(us['part_inventory'],"%Y-%m-%d %H:%M:%S").date()

    if horizon[0] > dt:  # Depends on Batch Job Schedule
        raise ImportError("T=0 Parts Inventory data has not been updated, Please reload inventory!, error code 101C")
        return None

    # Preprocess Inventroy data table from ERP in this function
    if bool(int(config['input_source']['mysql'])):
        import MySQLdb
        db = MySQLdb.connect(host=config['db']['host'], database=config['db']['db_name'], user=config['db']['user'],
                             password=config['db']['password'])
        db_cursor = db.cursor()

        # Index of Bird Types
        query_1 = "select * from inventory"
        db_cursor.execute(query_1)
        tbl = pandas.DataFrame(list(db_cursor.fetchall()), columns=['date','bird_type_id','available'])

        query_2 = "select * from sku_master"
        db_cursor.execute(query_2)
        i_master = pandas.DataFrame(list(db_cursor.fetchall()), columns=['sku_id','pgroup_id','bird_type_id','product_type','marination','active','selling_price','holding_cost','shelf_life'])
    else:
        tbl = pandas.read_csv("../input_files/inventory.csv")
        i_master = pandas.read_csv("../input_files/sku_master.csv") # Getting Inventory Shelf Life

    # if tbl.empty:
    #     raise ImportError("No inventory found; error code: 101D")


    i_master = i_master.filter(items =['pgroup_id','bird_type_id','product_type','shelf_life'])
    i_master.dropna(inplace=True)
    tbl = tbl.merge(i_master, on = ['pgroup_id','bird_type_id','product_type'])
    tbl = tbl[(tbl.inv_age <= tbl.shelf_life) & (tbl.inv_age > 0)]   # Age > 0 (Invalid Inventory Age for opening inv)

    frozen_dct = {}
    fresh_dct = {}

    if not tbl.empty:
        fresh_dct = tbl[(tbl.product_type == 1)].set_index(['pgroup_id','bird_type_id','inv_age']).to_dict(orient='dict')['q_on_hand']
        if fresh_dct == {}:
            warnings.warn("No inventory for Fresh products")

        frozen = tbl[(tbl.product_type == 2)]

        if not frozen.empty:
            frozen = frozen.drop(labels = ['inv_age','shelf_life','product_type'], axis=1)
            frozen = frozen.groupby(by = ['pgroup_id','bird_type_id']).sum()
            frozen_dct = frozen.to_dict(orient = 'dict')['q_on_hand']
        else:
            warnings.warn("No Inventory for Frozen products")
    else:
        warnings.warn("All inventory is finished/expired")

    var_data.part_inv_fresh = fresh_dct
    var_data.part_inv_frozen = frozen_dct
    # return {'Fresh':fresh_dct,'Frozen':frozen_dct}
    return var_data

if __name__=='__main__':
    import pickle
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)

    import configparser
    config = configparser.ConfigParser()
    config.read('../start_config.ini')
    from inputs import *
    with open("../cache/master_data","rb") as fp:
        master = pickle.load(fp)
    var_data = decision_input(datetime.date(2018,6,28),3)

    print ("Bird Inventory >>>")
    pandas.set_option('display.expand_frame_repr', False)
    print (pandas.DataFrame(get_birds(master,var_data,config).bird_availability, index=[0]))
    print ("\nPart Inventory >>>")
    var_data = get_parts(master,var_data,config)
    print (var_data.part_inv_fresh)
    print (var_data.part_inv_frozen)
