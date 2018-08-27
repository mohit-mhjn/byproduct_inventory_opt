"""
To update and read the shelf life of the products in the inventory
The input is static hence cached in a file : inv_life

Note:
If any changes inthe shelf life or addition of new SKU this program will be required to do the changes in the input files

Direct execution of this script will do the required changes in the inv_life
(function: update_inv_life() will be called)

Indirect execution will just read the dumped file and return that as a dictionary
(function read_inv_life() is used for that)

Used pickle for object serialization, bcoz of tuple type keys

While updating the data, the timestamp of the update event is stored in update_status file

Cached object Ref:
life_dct = {
    'shelf_life_fresh':shelf_life,
    'age_combinations_fresh':my_set
    }

shelf_life_fresh = {
    (product_group,bird_type) : shelf_life_value
    }    # Only Fresh

age_combinations_fresh = (set(product_group,bird_type,n) for n (int) in range lying in the interval [0, shelf_life])

##############################################################

To Do:
"""

import pandas
import json
import pickle
import datetime
import configparser
config = configparser.ConfigParser()
config.read('../start_config.ini')

def update_inv_life():

    if bool(int(config['input_source']['mySQL'])):
        import MySQLdb
        db = MySQLdb.connect(host=config['db']['host'], database=config['db']['db_name'], user=config['db']['user'],
                             password=config['db']['password'])
        db_cursor = db.cursor()

        # Index of Bird Types
        query_1 = "select * from inventory"
        db_cursor.execute(query_1)
        i_master = pandas.DataFrame(list(db_cursor.fetchall()), columns=['pgroup_id','bird_type_id','product_type'
            ,'inv_age','q_on_hand'])
    else:

        # Inventory Shelf Life from SKU Master
        i_master = pandas.read_csv("../input_files/sku_master.csv")

    i_master = i_master[(i_master.product_type == 1) & (i_master.bird_type_id < 99)]

    ## 99 because >> maximum 99 categories of bird types and fresh
    ## Fresh >> Inventory Ageing for only Fresh products is concerned

    i_master.drop(labels = ['marination','product_type'],axis=1,inplace = True)
    i_master.dropna(inplace=True)
    shelf_life = i_master.set_index(['pgroup_id','bird_type_id']).to_dict(orient = 'dict')['shelf_life']
    my_set = set()
    for i,y in shelf_life.items():
        for k in range(0,int(y)+1):
            my_set.add((i[0],i[1],k))

    life_dct = {'shelf_life_fresh':shelf_life,'age_combinations_fresh':my_set}
    #Cacheing the objects
    with open("../cache/inv_life","wb") as fp:
        pickle.dump(life_dct,fp)

    # Recording Event in the status file
    with open("../update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['ageing'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("../update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print("SUCCESS : Inventory shelf life data updated!")

    return None

def read_inv_life():
    # Loading the cached files
    with open("../cache/inv_life","rb") as fp:
        life_dct = pickle.load(fp)
    return life_dct

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    update_inv_life()
    # print (read_inv_life())
