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
1. Check if this operation can be merged in other
"""

import pandas
import json
import pickle
import datetime

def update_inv_life():

    # Inventory Shelf Life from SKU Master
    i_master = pandas.read_csv("input_files/sku_master.csv")
    i_master = i_master[(i_master.product_type == 'Fresh')]
    i_master.drop(labels = ['marination','product_type'],axis=1,inplace = True)
    i_master.dropna(inplace=True)
    shelf_life = i_master.set_index(['prod_group_index','bird_type_index']).to_dict(orient = 'dict')['shelf_life']
    my_set = set()
    for i,y in shelf_life.items():
        for k in range(0,int(y)+1):
            my_set.add((i[0],i[1],k))

    life_dct = {'shelf_life_fresh':shelf_life,'age_combinations_fresh':my_set}
    #Cacheing the objects
    with open("input_files/inv_life","wb") as fp:
        pickle.dump(life_dct,fp)

    # Recording Event in the status file
    with open("input_files/update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['ageing'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("input_files/update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print("SUCCESS : Inventory shelf life data updated!")

    return None

def read_inv_life():
    # Loading the cached files
    with open("input_files/inv_life","rb") as fp:
        life_dct = pickle.load(fp)
    return life_dct

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    # update_inv_life()
    # print (read_inv_life())
