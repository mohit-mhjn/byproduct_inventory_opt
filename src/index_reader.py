"""
To create indexes required by the model to build the variable and parameter matrices
These inputs are static hence they are stored in a index_file as json

Note:
If any of the master is updated then the indexes need to be updated as well.

Direct execution of this script will do the required changes in the index_file.json
(function: update_masters() will be called)

Indirect execution will just read the dumped index_file.json and return that as a dictionary
(function read_masters() is used for that)

Ref:
typ_dct : Bird Types and their description
pg_dct : Product Groups and their description
cp_dct : Cutting Patterns, associated sections with the cut and the respective processing rate
sec_dct : Sections and their associated/favourable Cutting Patterns
indexes dictionary is the collection of all above

indexes = {'bird_type':typ_dct,
           'cutting_pattern':cp_dct,
           'section':sec_dct,
           'product_group':pg_dct,
           'marination': marination,
           'c_priority': c_priority,
           'product_typ': product_typ}

bird_type = { bird_typ_index : {'bird_type': 'S/M/ML/L/XL', 'dist_p': distribution_area , 'abs_weight': avg weight in abs term }}
cutting_pattern = { cp_index: {'capacity_kgph': , 'description': 'Not Available', 'section': [1, 2, 3], 'rate': , 'operational_cost': , 'line': }}
section = {section_index : {'product_group': [32, 18, 19, 22, 29, 30, 31], 'cutting_pattern': [1, 2, 3, 4, 5, 6, 7, 8], 'description': 'Section 1'}}
product_group = {pg_index: {'product_group': 'BL BFLY BREAST', 'section': [2], 'n_parts': 1}}
marination = [0,1]
c_priority = [1,2]
product_typ = ["Fresh","Frozen"]

####################################################################

Used pickle for object serialization, bcoz json is converting int keys to string

While updating the data, the timestamp of the update event is stored in update_status file

To do:
1. dateindex
2. Check consistancy within masters (if not catch and report error)
3. Try custom class instead of dict in the output object
"""

import pandas
import pickle
import json
import datetime

def update_masters():
    # Index of Bird Types
    typ = pandas.read_csv("input_files/bird_type.csv")
    typ_dct = typ.set_index("bird_type_index").to_dict(orient = 'index')
    # print (typ_dct)

    # Index of Product Groups
    pg = pandas.read_csv("input_files/product_group.csv")
    pg["section"] = pg["section"].apply(lambda x: [int(i) for i in str(x)])
    pg_dct = pg.set_index("prod_group_index").to_dict(orient = 'index')
    # print (pg_dct)

    # Index of sections
    sec = pandas.read_csv("input_files/section.csv")
    sec_dct = sec.set_index("section_index").to_dict(orient = 'index')

    # Index of cutting patterns
    cp = pandas.read_csv("input_files/cutting_pattern.csv")
    cp['rate'] = cp['rate'].apply(lambda x: round(x,5))
    cp['capacity_kgph'] = cp['capacity_kgph'].apply(lambda x: round(x,5))
    cp_dct = cp.set_index("cutting_pattern").to_dict(orient = 'index')

    # Mapping Cutting pattern vs Sections
    for cut in cp_dct.keys():
        df_tmp = cp[(cp.cutting_pattern == cut)]
        cp_dct[cut]['section'] = list(set(df_tmp['section']))

    # Mapping Section vs cutting_pattern >> Inverse of previous
    for s in sec_dct.keys():
        df_tmp = cp[(cp.section == s)]
        sec_dct[s]['cutting_pattern'] = list(set(df_tmp['cutting_pattern']))

    sec_cut_p_not_available = []  ## >> Need to remove these sections >> Log the warning event and record the list
    for idx in list(sec_dct.keys()):
        if sec_dct[idx]['cutting_pattern'] == []:
            sec_cut_p_not_available.append(idx)
            del sec_dct[idx]                           # Removing Section
    # print (sec_cut_p_not_available)

    for s in sec_dct.keys():
        df_tmp = pg[pg.section.map(set([s]).issubset)]
        sec_dct[s]['product_group'] = list(set(df_tmp['prod_group_index']))

    # print (sec_dct)
    # print (cp_dct)
    product_typ = ['Fresh','Frozen']
    marination = [1,0]
    c_priority = [1,2]

    # Collect the required data in a python object
    indexes = {'bird_type':typ_dct,
               'cutting_pattern':cp_dct,
               'section':sec_dct,
               'product_group':pg_dct,
               'marination': marination,
               'c_priority': c_priority,
               'product_typ': product_typ}

    # Dump object in a cache file
    with open("input_files/index_file","wb") as fp:
        pickle.dump(indexes,fp)

    # Recording Event in the status file
    with open("input_files/update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['index_file'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("input_files/update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print ("SUCESS : index file updated!")

    return None

def read_masters():
    # Read cache file recreate the object
    with open('input_files/index_file',"rb") as fp:
        k = pickle.load(fp)
    return k

if __name__=='__main__':
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    update_masters()
    # read_masters()
