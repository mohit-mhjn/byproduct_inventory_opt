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

To do:
1. use pickle for object serialization, json is converting int keys to string
2. Check consistancy within masters (if not catch and report error)
"""

import pandas
import json

def update_masters():

    typ = pandas.read_csv("input_files/bird_type.csv")
    typ_dct = typ.set_index("bird_type_index").to_dict(orient = 'index')
    # print (typ_dct)

    pg = pandas.read_csv("input_files/product_group.csv")
    pg["section"] = pg["section"].apply(lambda x: [int(i) for i in str(x)])
    pg_dct = pg.set_index("prod_group_index").to_dict(orient = 'index')
    # print (pg_dct)

    sec = pandas.read_csv("input_files/section.csv")
    sec_dct = sec.set_index("section_index").to_dict(orient = 'index')

    cp = pandas.read_csv("input_files/cutting_pattern.csv")
    cp['rate'] = cp['rate'].apply(lambda x: round(x,5))
    cp['capacity_kgph'] = cp['capacity_kgph'].apply(lambda x: round(x,5))
    cp_dct = cp.set_index("cutting_pattern").to_dict(orient = 'index')

    for cut in cp_dct.keys():
        df_tmp = cp[(cp.cutting_pattern == cut)]
        cp_dct[cut]['section'] = list(set(df_tmp['section']))

    for s in sec_dct.keys():
        df_tmp = cp[(cp.section == s)]
        sec_dct[s]['cutting_pattern'] = list(set(df_tmp['cutting_pattern']))

    for s in sec_dct.keys():
        df_tmp = pg[pg.section.map(set([s]).issubset)]
        sec_dct[s]['product_group'] = list(set(df_tmp['prod_group_index']))

    # print (sec_dct)
    # print (cp_dct)

    # Collect the required data in a python object
    indexes = {'bird_type':typ_dct,
               'cutting_pattern':cp_dct,
               'section':sec_dct,
               'product_group':pg_dct}

    # Dump object in a cache file
    with open("input_files/index_file.json","w") as fp:
        json.dump(indexes,fp)
    print ("SUCESS : index file updated!")
    return None

def read_masters():
    # Read cache file recreate the object
    with open('input_files/index_file.json',"r") as fp:
        k = dict(json.load(fp))
    return k

if __name__=='__main__':
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    # update_masters()
    # read_masters()
