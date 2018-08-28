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

    if bool(int(config['input_source']['mySQL'])):
        import MySQLdb
        db = MySQLdb.connect(host=config['db']['host'], database=config['db']['db_name'], user=config['db']['user'],
                             password=config['db']['password'])
        db_cursor = db.cursor()

        # Index of Bird Types
        query_1 = "select * from bird_type"
        db_cursor.execute(query_1)
        typ = pandas.DataFrame(list(db_cursor.fetchall()),columns=['bird_type_id','description','min_weight'
            ,'max_weight','z_value'])

        # Index of Bird Types
        query_2 = "select * from product_group"
        db_cursor.execute(query_2)
        pg = pandas.DataFrame(list(db_cursor.fetchall()), columns=['pgroup_id','description','n_parts','section_id'])

        query_3 = "select * from section"
        db_cursor.execute(query_3)
        sec = pandas.DataFrame(list(db_cursor.fetchall()), columns=['section_id','description'])

        query_4 = "select * from cutting_pattern"
        db_cursor.execute(query_4)
        cp = pandas.DataFrame(list(db_cursor.fetchall()), columns=['cp_id','cp_line','section_id','description',
                                                                    'capacity','ops_cost'])

        query_5 = "select * from weight_range"
        db_cursor.execute(query_5)
        ranges = pandas.DataFrame(list(db_cursor.fetchall()), columns=['range_id','description','bird_type_id'])
    else:
        # Index of Bird Types
        typ = pandas.read_csv("../input_files/bird_type.csv")
        pg = pandas.read_csv("../input_files/product_group.csv")
        sec = pandas.read_csv("../input_files/section.csv")
        cp = pandas.read_csv("../input_files/cutting_pattern.csv")
        ranges = pandas.read_csv("../input_files/weight_range.csv")

    # Index of Bird Types

    typ_dct = typ.set_index("bird_type_id").to_dict(orient = 'index')

    # Index of Product Groups
    pg["section_id"] = pg["section_id"].apply(lambda x:[int(i) for i in x.split(".")])
    pg_dct = pg.set_index("pgroup_id").to_dict(orient = 'index')

    # Index of sections

    sec_dct = sec.set_index("section_id").to_dict(orient = 'index')

    # Index of cutting patterns
    # cp['rate'] = cp['rate'].apply(lambda x: round(x,5))
    cp['capacity'] = cp['capacity'].apply(lambda x: round(x,5))
    cp["section_id"] = cp["section_id"].apply(lambda x:[int(i) for i in x.split(".")])
    cp_dct = cp.set_index("cp_id").to_dict(orient = 'index')

    # Mapping Cutting pattern vs Sections

    # for cut in cp_dct.keys():
    #     df_tmp = cp[(cp.cp_id == cut)]
    #     cp_dct[cut]['section_id'] = list(set(df_tmp['section_id']))

    # Mapping Section vs cutting_pattern >> Inverse of previous
    for s in sec_dct.keys():
        df_tmp = cp[cp.section_id.map(set([s]).issubset)]
        sec_dct[s]['cp_id'] = list(set(df_tmp['cp_id']))

    sec_cut_p_not_available = []  ## >> Need to remove these sections >> Log the warning event and record the list
    for idx in list(sec_dct.keys()):
        if sec_dct[idx]['cp_id'] == []:
            sec_cut_p_not_available.append(idx)
            del sec_dct[idx]                           # Removing Section
    # print (sec_cut_p_not_available)

    for s in sec_dct.keys():
        df_tmp = pg[pg.section_id.map(set([s]).issubset)]
        sec_dct[s]['pgroup_id'] = list(set(df_tmp['pgroup_id']))

    # print (sec_dct)
    # print (cp_dct)

    product_typ = {1:{"description":"Fresh/Chilled"},2:{"description":"Frozen"}}
    marination = {1:{"description":"Marination Type-1"},0:{"description":"No Marination"}}
    c_priority = [1,2]

    ## Distinct Weight range set for flexible bird type products
    ranges['bird_type_id'] = ranges["bird_type_id"].apply(lambda x: [int(y) for y in x.split(".")])
    range_dct = ranges.set_index(["range_id"]).to_dict(orient = "index")

    # Collect the required data in a python object
    with open('../cache/master_data',"rb") as fp:
        master = pickle.load(fp)

    
    master.bird_type = typ_dct
    master.cutting_pattern = cp_dct
    master.section = sec_dct
    master.product_group = pg_dct
    master.marination = marination
    master.c_priority = c_priority
    master.product_typ = product_typ
    master.weight_range = range_dct

    # indexes = {'bird_type':typ_dct,
    #            'cutting_pattern':cp_dct,
    #            'section':sec_dct,
    #            'product_group':pg_dct,
    #            'marination': marination,
    #            'c_priority': c_priority,
    #            'product_typ': product_typ,
    #            'weight_range':range_dct}

    # Dump object in a cache file
    with open("../cache/master_data","wb") as fp:
        pickle.dump(master,fp)

    # Recording Event in the status file
    with open("../update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['indexes'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("../update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print ("SUCESS : index file updated!")

    return None


if __name__=='__main__':
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    import configparser
    config = configparser.ConfigParser()
    config.read('../start_config.ini')
    from inputs import *
    update_masters()
