"""
To generate the conversion factor from a bird type to the weight its individual parts
Conv. Factor is used for Pre-processing of sales orders and Post processing of inventroy

The values remain static untill yield(BOM)/index is updated. Hence the key:value object yld_dct is cached in a "conv_factor" file

Direct Execution of the program will call function : "update_conv_factor" to do the self explanatory job.
Indirectly, function "get_conv_factor" is used to import the cached file, parse it and return the dataframe object

While updating the data, the timestamp of the update event is stored in update_status file

Function names are self explanatory
Output
key : (product_group,size)
value : conversion_factor
"""

import pandas
import pickle
import json
import datetime

def update_conv_factor():
    if bool(int(config['input_source']['mysql'])):
        import MySQLdb
        db = MySQLdb.connect(host=config['db']['host'], database=config['db']['db_name'], user=config['db']['user'],
                             password=config['db']['password'])
        db_cursor = db.cursor()

        # Index of Bird Types
        query_1 = "select * from yield"
        db_cursor.execute(query_1)
        yld = pandas.DataFrame(list(db_cursor.fetchall()), columns=['cp_id','section_id','pgroup_id','n_parts','bird_type_id','yield_p'])
        query_2 = "select * from bird_type"
        db_cursor.execute(query_2)
        type_master = pandas.DataFrame(list(db_cursor.fetchall()),
                               columns=['bird_type_id','description','min_weight','max_weight','z_value'])

    else:
        yld = pandas.read_csv("../input_files/yield.csv")
        type_master = pandas.read_csv("../input_files/bird_type.csv")

    # Obtaining avg yield at product group and bird type level
    yld = yld.filter(items = ["pgroup_id","bird_type_id","yield_p"])
    yld = yld.groupby(by = ["pgroup_id","bird_type_id"]).mean()
    yld.reset_index(inplace = True, drop= False)

    # Obtain bird type
    type_master = type_master.filter(items = ["bird_type_id","max_weight"])

    # Merge Yield and Bird type
    yld = yld.merge(type_master,on = "bird_type_id")
    yld['conv_factor'] = round(yld["yield_p"]*yld["max_weight"],4)
    yld = yld.filter(items = ["pgroup_id","bird_type_id","conv_factor"])
    yld_dct = yld.to_dict(orient="records")

    with open("../cache/master_data","rb") as fp:
        master = pickle.load(fp)

    master.yld_dct = yld_dct

    with open("../cache/master_data","wb") as fp:
        pickle.dump(master,fp)

    # Recording Event in the status file
    with open("../update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['conv_factor'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("../update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)
    print ("SUCCESS : Conversion Factor updated!")
    return None

# def get_conv_factor():
#     with open("../cache/conv_factor","rb") as fp:
#         a = pickle.load(fp)
#     yld = pandas.DataFrame(a)
#     return yld

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    # get_conv_factor()
    import configparser
    config = configparser.ConfigParser()
    config.read('../start_config.ini')
    from inputs import *
    update_conv_factor()
