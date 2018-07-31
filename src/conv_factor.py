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
    # Obtaining avg yield at product group and bird type level
    yld = pandas.read_csv("input_files/yield.csv")
    yld = yld.filter(items = ["prod_group_index","bird_type_index","yield_p"])
    yld = yld.groupby(by = ["prod_group_index","bird_type_index"]).mean()
    yld.reset_index(inplace = True, drop= False)

    # Obtain bird type
    type_master = pandas.read_csv("input_files/bird_type.csv")
    type_master = type_master.filter(items = ["bird_type_index","abs_weight"])

    # Merge Yield and Bird type
    yld = yld.merge(type_master,on = "bird_type_index")
    yld['conv_factor'] = round(yld["yield_p"]*yld["abs_weight"],4)
    yld = yld.filter(items = ["prod_group_index","bird_type_index","conv_factor"])
    yld_dct = yld.to_dict(orient="records")

    with open("input_files/conv_factor","wb") as fp:
        pickle.dump(yld_dct,fp)

    # Recording Event in the status file
    with open("input_files/update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['conv_factor'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("input_files/update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)
    print ("SUCCESS : Conversion Factor updated!")
    return None

def get_conv_factor():
    with open("input_files/conv_factor","rb") as fp:
        a = pickle.load(fp)
    yld = pandas.DataFrame(a)
    return yld

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    # get_conv_factor()
    update_conv_factor()
