"""
flex_range2 = {1:["FLEX2"],2:["FLEX1","FLEX2"],3:["FLEX1","FLEX2"],4:["FLEX1","FLEX2"],5:["FLEX2"],6:["FLEX2"]}
flex_range1 = {"FLEX1":[2,3,4],"FLEX2":[1,2,3,4,5,6]}                     #  Testing
flex_set = [("FLEX1",2),("FLEX1",3),("FLEX1",4),("FLEX2",1),("FLEX2",2),("FLEX2",3),("FLEX2",4),("FLEX2",5),("FLEX2",6)]
"""
import pandas
import pickle
import json
import datetime

def update_flex_typ():

    def invert_dict(d):
        inverse = dict()
        for key in d:
            # Go through the list that is saved in the dict:
            for item in d[key]:
                # Check if in the inverted dict the key exists
                if item not in inverse:
                    # If not create a new list
                    inverse[item] = [key]
                else:
                    inverse[item].append(key)
        return inverse

    with open("../cache/master_data","rb") as fp:
        master = pickle.load(fp)

    # from index_reader import read_masters
    # typ_ranges = read_masters()["weight_range"]
    typ_ranges = master.weight_range
    ranges = pandas.DataFrame(typ_ranges).T
    ranges.reset_index(inplace = True, drop = False)
    ranges = ranges.rename(columns = {'index':'range_id'})

    flex_range1 = ranges.set_index(["range_id"]).to_dict(orient = "dict")["bird_type_id"]
    flex_range2 = invert_dict(flex_range1)
    flex_set = [(v,k) for k in flex_range2.keys() for v in flex_range2[k]]

    master.flex_range1 = flex_range1
    master.flex_range2 = flex_range2
    master.flex_set = flex_set

    # rng_dct = { "rng_to_type":flex_range1,
    #             "type_to_rng":flex_range2,
    #             "flex_rng_comb":flex_set}

    #Cacheing the objects

    with open("../cache/master_data","wb") as fp:
        pickle.dump(master,fp)

    # Recording Event in the status file
    with open("../update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['flex_set'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("../update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print("SUCCESS : flexible bird type combinations updated!")

    return None


# def read_flex_typ():
#     # Read the cached file
#     with open("../cache/weight_set","rb") as fp:
#         rng_dct = pickle.load(fp)
#     return rng_dct

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    import configparser
    config = configparser.ConfigParser()
    config.read('../start_config.ini')
    from inputs import *
    update_flex_typ()
