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

    from index_reader import read_masters
    typ_ranges = read_masters()["typ_ranges"]
    ranges = pandas.DataFrame(typ_ranges).T
    ranges.reset_index(inplace = True, drop = False)
    ranges = ranges.rename(columns = {'index':'rng_indx'})

    flex_range1 = ranges.set_index(["rng_indx"]).to_dict(orient = "dict")["bird_typ"]
    flex_range2 = invert_dict(flex_range1)
    flex_set = [(v,k) for k in flex_range2.keys() for v in flex_range2[k]]

    rng_dct = { "rng_to_type":flex_range1,
                "type_to_rng":flex_range2,
                "flex_rng_comb":flex_set}

    #Cacheing the objects
    with open("input_files/flex_set","wb") as fp:
        pickle.dump(rng_dct,fp)

    # Recording Event in the status file
    with open("input_files/update_status.json","r") as jsonfile:
        us = dict(json.load(jsonfile))
        us['flex_set'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

    with open("input_files/update_status.json","w") as jsonfile:
        json.dump(us,jsonfile)

    print("SUCCESS : flexible bird type combinations updated!")

    return None


def read_flex_typ():
    # Read the cached file
    with open("input_files/flex_set","rb") as fp:
        rng_dct = pickle.load(fp)
    return rng_dct

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    update_flex_typ()
    print (read_flex_typ())
