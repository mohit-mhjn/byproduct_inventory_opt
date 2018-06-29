"""
To read/update the bill_of_material/yield file and store as `bom_file` among input files
The BOM data is primarily static and changes are not so frequent, plus the dataset is large. Hence, it is cached

Note:
If any master is updated then the first >> indexes, second >> BOM needs to be updated

Direct Execution of this function will do the required changes in the bom_file
Indirect execution will just read the bom_file and return the file

function : update_combinations(), self explanatory

    function : get_product_set()
                it will render the non-whole bird product group associated with a particular section cutting pattern combination
                example:
                    (sec=2,cp=1) >> (RIB,KEEL)
    function : get_whole_product_set()
                it will render the whole bird product group associated with a particular section cutting pattern combination
                example:
                    (sec=2,cp=1) >> (PIECE CUT)

    function : yield_generator()
                it will render the yield as a percent
                If the non-dressed bird of type >> typ
                       Cut with cutting pattern >> cp
                       The product group >> pg
                       Will be yielded in (y)*weight(typ) KG's

                Note : In the output dict y is the value agains the key (pg,cp,typ)


function : read_combinations(), self explanatory

Required combinations for iteration in model:
1. section vs cutting_pattern: sec_cp_comb
2. (cutting_pattern,bird_type) vs product_group : cptyp_pg_comb
3. product_group vs (cutting_pattern,bird_type) : pg_cptyp_comb

Pickle is used here to serialize the dict object (couln't use json because the keys are tuples)
"""

import pandas
import json
import pickle

def update_combinations():

    def get_product_set(i,y):
        k = i[0]
        j = i[1]
        df_tmp = y[y.section.map(set([k]).issuperset)]
        df_tmp = df_tmp[(df_tmp.cutting_pattern == j)]
        l = set(df_tmp['prod_group_index'])
        return l

    def get_whole_product_set(h,y):
        k = h[0]
        j = h[1]
        df_tmp = y[y.section.map(set([k]).issubset)]
        df_tmp = df_tmp[(df_tmp.cutting_pattern == j)]
        l = set(df_tmp['prod_group_index']) - sc_pg1[h]
        return l

    def yield_generator(y):
        df_tmp = y.copy(deep=True)
        df_tmp['yld'] = y["yield_p"]*y["n_parts"]
        df_tmp = df_tmp.filter(items=["prod_group_index","cutting_pattern","bird_type_index","yld","yield_p"])  # Catch Errors here >> Resetting Index shouldn't change n_rows
        df_tmp.set_index(["prod_group_index","cutting_pattern","bird_type_index"], inplace = True)
        yd = df_tmp.to_dict(orient = "index")
        return yd

    y = pandas.read_csv('input_files/yield.csv')
    y["yield_p"] = y["yield_p"].apply(lambda x: round(x,2))
    y = y[y.yield_p > 0]   # Safeguard
    y["section"] = y["section"].apply(lambda x: [int(i) for i in str(x)])

    with open("input_files/index_file.json","r") as jsonfile:
        indx = dict(json.load(jsonfile))
        section_data = indx['section']
        pg_data = indx["product_group"]
        cp_data = indx["cutting_pattern"]
        b_typ_data = indx["bird_type"]

    sec_cp_comb = set()   # Set of section and favourable cutting patterns (section,cutting_pattern)
    for sec in section_data.keys():
        for cp in section_data[sec]['cutting_pattern']:
            sec_cp_comb.add((int(sec),int(cp)))

    sc_pg1 = {i:get_product_set(i,y) for i in sec_cp_comb}  # Cutting Pattern used on a section yields a set of product group (Non Whole Bird Entities)
    sc_pg2 = {h:get_whole_product_set(h,y) for h in sec_cp_comb}  # Cutting Pattern used on a section yields a set of product group (Whole Bird Entities)

    yd = yield_generator(y)
    pgcptyp_comb = list(yd.keys())

    cptyp_pg_comb = {(int(j),int(r)):set() for j in cp_data.keys() for r in b_typ_data.keys()}
    pg_cptyp_comb = {int(p):set() for p in pg_data.keys()}
    for i in pgcptyp_comb:
        cptyp_pg_comb[(i[1],i[2])].add(i[0])
        pg_cptyp_comb[i[0]].add((i[1],i[2]))

    bom_data = {'yield_data':yd,
                'sec_nwb_pg':sc_pg1,
                'sec_wb_pg':sc_pg2,
                'iter_combs':{
                'sec_cp':sec_cp_comb,
                'cptyp_pg':cptyp_pg_comb,
                'pg_cptyp':pg_cptyp_comb}}

    with open("input_files/bom_file","wb") as fp:
        pickle.dump(bom_data,fp)
    print("SUCCESS : bom file updated!")
    return None

def read_combinations():
    with open("input_files/bom_file","rb") as fp:
        bom_data = dict(pickle.load(fp))
    return bom_data

if __name__=='__main__':
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    # update_combinations()
    # read_combinations()
