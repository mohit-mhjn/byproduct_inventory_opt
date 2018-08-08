class model_input(object):

    def __init__(self):

        ## OBJ INFO
        self.client = "Knex Inc.Test"
        self.plant = "Hyderabad"

        # Indexes
        self.cutting_pattern = None
        self.section = None
        self.product_group = None
        self.marination = None
        self.c_priority = None
        self.product_typ = None
        self.range_dct = None

        # BOM
        self.yield_data = None
        self.sec_nwb_pg = None
        self.sec_wb_pg = None
        self.iter_combs = None
        self.sec_cp = None
        self.typ_cp = None
        self.pgcptyp = None
        self.typseccp = None
        self.cptyp_pg = None
        self.typseccpp = None
        self.pg_cptyp = None

        # Ageing Parameters
        self.life_dct = None

        # Capacity Parameters
        self.capacity_dct = None

        # Cost Parameters
        self.cost_dct = None

        # Conversion Factor
        self.yld_dct = None

        # Flex type
        self.flex_range1 = None
        self.flex_range2 = None
        self.flex_set = None

    def __repr__(self):
        return self.client + " : " + self.plant

def create_object():
    import os
    import pickle
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    cache_obj = model_input()
    with open("input_files/cache/input_object","wb") as fp:
        pickle.dump(cache_obj,fp)

if __name__=="__main__":
    print ("This module is the class definition of input object in the main program!")
    create_object()
