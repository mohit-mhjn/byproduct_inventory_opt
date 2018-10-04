import datetime

class master_input(object):

    def __init__(self):

        ## OBJ INFO
        self.client = "Knex Inc.Test"
        self.plant = "Hyderabad"

        # Indexes
        self.bird_type = None
        self.cutting_pattern = None
        self.section = None
        self.product_group = None
        self.marination = None
        self.c_priority = None
        self.product_typ = None
        self.weight_range = None

        # BOM
        self.yield_data = None
        self.sec_nwb_pg = None
        self.sec_wb_pg = None
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

class decision_input(object):

    def __init__(self,dt,lenth_of_plan):
        self.exec_date = dt

        self.horizon = [dt+datetime.timedelta(days = i) for i in range(lenth_of_plan)]
        self.t_dt_map = {t: str(self.horizon[t]) for t in range(lenth_of_plan)}
        self.dt_t_map = {self.horizon[k]:k for k,v in self.t_dt_map.items()}

        self.orders_aggregate = None
        self.order_breakup = None
        self.order_grouped = None

        self.flex_orders_aggregate = None
        self.flex_order_grouped = None
        self.flex_order_breakup = None

        self.bird_availability = None
        self.part_inv_fresh = None
        self.part_inv_frozen = None

def create_object(cache_path):
    cache_obj = master_input()
    with open(cache_path,"wb") as fp:
        pickle.dump(cache_obj,fp)

if __name__=="__main__":
    print ("IMPORTED : class definition of input objects in the main program!")
    import os
    import pickle
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    cache_path = "../cache/master_data"
    create_object(cache_path)
    print ("SUCEESS: input object created")
    exit(0)
