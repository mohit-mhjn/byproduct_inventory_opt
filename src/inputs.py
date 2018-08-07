class model_input(object):

    def __init__(self):

        # Indexes
        self.cutting_pattern = None
        self.section = None
        self.product_group = None
        self.marination = None
        self.c_priority = None
        self.product_typ = None

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

if __name__=="__main__":
    import os
    import pickle
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    chache_objct = model_input()
