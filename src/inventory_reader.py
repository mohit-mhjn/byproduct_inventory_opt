"""
To Extract inventroy data from the source files
function: get_birds() will import the inventory of live birds by date by size to the processing unit
function: get_parts() will import the initial available inventory of parts by product group by size at T=0

Direct Execution of this file will just read the data and print
In Indirect Execution it will be required to call the corresponding functions
"""

import pandas

# def read_data():
#     global inv
#     global indx
#     inv = pandas.read_csv("input_files/inventory.csv")
#     pg = pandas.read_csv("input_files/product_group.csv")
#     pg_subset = pg[(pg.product_group == 'LIVE_BIRD')]
#     indx = set(pg_subset['prod_group_index'])
#     return None

def get_birds():
    # global inv
    # global indx
    # tbl = inv[inv['product_group'].isin(indx)]
    tbl = pandas.read_csv("input_files/birds_available.csv")
    tbl.reset_index(inplace=True,drop=True)
    tbl_dct = tbl.set_index(['date','bird_type']).to_dict(orient='dict')['inventory']
    return tbl_dct

def get_parts():
    # global inv
    # global indx
    # tbl = inv[inv['product_group'].isin(indx) == False]
    tbl = pandas.read_csv("input_files/inventory.csv")
    tbl.reset_index(inplace=True,drop=True)
    tbl_dct = tbl.set_index(['product_group','bird_type']).to_dict(orient='dict')['inventory']
    return tbl_dct

if __name__=='__main__':
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    # read_data()
    print ("Bird Inventory >>>")
    print (get_birds())
    print ("\nPart Inventory >>>")
    print (get_parts())
# else:
#     read_data()
