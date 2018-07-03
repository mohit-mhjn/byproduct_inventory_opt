"""
To check Inventory, check Orders (may be forecast) to Generate Demand for the model

"""
import pandas
import datetime
import json

def get_demand():

    from sales_order_reader import get_orders
    from inventory_reader import get_parts

    part_inv = get_parts()
    orders = get_orders()

    orders_p1 = orders[(orders.priority == 1)]
    orders_p1 = orders_p1.drop(labels = ["priority","marination"],axis = 1)
    orders_p1 = orders_p1.groupby(by = ["date","product_type","prod_group_index","bird_type_index"]).sum()
    orders_p1.reset_index(inplace = True, drop = False)

    orders_p2 = orders[(orders.priority==2)]
    orders_p2 = orders_p2.drop(labels = ["priority","marination"],axis = 1)
    orders_p2 = orders_p2.groupby(by = ["date","product_type","prod_group_index","bird_type_index"]).sum()
    orders_p2.reset_index(inplace = True, drop = False)







    pass

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    get_demand()



"""


    orders_p1.rename(columns = {'orders_qty':'orders1'})


    orders_p1.rename(columns = {'orders_qty':'orders2'})

    dt_all = orders['date'].unique()
    dt_all.sort()

    for dt in dt_all:
        dt_inv = list(part_inv['date'].unique())[0]
        diff = (dt - dt_inv).days
        part_inv['inv_age'] = part_inv['inv_age'].apply(lambda x: x + diff)
        inv_tbl = part_inv[(part_inv.inv_age < part_inv.shelf_life)]
        inv_tbl = inv_tbl.drop(labels = ["inv_age","date"],axis = 1)
        inv_tbl = inv_tbl.groupby(by = ["product_type","prod_group_index","bird_type_index"]).sum()
        inv_tbl.reset_index(inplace = True, drop = False)
        inv_tbl['date'] = dt
        demand_supply = inv_tbl.merge(orders_p1, on =["date","product_type","prod_group_index","bird_type_index"])
        demand_supply = demand_supply.merge(orders_p1, on =["date","product_type","prod_group_index","bird_type_index"])
        print (demand_supply)

"""
