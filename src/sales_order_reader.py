"""
To read the sales order file and transform the dataset to the required format

function get_orders is self explanatory

Sales Order is avaiable in the format date,customer,sku_index,order_weight/order_count
Merging with SKU Master to match the sku details as : product group, Bird Size, Marination, fresh/frozen
Merging with Conversion Factor to get multiplier to transform number to weight
Merging with Customer Master to Get Customer Priority

The output format of sales orders is grouped by index >> (date,priority,Fresh/Frozen,marination,product_group,bird_type)

Note:
conversion factor is used from the cached file and read by conv_factor program
"""

import pandas

def get_orders():
    # Get Orders
    orders = pandas.read_csv("input_files/sales_order.csv")

    # Get Conversion Factor
    from conv_factor import get_conv_factor
    yld = get_conv_factor()

    # Get Customer Masters
    c_master = pandas.read_csv("input_files/customer.csv")
    c_master = c_master.filter(items = ["customer_number","priority"])

    # Get SKU Master
    i_master = pandas.read_csv("input_files/sku_master.csv")
    i_master = i_master.filter(items = ["sku_index","prod_group_index","bird_type_index","product_type","marination"])

    #Merge Conv Factor to orders
    orders = orders.merge(i_master, on = "sku_index")
    orders = orders.merge(c_master, on = "customer_number")
    orders = orders.merge(yld, on =["prod_group_index","bird_type_index"])
    orders['order_qty'] = orders.apply(lambda row: max(row['order_weight'],row['order_count']*row['conv_factor']),axis = 1) # Converting orders to weight
    orders.drop(labels = ["order_number","customer_number","sku_index","order_weight","order_count","conv_factor"], axis = 1, inplace = True)
    orders = orders.groupby(by = ["date","priority","product_type","marination","prod_group_index","bird_type_index"]).sum()
    # print (orders)
    return orders

if __name__=="__main__":
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    orders = get_orders()
    # print (orders)
    print ("SUCCESS : orders imported!")
