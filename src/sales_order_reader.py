"""
To read the sales order file and transform the dataset to the required format

function get_orders is self explanatory

Sales Order is avaiable in the format date,customer,sku_index,order_weight/order_count
Merging with SKU Master to match the sku details as : product group, Bird Size, Marination, fresh/frozen, order selling price
Merging with Conversion Factor to get multiplier to transform number to weight
Merging with Customer Master to Get Customer Priority

The output is exported in 3 forms:

Agreegated by customer_priority, SKU_key
Individual order line items
Against a customer_priority, SKU_key >> group of order numbers

output ref:

orders = {"strict/flexible" : {"aggregate":, "breakup":, "grouped_by_product"} }

The output format of agreegate sales orders is grouped by index >> (date,priority,Fresh/Frozen,marination,product_group,bird_type)

Note:
conversion factor is used from the cached file and read by conv_factor program
"""

import pandas
import datetime
import warnings

def get_orders(indexes, horizon):
    # Get Orders
    orders = pandas.read_csv("input_files/sales_order.csv")
    if orders.empty:
        raise ImportError("No orders found; error code: 100A")

    # Get Conversion Factor
    from conv_factor import get_conv_factor
    yld = get_conv_factor()

    # Getting Flexible size ranges
    from index_reader import read_masters
    ro = read_masters()
    flexible_types = list(ro["typ_ranges"].keys())

    # Get Customer Masters
    c_master = pandas.read_csv("input_files/customer.csv")
    c_master = c_master.filter(items = ["customer_number","priority","customer_sla"])

    # Get SKU Master
    i_master = pandas.read_csv("input_files/sku_master.csv")
    i_master = i_master.filter(items = ["sku_index","prod_group_index","bird_type_index","product_type","marination","selling_price"]) # Temporary getting selling price from SKU_Master
                                                                                                                                       # Planning to get this from Customer SKU Pricing table (create new)
    #Merge Conv Factor to orders
    orders['date'] = orders['date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").date())
    orders = orders[orders['date'].isin(horizon)]
    orders['date'] = orders['date'].apply(lambda x: str(x))
    orders = orders.merge(i_master, on = "sku_index")
    orders = orders.merge(c_master, on = "customer_number")

    # >> Separate Flex Type SKU orders Here >>
    ## Processing flex first
    flex_orders = orders[(orders.bird_type_index > 99)] # >> memoized this, will process it afterwards
    orders = orders.drop(flex_orders.index)

    ## NON FLEXIBLE/STRICT SIZE TYPE ############################

    # Continuing with Non-Flex type SKU orders >> Covnversion Only possible with strict bird type SKU Categoreis
    orders = orders.merge(yld, on =["prod_group_index","bird_type_index"])
    orders['order_qty'] = 0
    if not orders.empty:
        orders['order_qty'] = orders.apply(lambda row: max(row['order_weight'],row['order_count']*row['conv_factor']),axis = 1) # Converting orders to weight
        orders = orders[(orders.order_qty > 0)]
    else:
        warnings.warn("No orders found; error code: 100B")

    if orders.empty:  # Uncomment if empty orders is not permissible
        warnings.warn("No orders found; error code: 100C")

    order_breakup_df = orders.filter(items = ["date","priority","order_number","prod_group_index","bird_type_index","product_type","marination","customer_sla","selling_price","order_qty"])
    order_breakup = order_breakup_df.set_index(["order_number"]).to_dict(orient = 'index')
    order_group_df = order_breakup_df.groupby(by = ["date","priority","prod_group_index","bird_type_index","product_type","marination"])["order_number"].apply(list)
    # order_group_df = order_breakup_df.groupby(by = ["date","priority"])["order_number"].apply(list)
    order_group = order_group_df.to_dict()

    orders.drop(labels = ["order_number","customer_number","sku_index","order_weight","order_count","conv_factor"], axis = 1, inplace = True)
    orders = orders.groupby(by = ["date","priority","prod_group_index","bird_type_index","product_type","marination"]).sum()
    order_dct = orders.to_dict(orient = 'dict')['order_qty']

    ## FLEXIBLE SIZE/TYPE ##############################

    # > 99 are flex type orders >> As defined in flex type
    ## Orders are avaialble in weight (Assumption)
    flex_orders = flex_orders.filter(items = ["date","priority","order_number","prod_group_index","bird_type_index","product_type","marination","customer_sla","selling_price","order_weight"])
    flex_orders['order_qty'] = flex_orders['order_weight']
    flex_orders = flex_orders[(flex_orders.order_qty > 0)]
    if flex_orders.empty:
        warnings.warn("flexible size type orders not found, error code: 100D")

    # print(flex_orders)
    flex_order_breakup_df = flex_orders.filter(items=["date","priority","order_number","prod_group_index","bird_type_index","product_type","marination","customer_sla","selling_price","order_qty"])
    flex_order_breakup = flex_order_breakup_df.set_index(["order_number"]).to_dict(orient = 'index')

    flex_order_group_df = flex_order_breakup_df.groupby(by = ["date","priority","prod_group_index","bird_type_index","product_type","marination"])["order_number"].apply(list)
    flex_order_group = flex_order_group_df.to_dict()

    flex_orders.drop(labels = ["order_number","order_weight"], axis = 1, inplace = True)
    flex_orders = flex_orders.groupby(by = ["date","priority","prod_group_index","bird_type_index","product_type","marination"]).sum()
    flex_order_dct = flex_orders.to_dict(orient = 'dict')['order_qty']

    return {"strict":{'aggregate':order_dct,"breakup":order_breakup,"grouped_by_product":order_group},
            "flexible":{'aggregate':flex_order_dct,"breakup":flex_order_breakup,"grouped_by_product":flex_order_group}}


if __name__=="__main__":

    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    from index_reader import read_masters
    indexes = read_masters()
    horizon = [datetime.date(2018,6,28),datetime.date(2018,6,29),datetime.date(2018,6,30)]
    orders = get_orders(indexes,horizon)

    print ("\nFLEXIBLE BIRD SIZE/TYPE SKU >>>\n")
    print (" Format 1 :")
    print (orders['flexible']["aggregate"])
    print (" Format 2 :")
    print (orders['flexible']["breakup"])
    print (" Format 3 :")
    print (orders['flexible']["grouped_by_product"])

    print ("\nSTRICT BIRD SIZE/TYPE SKU >>>\n")
    print (" Format 1 :")
    print (orders['strict']["aggregate"])
    print (" Format 2 :")
    print (orders['strict']["breakup"])
    print (" Format 3 :")
    print (orders['strict']["grouped_by_product"])

    print ("SUCCESS : orders imported!")
