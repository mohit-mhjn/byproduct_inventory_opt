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

def get_orders(master,var_data,config):

    horizon = var_data.horizon

    # Get Orders
    if bool(int(config['input_source']['mySQL'])):
        import MySQLdb
        db = MySQLdb.connect(host=config['db']['host'], database=config['db']['db_name'], user=config['db']['user'],
                             password=config['db']['password'])
        db_cursor = db.cursor()

        # Index of Bird Types
        query_1 = "select * from sales_order"
        db_cursor.execute(query_1)
        orders = pandas.DataFrame(list(db_cursor.fetchall()), columns=['order_dt','bill_number','customer_id','sku_id','part_count','part_weight'])

        query_2 = "select * from sku_master"
        db_cursor.execute(query_2)
        i_master = pandas.DataFrame(list(db_cursor.fetchall()), columns=['sku_id','pgroup_id','bird_type_id','product_type','marination','active','selling_price','holding_cost','shelf_life'])

        query_3 = "select * from customer"
        db_cursor.execute(query_3)
        c_master = pandas.DataFrame(list(db_cursor.fetchall()), columns=['customer_id','description','priority','serv_agrmnt'])

    else:
        orders = pandas.read_csv("../input_files/sales_order.csv")
        i_master = pandas.read_csv("../input_files/sku_master.csv") # Getting Inventory Shelf Life
        c_master = pandas.read_csv("../input_files/customer.csv")

    if orders.empty:
        raise ImportError("No orders found; error code: 100A")

    # Get Conversion Factor
    yld = pandas.DataFrame(master.yld_dct)

    # Getting Flexible size ranges
    flexible_types = list(master.weight_range.keys())

    # Get Customer Masters

    c_master = c_master.filter(items = ["customer_id","priority","serv_agrmnt"])

    # Get SKU Master
    i_master = i_master.filter(items = ["sku_id","pgroup_id","bird_type_id","product_type","marination","selling_price"]) # Temporary getting selling price from SKU_Master
                                                                                                                                       # Planning to get this from Customer SKU Pricing table (create new)
    #Merge Conv Factor to orders
    orders['order_dt'] = orders['order_dt'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").date())
    orders = orders[orders['order_dt'].isin(horizon)]
    orders['order_dt'] = orders['order_dt'].apply(lambda x: str(x))
    orders = orders.merge(i_master, on = "sku_id")
    orders = orders.merge(c_master, on = "customer_id")

    # >> Separate Flex Type SKU orders Here >>
    ## Processing flex first
    flex_orders = orders[(orders.bird_type_id > 99)] # >> memoized this, will process it afterwards
    orders = orders.drop(flex_orders.index)

    ## NON FLEXIBLE/STRICT SIZE TYPE ############################

    # Continuing with Non-Flex type SKU orders >> Covnversion Only possible with strict bird type SKU Categoreis
    # print(orders.columns,yld.columns)
    orders = orders.merge(yld, on =["pgroup_id","bird_type_id"])
    orders['order_qty'] = 0
    if not orders.empty:
        orders['order_qty'] = orders.apply(lambda row: max(row['part_weight'],row['part_count']*row['conv_factor']),axis = 1) # Converting orders to weight
        orders = orders[(orders.order_qty > 0)]
    else:
        warnings.warn("No orders found; error code: 100B")

    if orders.empty:  # Uncomment if empty orders is not permissible
        warnings.warn("No orders found; error code: 100C")

    order_breakup_df = orders.filter(items = ["order_dt","priority","bill_number","pgroup_id","bird_type_id","product_type","marination","serv_agrmnt","selling_price","order_qty"])
    order_breakup = order_breakup_df.set_index(["bill_number"]).to_dict(orient = 'index')
    order_group_df = order_breakup_df.groupby(by = ["order_dt","priority","pgroup_id","bird_type_id","product_type","marination"])["bill_number"].apply(list)
    # order_group_df = order_breakup_df.groupby(by = ["order_dt","priority"])["bill_number"].apply(list)
    order_group = order_group_df.to_dict()

    orders.drop(labels = ["bill_number","customer_id","sku_id","part_weight","part_count","conv_factor"], axis = 1, inplace = True)
    orders = orders.groupby(by = ["order_dt","priority","pgroup_id","bird_type_id","product_type","marination"]).sum()
    order_dct = orders.to_dict(orient = 'dict')['order_qty']

    ## FLEXIBLE SIZE/TYPE ##############################

    # > 99 are flex type orders >> As defined in flex type
    ## Orders are avaialble in weight (Assumption)
    flex_orders = flex_orders.filter(items = ["order_dt","priority","bill_number","pgroup_id","bird_type_id","product_type","marination","serv_agrmnt","selling_price","part_weight"])
    flex_orders['order_qty'] = flex_orders['part_weight']
    flex_orders = flex_orders[(flex_orders.order_qty > 0)]
    if flex_orders.empty:
        warnings.warn("flexible size type orders not found, error code: 100D")

    # print(flex_orders)
    flex_order_breakup_df = flex_orders.filter(items=["order_dt","priority","bill_number","pgroup_id","bird_type_id","product_type","marination","serv_agrmnt","selling_price","order_qty"])
    flex_order_breakup = flex_order_breakup_df.set_index(["bill_number"]).to_dict(orient = 'index')

    flex_order_group_df = flex_order_breakup_df.groupby(by = ["order_dt","priority","pgroup_id","bird_type_id","product_type","marination"])["bill_number"].apply(list)
    flex_order_group = flex_order_group_df.to_dict()

    flex_orders.drop(labels = ["bill_number","part_weight"], axis = 1, inplace = True)
    flex_orders = flex_orders.groupby(by = ["order_dt","priority","pgroup_id","bird_type_id","product_type","marination"]).sum()
    flex_order_dct = flex_orders.to_dict(orient = 'dict')['order_qty']

    ## Append Data in Var_data object :
    var_data.orders_aggregate = order_dct
    var_data.order_breakup = order_breakup
    var_data.order_grouped = order_group

    var_data.flex_orders_aggregate = flex_order_dct
    var_data.flex_order_grouped = flex_order_group
    var_data.flex_order_breakup = flex_order_breakup

    # return {"strict":{'aggregate':order_dct,"breakup":order_breakup,"grouped_by_product":order_group},
    #        "flexible":{'aggregate':flex_order_dct,"breakup":flex_order_breakup,"grouped_by_product":flex_order_group}}
    return var_data


if __name__=="__main__":
    import pickle
    import os
    from inputs import *
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    # from index_reader import read_masters
    # indexes = read_masters()
    # horizon = [datetime.date(2018,6,28),datetime.date(2018,6,29),datetime.date(2018,6,30)]
    import configparser
    config = configparser.ConfigParser()
    config.read('../start_config.ini')

    with open("../cache/master_data","rb") as fp:
        master = pickle.load(fp)

    var_data = decision_input(datetime.date(2018,6,28),3)
    var_data = get_orders(master,var_data,config)

    print ("\nFLEXIBLE BIRD SIZE/TYPE SKU >>>\n")
    print (" Format 1 :")
    print (var_data.flex_orders_aggregate)
    print (" Format 2 :")
    print (var_data.flex_order_breakup)
    print (" Format 3 :")
    print (var_data.flex_order_grouped)

    print ("\nSTRICT BIRD SIZE/TYPE SKU >>>\n")
    print (" Format 1 :")
    print (var_data.orders_aggregate)
    print (" Format 2 :")
    print (var_data.order_breakup)
    print (" Format 3 :")
    print (var_data.order_grouped)

    print ("SUCCESS : orders imported!")
