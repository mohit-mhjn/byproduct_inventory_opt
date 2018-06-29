import pandas

def get_orders():
    s = pandas.read_csv("input_files/sales_order.csv")
    return s
