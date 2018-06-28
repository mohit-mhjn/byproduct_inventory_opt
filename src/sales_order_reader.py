import pandas
import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

def get_orders():
    s = pandas.read_csv("sales_order.csv")
    print (s)
    return None

get_orders()



Size,PG >> Weight
