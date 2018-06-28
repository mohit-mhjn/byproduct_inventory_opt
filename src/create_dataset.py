import csv
import os
import itertools
import pandas
import random
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

print (os.getcwd())

with open("input_files/bird_type.csv") as infile1:
    k1 = list(csv.reader(infile1))
    k1 = list(map(lambda x:x[0][0],list(zip(k for k in k1))[1:]))

with open("input_files/product_group.csv") as infile2:
    k2 = list(csv.reader(infile2))
    k2 = list(map(lambda x:x[0], k2[1:]))

# a = pandas.DataFrame(list(itertools.product(k2,k1)))

# a.columns = ['product_group','bird_type']
# a['Inventory'] = 0
# a.to_csv("input_files/inventory.csv", index= False)
#
# b_lst = list(itertools.product(range(0,10),k1,k2))
#
# b = pandas.DataFrame(random.sample(b_lst,100))
# b.columns = ['customer_number','bird_type','product_group']
# b['order_number'] = 'Not Available'
# b['order_count'] = 0
# b['order_weight'] = 0
# b.to_csv("input_files/sales_order.csv", index = False)

A = pandas.read_excel("../PP_Model/table_yield.xlsx", sheet_name = 'yield2')
A = A[(A.yield_p >= 0)]
A = A.filter(items = ["product_group","cutting_pattern","n_parts","section","yield_p"])
A = A.groupby(by = ["product_group","cutting_pattern","section"]).max()
A.reset_index(inplace = True, drop = False)

print (A)