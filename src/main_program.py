import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)
from sales_order_reader import get_orders
from inventory_reader import get_birds, get_parts
from index_reader import read_masters
from BOM_reader import read_combinations

orders = get_orders()
birds_inv = get_birds()
parts_inv = get_parts()

print (birds_inv)
print (parts_inv)
