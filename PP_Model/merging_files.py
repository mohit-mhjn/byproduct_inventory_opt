## merging files

import pandas as pd
import os

#1. Fresh sold
fresh_sold = pd.read_csv('Fresh_Product_Sold.csv')
fresh_sold['keyfigure'] = "fresh_sold"
fresh_sold[['Products','Customer']] = pd.DataFrame(fresh_sold.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
del fresh_sold['days']


#2. Frozen sold
frozen_sold = pd.read_csv("Frozen_Product_Sold.csv")
frozen_sold['keyfigure'] = "frozen_sold"
frozen_sold[['Products','Customer']] = pd.DataFrame(frozen_sold.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
del frozen_sold['days']


#3. Unsatisfied Demand Fresh
unsatisfied_fresh = pd.read_csv("Unsatisfied_Fresh.csv")
unsatisfied_fresh['keyfigure'] = "unsatisfied_fresh"
unsatisfied_fresh[['Products','Customer']] = pd.DataFrame(unsatisfied_fresh.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
del unsatisfied_fresh['days']

#4. Unsatisfied Demand Frozen
unsatisfied_frozen = pd.read_csv("Unsatisfied_Frozen.csv")
unsatisfied_frozen['keyfigure'] = "unsatisfied_frozen"
unsatisfied_frozen[['Products','Customer']] = pd.DataFrame(unsatisfied_frozen.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
del unsatisfied_frozen['days']


#5. Production
production = pd.read_csv("Production.csv")
production['keyfigure'] = "production"
production[['Products','Customer']] = pd.DataFrame(production.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
del production['days']

#6. Frozen Production
frozen_production = pd.read_csv("Frozen_Product_Production.csv")
frozen_production['keyfigure'] = "frozen_production"
frozen_production[['Products','Customer']] = pd.DataFrame(frozen_production.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
del frozen_production['days']

#7. Frozen Inventory
frozen_inventory = pd.read_csv("Frozen_Product_Inventory.csv")
frozen_inventory['keyfigure'] = "frozen_inventory"
frozen_inventory[['Products','Customer']] = pd.DataFrame(frozen_inventory.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
del frozen_inventory['days']

#8. Demand Fresh
demand_fresh = pd.read_excel('Model_Input_Customer.xlsm', sheet_name="Demand_Fresh")
demand_fresh = demand_fresh.dropna(axis='columns', how='all')
demand_fresh['keyfigure'] = "demand_fresh"
demand_fresh[['Products','Customer']] = pd.DataFrame(demand_fresh.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
#demand_fresh.columns = ['Customer','Day_0','Day_1','Day_2','Day_3','Day_4','Products','keyfigure']
del demand_fresh['days']


#9. Demand Frozen
demand_frozen = pd.read_excel('Model_Input_Customer.xlsm', sheet_name="Demand_Frozen")
demand_frozen = demand_frozen.dropna(axis='columns', how='all')
demand_frozen['keyfigure'] = "demand_frozen"
demand_frozen[['Products','Customer']] = pd.DataFrame(demand_frozen.days.str.split('_',1).tolist(),
                                   columns = ['Products','Customer'])
#demand_frozen.columns = ['Customer','Day_0','Day_1','Day_2','Day_3','Day_4','Products','keyfigure']
del demand_frozen['days']

#10. Bird Count
bird_count = pd.read_csv("Bird_Count.csv")
bird_count.columns = ['Days','Number']
bird_count = bird_count.T
bird_count = bird_count.drop("Days")
bird_count.columns = ['Day_1','Day_2','Day_3','Day_4']
bird_count['keyfigure'] = "bird_count"

#11. Satsfied Demand Fresh
temp1 = demand_fresh.drop(['Products','keyfigure','Customer'], axis=1)
temp2 = unsatisfied_fresh.drop(['Products','keyfigure','Customer'], axis=1)
satisfied_fresh = temp1 - temp2
satisfied_fresh['Customer'] = unsatisfied_fresh['Customer']
satisfied_fresh['Products'] = unsatisfied_fresh['Products']
satisfied_fresh['keyfigure'] = 'satisfied_fresh'

#12. Satsfied Demand Frozen

temp1 = demand_frozen.drop(['Products','keyfigure','Customer'], axis=1)
temp2 = unsatisfied_frozen.drop(['Products','keyfigure','Customer'], axis=1)
satisfied_frozen = temp1 - temp2
satisfied_frozen['Customer'] = unsatisfied_frozen['Customer']
satisfied_frozen['Products'] = unsatisfied_frozen['Products']
satisfied_frozen['keyfigure'] = 'satisfied_frozen'
df_final = pd.concat([fresh_sold,frozen_sold,unsatisfied_fresh,unsatisfied_frozen,production,frozen_production,frozen_inventory,demand_fresh,demand_frozen,satisfied_fresh,satisfied_frozen,bird_count],axis=0)
df_final.to_csv("Dashboard.csv")



