## merging files
import numpy as np
import pandas as pd
import os

directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)


def convert(df):
    l = []
    for f in df['item_code']:
        l.append(f[:-1]+'F')
    df['item_code'] = l
def convert_fresh(df):
    l = []
    for f in df['item_code']:
        l.append(f[:-1] + 'C')
    df['item_code'] = l

#1. Fresh sold
fresh_sold = pd.read_csv('Fresh_Product_Sold.csv')
fresh_sold.columns = ['item_code','0','1','2','3','4']
fresh_sold['keyfigure'] = "fresh_sold"
# del fresh_sold['days']
convert_fresh(fresh_sold)

#2. Frozen sold
frozen_sold = pd.read_csv("Frozen_Product_Sold.csv")
frozen_sold.columns = ['item_code','0','1','2','3','4']
frozen_sold['keyfigure'] = "frozen_sold"
# frozen_sold[['Products','Customer']] = pd.DataFrame(frozen_sold.days.str.split('_',1).tolist(),
#                                    columns = ['Products','Customer'])
# del frozen_sold['days']
convert(frozen_sold)

#3. Unsatisfied Demand Fresh
unsatisfied_fresh = pd.read_csv("Unsatisfied_Fresh.csv")
unsatisfied_fresh.columns = ['item_code','0','1','2','3','4']
unsatisfied_fresh['keyfigure'] = "unsatisfied_fresh"
convert_fresh(unsatisfied_fresh)
fresh_products = unsatisfied_fresh['item_code']

#4. Unsatisfied Demand Frozen
unsatisfied_frozen = pd.read_csv("Unsatisfied_Frozen.csv")
unsatisfied_frozen.columns = ['item_code','0','1','2','3','4']
unsatisfied_frozen['keyfigure'] = "unsatisfied_frozen"
# unsatisfied_frozen[['Products','Customer']] = pd.DataFrame(unsatisfied_frozen.days.str.split('_',1).tolist(),
#                                    columns = ['Products','Customer'])
# del unsatisfied_frozen['days'
convert(unsatisfied_frozen)
frozen_products = unsatisfied_frozen['item_code']

#5. Production
production = pd.read_csv("Production.csv")
production.columns = ['item_code','0','1','2','3','4']
production['keyfigure'] = "production"
# production[['Products','Customer']] = pd.DataFrame(production.days.str.split('_',1).tolist(),
#                                    columns = ['Products','Customer'])
# del production['days']
convert_fresh(production)
#6. Frozen Production
frozen_production = pd.read_csv("Frozen_Product_Production.csv")
frozen_production.columns = ['item_code','0','1','2','3','4']
frozen_production['keyfigure'] = "frozen_production"
# frozen_production[['Products','Customer']] = pd.DataFrame(frozen_production.days.str.split('_',1).tolist(),
#                                    columns = ['Products','Customer'])
# del frozen_production['days']
convert(frozen_production)
#7. Frozen Inventory
frozen_inventory = pd.read_csv("Frozen_Product_Inventory.csv")
print(frozen_inventory)
frozen_inventory.columns = ['item_code','0','1','2','3','4']
frozen_inventory['keyfigure'] = "frozen_inventory"
# frozen_inventory[['Products','Customer']] = pd.DataFrame(frozen_inventory.days.str.split('_',1).tolist(),
#                                    columns = ['Products','Customer'])
# del frozen_inventory['days']
convert(frozen_inventory)
#8. Demand Fresh
demand_fresh = pd.read_excel('Model_Input_Final.xlsm', sheetname="Demand_Fresh")
demand_fresh = demand_fresh.dropna(axis='columns', how='all')
demand_fresh['0'] = 0
demand_fresh = demand_fresh.filter(items = ['item code','0',1,2,3,4])
demand_fresh.columns = ['item_code','0','1','2','3','4']
convert_fresh(demand_fresh)
demand_fresh['keyfigure'] = "demand_fresh"
demand_fresh = demand_fresh[demand_fresh['item_code'].isin(fresh_products)]

# demand_fresh[['Products','Customer']] = pd.DataFrame(demand_fresh['days'].str.split('_',1).tolist(),
#                                    columns = ['Products','Customer'])
# del demand_fresh['days']
# print(demand_fresh.head())
# demand_fresh.columns =  [0,1,2,3,4,'keyfigure','Products',]

#9. Demand Frozen
demand_frozen = pd.read_excel('Model_Input_Final.xlsm', sheetname="Demand_Frozen")
demand_frozen = demand_frozen.dropna(axis='columns', how='all')
demand_frozen['0'] = 0
demand_frozen = demand_frozen.filter(items = ['item code','0',1,2,3,4])
demand_frozen.columns = ['item_code','0','1','2','3','4']
#print (demand_frozen)
demand_frozen['keyfigure'] = "demand_frozen"
convert(demand_frozen)
demand_frozen = demand_frozen[demand_frozen['item_code'].isin(frozen_products)]
# print(frozen_inventory)

# demand_frozen[['Products','Customer']] = pd.DataFrame(demand_frozen.days.str.split('_',1).tolist(),
#                                    columns = ['Products','Customer'])
# del demand_frozen['days']
# demand_frozen.columns = [0,1,2,3,4,'keyfigure','Products',]


#10. Bird Count
bird_count = pd.read_csv("Bird_Count.csv")
bird_count.columns = ['Days','Number']
bird_count = bird_count.T
bird_count = bird_count.drop("Days")
bird_count['0'] = 0
bird_count['item_code'] = 'BIRD COUNT'
#bird_count.set_index('item_code', drop = True, inplace = True)
bird_count = bird_count.filter(items = ['item_code','0',0,1,2,3])
# bird_count.columns =  [0,1,2,3,4,'keyfigure','Products',]
# bird_count['3'] = 0
# bird_count['2'] = 0
# bird_count['4'] = 0

bird_count.columns = ['item_code','0','1','2','3','4']
bird_count['keyfigure'] = "bird_count"

#11. Satsfied Demand Fresh
# temp1 = demand_fresh.drop(['Products','keyfigure'], axis=1)
# temp2 = unsatisfied_fresh.drop(['Products','keyfigure'], axis=1)
# print(temp1)
# print(temp2)
# satisfied_fresh = temp1 - temp2
# satisfied_fresh['Customer'] = unsatisfied_fresh['Customer']
# satisfied_fresh['Products'] = unsatisfied_fresh['Products']
# satisfied_fresh['keyfigure'] = 'satisfied_fresh'

#12. Satsfied Demand Frozen

# temp1 = demand_frozen.drop(['Products','keyfigure'], axis=1)
# temp2 = unsatisfied_frozen.drop(['Products','keyfigure'], axis=1)
# satisfied_frozen = temp1 - temp2
# satisfied_frozen['Customer'] = unsatisfied_frozen['Customer']
# satisfied_frozen['Products'] = unsatisfied_frozen['Products']
# satisfied_frozen['keyfigure'] = 'satisfied_frozen'


#13. Satsfied Demand Frozen

temp1 = production.drop(['item_code','keyfigure'], axis=1)
temp2 = frozen_production.drop(['item_code','keyfigure'], axis=1)
fresh_production = temp1 - temp2
fresh_production['item_code'] = production['item_code']
fresh_production['keyfigure'] = 'fresh_production'
cols = fresh_production.columns.tolist()
cols = [cols[5]] + cols[:5] + cols[-1:]
fresh_production = fresh_production[cols]
# Fresh Inventory
temp1 = production.drop(['item_code','keyfigure'],axis=1)
temp2 = frozen_production.drop(['item_code','keyfigure'],axis=1)
temp3 = fresh_sold.drop(['item_code','keyfigure'],axis=1)
fresh_inventory = temp1 - temp2 - temp3
for i in fresh_inventory.columns[:-1]:
    fresh_inventory[str(int(i)+1)] = fresh_inventory[i]+fresh_inventory[str(int(i)+1)]
fresh_inventory[fresh_inventory < 0] = 0
fresh_inventory['item_code'] = production['item_code']
fresh_inventory['keyfigure'] = 'fresh_inventory'
# print(frozen_production.columns)
# print(fresh_inventory.columns)

cols = fresh_inventory.columns.tolist()
cols = [cols[5]] + cols[:5] + cols[-1:]
fresh_inventory = fresh_inventory[cols]
ls = [fresh_sold,frozen_sold,unsatisfied_fresh,unsatisfied_frozen,fresh_production,frozen_production,frozen_inventory,demand_fresh,demand_frozen,bird_count]
# print (ls)
# df_final = pd.concat([fresh_sold,frozen_sold,fresh_production,unsatisfied_fresh,unsatisfied_frozen,frozen_production,fresh_inventory,frozen_inventory,demand_fresh,demand_frozen,bird_count],axis=0)
df_final = pd.concat(ls)
df_final.columns = ['item_code', 'Day0', 'Day1', 'Day2', 'Day3', 'Day4', 'keyfigure']
#df_final.reset_index(inplace=True,drop=True)
#df_final = pd.wide_to_long(df_final, stubnames='Day', i=['keyfigure','item_code'], j='Days')
arr = []
for indx, row in df_final.iterrows():
    arr.append({'item_code':row['item_code'],'Day':0,'Value':row['Day0'],'keyfigure':row['keyfigure']})
    arr.append({'item_code': row['item_code'], 'Day': 1, 'Value': row['Day1'], 'keyfigure': row['keyfigure']})
    arr.append({'item_code': row['item_code'], 'Day': 2, 'Value': row['Day2'], 'keyfigure': row['keyfigure']})
    arr.append({'item_code': row['item_code'], 'Day': 3, 'Value': row['Day3'], 'keyfigure': row['keyfigure']})
    arr.append({'item_code': row['item_code'], 'Day': 4, 'Value': row['Day4'], 'keyfigure': row['keyfigure']})

df_final = pd.DataFrame(arr)
df_final = df_final.reset_index(drop = True)
# df_final.columns =  ['Products','keyfigure','Days','Value']
# df_final['Products']=df_final['Products'].astype(str)

#
# df_final = df_final[cols]
# df_final.columns =  ['Products','keyfigure','Days','Value']
df_final['ProdType'] = np.where(df_final['item_code'].str[-1]=='F', 'Frozen', 'Fresh')
cols = df_final.columns.tolist()
cols = [cols[2]]+ [cols[-1]] +[cols[-2]] + cols[0:2]
df_final = df_final[cols]
selling_fresh = pd.read_excel('Model_Input_Final.xlsm', sheetname="Selling_Price_Fresh")
selling_fresh = selling_fresh.dropna(axis='columns', how='all')
selling_fresh = selling_fresh.rename(index=str, columns={"item code": "item_code"})
df_final= df_final.merge(selling_fresh, on='item_code', how='left')
del df_final['Selling Price Fresh (in RM/kg)']
selling_frozen = pd.read_excel('Model_Input_Final.xlsm', sheetname="Selling_Price_Frozen")
selling_frozen = selling_frozen.dropna(axis='columns', how='all')
selling_frozen = selling_frozen.rename(index=str, columns={"item code": "item_code"})
df_final = df_final.merge(selling_frozen, on='item_code', how='left')
df_final['Priority'] = np.max(df_final[['Priority_x', 'Priority_y']], axis=1)
df_final = df_final.drop(labels=['Priority_x', 'Priority_y'], axis=1)
del df_final['Selling Price Frozen (in RM/kg)']

df_final['Priority'] = df_final['Priority'].fillna(2)
df_final.columns =  ['Products','ProdType','keyfigure','Days','Value', 'Priority']

df_final.to_csv("Dashboard.csv",index=False)
