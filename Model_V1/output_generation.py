import pandas as pd
import numpy as np
import os
os.chdir('D:\Model_V1')
import csv
import json
import re
with open('results.json') as data_file:
    data = json.load(data_file)
    #print(data)
    solution = data['Solution'][1]['Variable']
    print(solution.items())
#     i_sol = [list(map(int,re.findall(r'\d+',k))) for k,v in solution.items() if 'i' in k]
    i_sol = {k[1:]: v['Value'] for k,v in solution.items() if 'i' in k}
    x_sol = {k[1:]: v['Value'] for k,v in solution.items() if 'x' in k}

d = pd.DataFrame(0, index=range(1,5), columns=range(1,8))
for x,v in x_sol.items():
    #creates list of strings that match the regular expression
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    #joins all of the elements of the word array into a string separated by a space
    outputString = " ".join((words))
    d[int(words[0][2])][int(words[0][0])]=v
    #output is "Hello World How"
d.index=['6P','8P','7P','9P']
d.columns = ['Day_1','Day_2','Day_3','Day_4','Day_5','Day_6','Day_7']
print(d)

d.to_csv('Bird_count.csv')
import pandas as pd
d = pd.DataFrame(0, index=range(1,8), columns=range(1,8))
for x,v in i_sol.items():
    #creates list of strings that match the regular expression
    words = re.findall(r'[\d]+,[\d\.-]+', x)
    #joins all of the elements of the word array into a string separated by a space
    outputString = " ".join((words))
    d[int(words[0][2])][int(words[0][0])]=v
    #output is "Hello World How"
d.index=['Ribs','Keels','Legs','Thighs','Drumbstick','Wings','Breasts']
d.columns = ['Day_1','Day_2','Day_3','Day_4','Day_5','Day_6','Day_7']
d.to_csv('Inventory.csv')
print(d)
