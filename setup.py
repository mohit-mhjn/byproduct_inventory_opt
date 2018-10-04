import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

import configparser
config = configparser.ConfigParser()
config.read('start_config.ini')

## Create Required Directories >>>>>>>>>>>>>>>>>>>>>>>

list_of_dir = [ directory+"/cache",
                directory+"/input_files",
                directory+"/output_files",
                directory+"/logs" ]

check_dir = [os.path.exists(d) for d in list_of_dir]

for i,chk in enumerate(check_dir):
    if not chk:
        os.mkdir(list_of_dir[i])  # Input video

print ("SUCCESS : directories created!")
#######################################################
## Create Required Objects in Cache

# > Create Empty Object : master
# > Append Index in obj
# > Append BOM in obj
# > Append conv_factor in obj
# > Append coef_param in obj
# > Append Inv Ageing in obj
# > Append Flex in Obj
os.system("python3 src/inputs.py")
os.system("python3 src/indexes.py")
os.system("python3 src/BOM_reader.py")
os.system("python3 src/coef_param.py")
os.system("python3 src/ageing_param.py")
os.system("python3 src/flex_typ.py")

## Create SKU Master
