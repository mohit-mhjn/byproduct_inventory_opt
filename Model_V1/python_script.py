import os
from subprocess import *

#run child script 1
p = Popen([r("cmd.exe /S /C" "pyomo solve --solver=cbc C:\Users\administrator\Desktop\SNO\6_inventory_min_concerete.py"), "ArcView"], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()
print (output[0])

#run child script 2
p = Popen([r'C:\childScript2.py', "ArcEditor"], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()
print (output[0])
