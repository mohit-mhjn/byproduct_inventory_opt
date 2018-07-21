"""
The results obtained from the pyomo model is in form of numeric indices

This program is used to parse the solved model instance and map the results back in the form of user defined descriptions
which results in an understandable form of output and collect + foramt the results in form of predefined tables

pandas is used for summarization of data tables

Handle with Care >>
The variable names are highly dependent on the variable names in the main program
++ Figureout a mapping to make both mainprogram and postprocessing independent of variable names >> To be done later

To Do:
1. Arrangement of Columns in print and CSV
2. get key,value response back in the main program

"""

from pyomo.environ import *   # To get the value method in the current environment

def summarize_results(model,horizon,indexes,print_tables=False,keep_files = False):

    import pandas
    import itertools

    # Bird Type Requirement
    bird_req_data = []
    for t,r in itertools.product(model.T,model.R):
        bird_req_data.append({'date':str(horizon[t]),'bird_size':indexes['bird_type'][r]['bird_type'],'req_number':model.z[t,r].value})
    bird_type_requirement = pandas.DataFrame(bird_req_data)
    bird_type_requirement = bird_type_requirement[(bird_type_requirement.req_number > 0)]


    #Cutting Pattern Plan
    cutting_pattern_data = []
    for t,(r,k,j) in itertools.product(model.T,model.indx_rkj):
        cutting_pattern_data.append({'date':str(horizon[t]),'bird_type':indexes['bird_type'][r]['bird_type'],'section':indexes['section'][k]['description'],'cutting_pattern':j,'line':indexes['cutting_pattern'][j]['line'], 'pattern_count':model.zkj[t,r,k,j].value })
    cutting_pattern_plan = pandas.DataFrame(cutting_pattern_data)
    cutting_pattern_plan = cutting_pattern_plan[(cutting_pattern_plan.pattern_count > 0)]
    cutting_pattern_plan.sort_values(by = ['date','bird_type','pattern_count','section'], inplace = True)

    #Output Production and Processing
    production_data1 = []
    production_data2 = []
    production_data3 = []

    for t,p,r in itertools.product(model.T, model.P, model.R):
        production_data1.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_produced':value(model.xpr[t,p,r]),'UOM':'KG'})
        production_data2.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_produced':model.x_freezing[t,p,r].value,'UOM':'KG'})
        production_data3.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_produced':model.x_marination[t,p,r].value,'UOM':'KG'})

    fresh_production = pandas.DataFrame(production_data1)
    fresh_production = fresh_production[(fresh_production.quantity_produced > 0)]

    freezing_lots = pandas.DataFrame(production_data2)
    freezing_lots = freezing_lots[(freezing_lots.quantity_produced > 0)]

    marination_lots = pandas.DataFrame(production_data3)
    marination_lots = marination_lots[(marination_lots.quantity_produced > 0)]

    inventory_report1 = []
    for t,(p,r,l) in itertools.product(model.T,model.INV_Fresh):
        inventory_report1.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'age_days':l,'quantity_on_hand':model.ifs[t,p,r,l].value,'UOM':'KG'})
    fresh_inventory_report = pandas.DataFrame(inventory_report1)
    fresh_inventory_report = fresh_inventory_report[(fresh_inventory_report.quantity_on_hand > 0)]

    inventory_report2 = []
    for t,p,r in itertools.product(model.T,model.P, model.R):
        inventory_report2.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_on_hand':model.ifz[t,p,r].value,'UOM':'KG'})
    frozen_inventory_report = pandas.DataFrame(inventory_report2)
    frozen_inventory_report = frozen_inventory_report[(frozen_inventory_report.quantity_on_hand > 0)]

    sales_cost_report1 = []
    sales_cost_report2 = []
    sales_cost_report3 = []

    for t,p,r in itertools.product(model.T,model.P,model.R):
        fresh_satisfied_q = model.u_fresh[t,p,r].value
        fresh_unsatisfied_q = model.v_fresh[t,p,r].value
        orders1 = sum(model.sales_order[t,c,p,r,'Fresh',0] for c in model.C_priority)
        selling_gains1 = model.selling_price[p,r,'Fresh',0]*fresh_satisfied_q
        sales_cost_report1.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'orders':value(orders1),'satisfied':fresh_satisfied_q,'unsatisfied':fresh_unsatisfied_q,'selling_gains':value(selling_gains1)})

        marinated_satisfied_q = model.um_fresh[t,p,r].value
        marinated_unsatisfied_q = model.vm_fresh[t,p,r].value
        orders2 = sum(model.sales_order[t,c,p,r,'Fresh',1] for c in model.C_priority)
        selling_gains2 = model.selling_price[p,r,'Fresh',1]*marinated_satisfied_q
        sales_cost_report2.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'orders':value(orders2),'satisfied':marinated_satisfied_q,'unsatisfied':marinated_unsatisfied_q,'selling_gains':value(selling_gains2)})

        frozen_satisfied_q = model.u_frozen[t,p,r].value
        frozen_unsatisfied_q = model.v_frozen[t,p,r].value
        orders3 = sum(model.sales_order[t,c,p,r,'Frozen',0] for c in model.C_priority)
        selling_gains3 = model.selling_price[p,r,'Frozen',0]*frozen_satisfied_q
        sales_cost_report3.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'orders':value(orders3),'satisfied':frozen_satisfied_q,'unsatisfied':frozen_unsatisfied_q,'selling_gains':value(selling_gains3)})


    sales_fresh_sku = pandas.DataFrame(sales_cost_report1)
    sales_fresh_sku = sales_fresh_sku[(sales_fresh_sku.orders > 0)]

    sales_marination_sku = pandas.DataFrame(sales_cost_report2)
    sales_marination_sku = sales_marination_sku[(sales_marination_sku.orders > 0)]

    sales_frozen_sku = pandas.DataFrame(sales_cost_report3)
    sales_frozen_sku = sales_frozen_sku[(sales_frozen_sku.orders > 0)]

    cost_report1 = []
    for t in model.T:
        cost_report1.append({'date':str(horizon[t]),'COGS':value(model.operations_cost[t]),'HoldingCost':value(model.holding_cost[t])})  # ,'Revenue':value(model.selling_gains[t])
    cost_summary = pandas.DataFrame(cost_report1)


    if print_tables:
        print ("\n\t bird requirement >> \n ")
        print (bird_type_requirement)
        print ("\n\t cutting_pattern_plan >> \n ")
        print (cutting_pattern_plan)
        print ("\n\t fresh production >> \n ")
        print (fresh_production)
        print ("\n\t freezing_lots >> \n ")
        print (freezing_lots)
        print ("\n\t marination_lots >> \n ")
        print (marination_lots)
        print ("\n\t Projected Fresh Inventory >> \n ")
        print (fresh_inventory_report)
        print ("\n\t Projected Frozen Inventory >> \n ")
        print (frozen_inventory_report)
        print ("\n\t Demand Fulfillment Plan Fresh >> \n ")
        print (sales_fresh_sku)
        print ("\n\t Demand Fulfillment Plan Frozen >> \n ")
        print (sales_frozen_sku)
        print ("\n\t Demand Fulfillment Plan Fresh with Marination >> \n ")
        print (sales_marination_sku)
        print ("\n\t Summary of Projected Costs >> \n ")
        print (cost_summary)

    if keep_files:

        import os
        directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(directory)

        bird_type_requirement.to_csv(path_or_buf = "output_files/bird_type_requirement.csv",index = False)
        cutting_pattern_plan.to_csv(path_or_buf = "output_files/cutting_pattern_plan.csv",index = False)
        fresh_production.to_csv(path_or_buf = "output_files/fresh_production.csv",index = False)
        freezing_lots.to_csv(path_or_buf = "output_files/freezing_lots.csv",index = False)
        marination_lots.to_csv(path_or_buf = "output_files/marination_lots.csv",index = False)
        fresh_inventory_report.to_csv(path_or_buf = "output_files/fresh_inventory_report.csv",index = False)
        frozen_inventory_report.to_csv(path_or_buf = "output_files/frozen_inventory_report.csv",index = False)
        sales_fresh_sku.to_csv(path_or_buf = "output_files/sales_fresh_sku.csv",index = False)
        sales_frozen_sku.to_csv(path_or_buf = "output_files/sales_frozen_sku.csv",index = False)
        sales_marination_sku.to_csv(path_or_buf = "output_files/sales_marination_sku.csv",index = False)
        cost_summary.to_csv(path_or_buf = "output_files/cost_summary.csv",index = False)

    print ("postprocessing complete!")
    return None

if __name__=="__main__":
    print ("This Function execution is only to post process results received as a response from main program  \nCannot execute the program directly\n")
    exit(0)
