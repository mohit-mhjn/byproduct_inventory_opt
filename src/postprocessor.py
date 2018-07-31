"""
The results obtained from the pyomo model is in form of numeric indices

This program is used to parse the solved model instance and map the results back in the form of user defined descriptions
which results in an understandable form of output and collect + foramt the results in form of predefined tables

pandas is used for summarization of data tables

The function summarize_tables will do the required summarization of the output. The labelled parameters passed to this function are print_tables and keep_files
keep_files : It will write the output csv files in output_files directory (default : False)
print_tables : It will print the output tables generated by the function (default : False)

NOTE : Handle with Care >>
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
    bird_type_requirement.name = 'bird requirement'


    #Cutting Pattern Plan
    cutting_pattern_data = []
    for t,(r,k,j) in itertools.product(model.T,model.indx_rkj):
        cutting_pattern_data.append({'date':str(horizon[t]),'bird_type':indexes['bird_type'][r]['bird_type'],'section':indexes['section'][k]['description'],'cutting_pattern':j,'line':indexes['cutting_pattern'][j]['line'], 'pattern_count':model.zkj[t,r,k,j].value })
    cutting_pattern_plan = pandas.DataFrame(cutting_pattern_data)
    cutting_pattern_plan = cutting_pattern_plan[(cutting_pattern_plan.pattern_count > 0)]
    cutting_pattern_plan.sort_values(by = ['date','bird_type','pattern_count','section'], inplace = True)
    cutting_pattern_plan.name = 'cutting_pattern_plan'

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
    fresh_production.name = 'fresh production'

    freezing_lots = pandas.DataFrame(production_data2)
    freezing_lots = freezing_lots[(freezing_lots.quantity_produced > 0)]
    freezing_lots.name = 'freezing_lots'

    marination_lots = pandas.DataFrame(production_data3)
    marination_lots = marination_lots[(marination_lots.quantity_produced > 0)]
    marination_lots.name = 'marination_lots'

    inventory_report1 = []
    for t,(p,r,l) in itertools.product(model.T,model.INV_Fresh):
        inventory_report1.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'age_days':l,'quantity_on_hand':model.ifs[t,p,r,l].value,'UOM':'KG'})
    fresh_inventory_report = pandas.DataFrame(inventory_report1)
    fresh_inventory_report = fresh_inventory_report[(fresh_inventory_report.quantity_on_hand > 0)]
    fresh_inventory_report.name = 'Projected Fresh Inventory'

    inventory_report2 = []
    for t,p,r in itertools.product(model.T,model.P, model.R):
        inventory_report2.append({'date':str(horizon[t]),'product_group':indexes['product_group'][p]['product_group'],'bird_size':indexes['bird_type'][r]['bird_type'],'quantity_on_hand':model.ifz[t,p,r].value,'UOM':'KG'})
    frozen_inventory_report = pandas.DataFrame(inventory_report2)
    frozen_inventory_report = frozen_inventory_report[(frozen_inventory_report.quantity_on_hand > 0)]
    frozen_inventory_report.name = 'Projected Frozen Inventory'

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
    sales_fresh_sku.name = 'Demand Fulfillment Plan Fresh'

    sales_marination_sku = pandas.DataFrame(sales_cost_report2)
    sales_marination_sku = sales_marination_sku[(sales_marination_sku.orders > 0)]
    sales_marination_sku.name = 'Demand Fulfillment Plan Fresh with Marination'

    sales_frozen_sku = pandas.DataFrame(sales_cost_report3)
    sales_frozen_sku = sales_frozen_sku[(sales_frozen_sku.orders > 0)]
    sales_frozen_sku.name = 'Demand Fulfillment Plan Frozen'

    cost_report1 = []
    for t in model.T:
        revenue_generated = sum(value(model.order_qty_supplied[o])*model.order_sp[o] for o in model.O if t == model.order_date[o])
        cost_report1.append({'date':str(horizon[t]),'COGS':value(model.operations_cost[t]),'HoldingCost':value(model.holding_cost[t]),'Revenue':revenue_generated})
    cost_summary = pandas.DataFrame(cost_report1)
    cost_summary.name = 'Summary of Projected Costs'

    order_report = []
    for o in model.O:
        order_report.append({'order_number':o,'order_date':horizon[model.order_date[o]],'total_order_quantity':model.order_qty[o],'quantity_fulfilled':value(model.order_qty_supplied[o])})
    order_fulfillment_summary = pandas.DataFrame(order_report)
    order_fulfillment_summary["service_level"] = 100*order_fulfillment_summary["quantity_fulfilled"]/order_fulfillment_summary["total_order_quantity"]
    order_fulfillment_summary.name = 'Order wise fulfillment data'


    output_tables = [bird_type_requirement,
                    cutting_pattern_plan,
                    fresh_production,
                    freezing_lots,
                    marination_lots,
                    fresh_inventory_report,
                    frozen_inventory_report,
                    sales_fresh_sku,
                    sales_frozen_sku,
                    sales_marination_sku,
                    cost_summary,
                    order_fulfillment_summary]

    if print_tables:
        pandas.set_option('display.expand_frame_repr', False)
        for table in output_tables:
            print ("\n\t %s >> \n "%(table.name))
            print (table)


    if keep_files:

        import os
        directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(directory)

        for df in output_tables:
            df.to_csv(path_or_buf = "output_files/%s.csv"%(df.name),index = False)

    print ("postprocessing complete!")
    return None

if __name__=="__main__":
    print ("This Function execution is only to post process results received as a response from main program  \nCannot execute the program directly\n")
    exit(0)
