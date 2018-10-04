"""
Note: Compatible with Python 3

This file handles the main pyomo MILP that comprises of required Sets, Constraints, Expression, Param and Model objects.
Concrete model is used for pyomo instance creation

Loading Data in concrete model:
To load the static data cached data is imported from the respective reader modules
To load the dynamic data input files are read and transformed using the function of the respective readers (sales,inventory)

General Ref:
combination_gen : function used to load pyomo indexed sets in the model. The combinations are preprocessed and cached
expression_gen : fucntion used to generate pyomo expressions that are mentioned against
A [i] Constraint : Name used to indicate constraint with index i
In Comments, Product is reffered as product group

Assumptions:
1. If a cutting pattern is applied on whole bird it applies on all the sections of the birds in equal number of times
2. Inventory transfer if FIFO
3. Frozen product doesn't age (long shelf life in comparison with planning horizon)

To Do:
1. Remove print and add logger
3. Warehouse Capacity
4  MOQ at Lines/CP
"""
from pyomo.environ import *
import datetime

def create_instance(master,var_data,scenario = 1):

    model = ConcreteModel()

    ## Index Definition #####################################################
    model.T = Set(initialize= list(range(len(var_data.horizon))), ordered = True) # planning horizon* Temporary >> To include initialize in indx
    model.J = Set(initialize= master.cutting_pattern.keys())   # cutting patterns
    model.P = Set(initialize= master.product_group.keys())     # products
    model.K = Set(initialize= master.section.keys())          # no of sections
    model.R = Set(initialize= master.bird_type.keys())         # type of carcasses
    model.P_type = Set(initialize = master.product_typ.keys())          # Type of Products
    model.M = Set(initialize = master.marination.keys())                # Marination Indicator
    model.C_priority = Set(initialize = master.c_priority)       # Customer Priority Indicator
    model.O = Set(initialize = var_data.order_breakup.keys())                 # Order Id's for strict bird type products
    model.RNG = Set(initialize = master.weight_range.keys())      # Distinct range sets of flexible size range
    model.flex_O = Set(initialize = var_data.flex_order_breakup.keys())      # Order Id's for flexible bird type products

    ## Generating combinations ###############################################
    def combination_gen1(model,j):
        # global master
        return master.cutting_pattern[j]['section_id']
    model.Jk = Set(model.J, initialize = combination_gen1)       # Cutting Pattern >> set(Sections)

    def combination_gen2(model,k):
        # global master
        return master.section[k]['cp_id']
    model.Kj = Set(model.K, initialize = combination_gen2)       # Section vs set(Cutting Patterns)  (Inverse of previous)

    def combination_gen3(model):
        # global master
        return master.sec_cp
    model.indx_kj = Set(dimen = 2, initialize = combination_gen3)    # Combinations of (section,cutting_pattern)

    def combination_gen4(model,k,j):
        # global master
        return master.sec_nwb_pg[(k,j)]
    model.KJp1 = Set(model.indx_kj, initialize = combination_gen4)   # Non Whole bird products coming from a (section,cutting_pattern)

    def combination_gen5(model,k,j):
        # global master
        return master.sec_wb_pg[(k,j)]
    model.KJp2 = Set(model.indx_kj, initialize = combination_gen5)   # Whole bird products coming from a (section,cutting_pattern)

    def combination_gen6(model,k):
        # global master
        return master.section[k]['pgroup_id']
    model.Kp = Set(model.K, initialize = combination_gen6)     # Products from coming section

    def combination_gen7(model):
        # global master
        return master.pgcptyp
    model.indx_pjr = Set(dimen = 3, initialize = combination_gen7) # Combinations of (product,cutting_pattern,bird_type)

    def combination_gen8(model):
        # global master
        return master.typ_cp
    model.indx_rj = Set(dimen = 2, initialize = combination_gen8)    # Combinations of (bird_type, cutting_pattern)

    def combination_gen9(model):
        # global master
        return master.typseccp
    model.indx_rkj = Set(dimen = 3, initialize = combination_gen9)   # Combinations of (bird_type,section,cutting_pattern)

    def combination_gen10(model,p):
        # global master
        return master.pg_cptyp[p]
    model.Pjr = Set(model.P,dimen=2,initialize = combination_gen10)    # Combinations of (cutting_pattern, bird_type) that yield product P

    def combination_gen11(model,r,j):
        # global master
        return master.cptyp_pg[(j,r)]
    model.RJp = Set(model.indx_rj,initialize = combination_gen11)  # Products yielded by a particular (bird_type,cutting_pattern) (Inverse of previous)

    def combination_gen12(model):
        # global master
        return master.life_dct['age_combinations_fresh']
    model.INV_Fresh = Set(dimen = 3, initialize=combination_gen12)   # For fresh products inventory index with age (product,bird_type,age) where age lies in range [1,shelf_life]

    def combination_gen13(model):   # This can be cached >>> To Do
        # global master
        return master.typseccpp
    model.indx_rkjp = Set(dimen = 4, initialize = combination_gen13)     # Combination of (bird type,section,cutting_pattern,product)

    order_iter_set = set()
    def combination_gen15(model,t,c,p,r,typ,m):       # Time Consuming >> Temporarily Fine
        # global var_data
        dt = var_data.t_dt_map[t]
        if (dt,c,p,r,typ,m) in var_data.order_grouped.keys():
            order_set = var_data.order_grouped[(dt,c,p,r,typ,m)]
            for o in order_set:
                order_iter_set.add((t,c,p,r,typ,m,o))
            return order_set
        else:
            return []
    model.order_group = Set(model.T,model.C_priority,model.P, model.R, model.P_type, model.M, initialize = combination_gen15)

    def combination_gen16(model):                                        #  Combination of date,c_priority, product, bird_type, fresh/frozen, marination, order_no
        return list(order_iter_set)
    model.indx_o_filling = Set(dimen = 7, initialize = combination_gen16)

    def combination_gen17(model,rng):                                        # Bird Types for several flexible weight ranges
        # global master
        return master.flex_range1[rng]
    model.wt_set1 = Set(model.RNG, initialize = combination_gen17)

    def combination_gen18(model,r):                                        #  Flexible weight ranges for weight R
        # global master
        return master.flex_range2[r]
    model.wt_set2 = Set(model.R, initialize = combination_gen18)

    def combination_gen19(model):                                   #  Flexible weight range v/s weight combinations
        # global master
        return master.flex_set
    model.wt_set3 = Set(dimen = 2, initialize = combination_gen19)

    #####  +++++++++++++ FLEX Orders Grouped ++++++

    flex_order_iter_set = set()
    def combination_gen20(model,t,c,p,r,typ,m):       # Time Consuming >> Temporarily Fine
        # global var_data
        dt = var_data.t_dt_map[t]
        if (dt,c,p,r,typ,m) in var_data.flex_order_grouped.keys():
            order_set = var_data.flex_order_grouped[(dt,c,p,r,typ,m)]
            for o in order_set:
                flex_order_iter_set.add((t,c,p,r,typ,m,o))
            return order_set
        else:
            return []
    model.flex_order_group = Set(model.T,model.C_priority,model.P, model.RNG, model.P_type, model.M, initialize = combination_gen20)

    ## Loading Input Parameters #############################################
    model.BigM = Param(initialize = 9999999)
    # Highly Controversial Value >> Careful with this while tuning the real dataset
    # Solvers might sometime fail because of big M values, If in case reduce the value by a factor of 10 and try. It shall work otherwise repeat this step again

    def inv_gen1(model,t,r):
        # global var_data
        dt = var_data.t_dt_map[t]
        return var_data.bird_availability[(dt,r)]
    model.H = Param(model.T, model.R, initialize = inv_gen1)         # availability of birds of type r at time t

    def inv_gen2(model,p,r,l):  # Opening Inventory
        # global var_data
        if (p,r,l) in var_data.part_inv_fresh.keys():
            return var_data.part_inv_fresh[(p,r,l)]
        else:
            return 0
    model.initial_inv_fresh = Param(model.INV_Fresh, initialize = inv_gen2)     # Initial inventory of fresh products p of bird type r  with age l (in days)

    def inv_gen3(model,p,r):
        # global var_data
        if (p,r) in var_data.part_inv_frozen.keys():
            return var_data.part_inv_frozen[(p,r)]
        else:
            return 0
    model.initial_inv_frozen = Param(model.P,model.R, initialize = inv_gen3)     # Initial inventory of frozen product p of bird type r

    def order_gen1(model,t,cp,p,r,pt,m):
        # global var_data
        dt = var_data.t_dt_map[t]
        if (dt,cp,p,r,pt,m) in var_data.orders_aggregate.keys():
            return var_data.orders_aggregate[(dt,cp,p,r,pt,m)]
        else:
            return 0
    model.sales_order = Param(model.T,model.C_priority,model.P,model.R,model.P_type,model.M, initialize = order_gen1)   # Sales order (day,customer_priority, product,bird_type, fresh/frozen, marination (1/0))

    def order_gen2(model,t):
        return 10
    model.further_proessing = Param(model.T, initialize = order_gen2)   # Demand of fresh products for further processing

    def yield_gen(model,p,j,r):
        # global master
        return master.yield_data[(p,j,r)]['yld']
    model.yd = Param(model.indx_pjr, initialize = yield_gen)    # Yield of product p from bird type r at cutting pattern j

    def shelflife_gen(model,p,r): # Only Fresh
        # global master
        return master.life_dct['shelf_life_fresh'][(p,r)]
    model.L = Param(model.P,model.R,initialize = shelflife_gen)       # Shelf Life of fresh product p of bird type r

    def capacity_gen1(model,process):
        # global master
        return master.capacity_dct[process]
    model.process_capacity = Param(['freezing','marination'],initialize= capacity_gen1)   # Hard Coded process steps | Marination and freezing Capacity

    def capacity_gen2(model,j):
        # global master
        return master.capacity_dct['cutting_pattern'][j]
    model.cutting_capacity = Param(model.J, initialize = capacity_gen2)       # Cutting Capacity at cutting pattern J (birds/sections per hour)

    # def cost_gen1(model,process):
    #     global cost
    #     return cost_data[process]
    model.unit_freezing_cost = Param(initialize= master.cost_dct['freezing_cost'])   # Operations Cost Coef. Freezing Cost

    # def cost_gen2(model,process):
    #     global cost
    #     return cost_data[process]
    model.unit_marination_cost = Param(initialize= master.cost_dct['marination_cost'])   # Operations Cost Coef. Marination Cost

    def cost_gen3(model,j):
        # global master
        return master.cost_dct['cutting_cost'][j]
    model.unit_cutting_cost = Param(model.J,initialize = cost_gen3)    # Operations Cost Cost of Cutting >> Related with Cutting Pattern

    def cost_gen4(model,p,r,t):
        # global master
        return master.cost_dct['holding_cost'][(p,r,t)]
    model.inventory_holding_cost = Param(model.P,model.R,model.P_type, initialize = cost_gen4) # Cost coef of Inventroy Holding >> Fresh and Frozen product 'P' of bird type 'R'

    def cost_gen5(model,p,r,t,m):
        # global master
        if t == 2 and m == 1: # >> Removed from Consideration frozen marinated sku
            return 0
        else:
            return master.cost_dct['selling_price'][(p,r,t,m)]
    model.selling_price = Param(model.P,model.R,model.P_type,model.M, initialize = cost_gen5)  # Selling Price (profit Cost Coef) selling price of product p bird type R against fresh/frozen and marination 1/0

    def sla_fulfillment(model,o):
        # global var_data
        return var_data.order_breakup[o]["serv_agrmnt"]*var_data.order_breakup[o]["order_qty"]
    model.order_sla = Param(model.O, initialize = sla_fulfillment)                        # Service Level Agreement against each order line item

    def order_selling_price_gen(model,o):
        # global var_data
        return var_data.order_breakup[o]["selling_price"]
    model.order_sp = Param(model.O, initialize = order_selling_price_gen)                # Selling price of order

    def order_qty_gen(model,o):
        # global var_data
        return var_data.order_breakup[o]["order_qty"]
    model.order_qty = Param(model.O, initialize = order_qty_gen)                         # Quantity contained in a order

    def order_date_gen(model,o):
        # global var_data
        dt = var_data.horizon.index(datetime.datetime.strptime(var_data.order_breakup[o]["order_dt"],"%Y-%m-%d").date())
        return dt
    model.order_date = Param(model.O, initialize = order_date_gen)                        # date of an order

    def order_priority_gen(model,o):
        # global var_data
        return var_data.order_breakup[o]["priority"]
    model.order_priority = Param(model.O, initialize = order_priority_gen)             # C_Priority of an oder

    ## Following is for flexible >> Under Testing

    def order_gen3(model,t,c,p,rng,typ,m):                                                     # Sales order for Flex type SKU
        # global var_data
        dt = var_data.t_dt_map[t]
        if (dt,c,p,rng,typ,m) in var_data.flex_orders_aggregate.keys():
            return var_data.flex_orders_aggregate[(dt,c,p,rng,typ,m)]
        else:
            return 0
    model.flex_sales_order = Param(model.T,model.C_priority,model.P,model.RNG,model.P_type,[0],initialize = order_gen3)   # Only m = 0 >> No Marination for flex (not in scope)

    def flex_sla_fulfillment(model,o):
        # global var_data
        return var_data.flex_order_breakup[o]["serv_agrmnt"]*var_data.flex_order_breakup[o]["order_qty"]
    model.flex_order_sla = Param(model.flex_O, initialize = flex_sla_fulfillment)                        # Service Level Agreement against each order line item

    def flex_order_qty_gen(model,o):
        # global var_data
        return var_data.flex_order_breakup[o]["order_qty"]
    model.flex_order_qty = Param(model.flex_O, initialize = flex_order_qty_gen)                         # Quantity contained in a order

    ## Variable Objects #######################################

    model.z = Var(model.T, model.R, domain= NonNegativeIntegers)             # no of carcass r processed in period T
    model.zk = Var(model.T, model.R, model.K, domain = NonNegativeIntegers)  # Number of k section produced by carcass type R
    model.zkj = Var(model.T, model.indx_rkj, domain = NonNegativeIntegers)   # Number of times cutting pattern j is applied on section k of carcass type R
    model.zj = Var(model.T, model.indx_rj, domain = NonNegativeIntegers)     # Number of times cutting pattern j is applied on bird of size R

    model.xpr = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Quantity of product P of bird type R produced in time slot 't'
    model.ifs = Var(model.T, model.INV_Fresh, domain = NonNegativeReals)     # Auxillary variable to check Inv Fresh
    model.ifz = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Auxillary variable to check Inv Frozen (Aeging Diff not considered)
    model.xpjr = Var(model.T, model.indx_pjr, domain = NonNegativeReals)     # Quantity of product P of bird type R produced in time slot 't' with cutting pattern J

    model.il = Var(model.T, model.INV_Fresh, domain = NonNegativeReals)      # Fresh Inventory of Age L used

    model.coef_serv_L = Var(model.T,model.C_priority,model.P,model.R,model.P_type,model.M, bounds = (0,1), domain = NonNegativeReals)  # Service Level Fulfillment percent
    model.coef_serv_L_indicator = Var(model.T,model.C_priority,model.P,model.R,model.P_type,model.M, domain = Binary)  # Service Level Fulfillment percent

    model.order_qty_supplied = Var(model.O, domain = NonNegativeReals)              # For an order O, how much quantity is fulfilled
    model.flex_order_qty_supplied = Var(model.flex_O, domain = NonNegativeReals)    # Quantity supplied in flexible size type orders

    model.x_freezing = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Quantity of Product P of bird type R converted from Fresh to Frozen
    model.x_marination = Var(model.T,model.P,model.R, domain = NonNegativeReals)    # Quantity of Product P of bird type R converted from Fresh to Fresh Marinated

    model.u_fresh = Var(model.T, model.P, model.R, domain = NonNegativeReals)       # Demand Satisfied Fresh
    model.v_fresh = Var(model.T, model.P, model.R, domain = NonNegativeReals)       # Demand Unsatisfied Fresh
    model.um_fresh = Var(model.T,model.P, model.R, domain = NonNegativeReals)       # Demand Satisfied marinated fresh
    model.vm_fresh = Var(model.T, model.P, model.R, domain = NonNegativeReals)      # Demand Unsatisfied marinated fresh
    model.u_frozen = Var(model.T, model.P, model.R, domain = NonNegativeReals)      # Demand Satisfied Frozen
    model.v_frozen = Var(model.T, model.P, model.R, domain = NonNegativeReals)      # Demand Unsatisfied Frozen

    model.a_flex = Var(model.T, model.P, model.wt_set3, model.P_type, domain = NonNegativeReals)  # Breakup of demand satisfied for each product from each size against each rng element
    model.u_flex = Var(model.T, model.P, model.RNG, model.P_type, domain = NonNegativeReals)      # Demand Satisfied, flexible sku types
    model.v_flex = Var(model.T, model.P, model.RNG, model.P_type, domain = NonNegativeReals)      # Demand Unsatisfied flexible, flexible sku types

    #model.test = Var(model.T, model.P, model.R, domain = NonNegativeReals)    # Test
    ## Constraints ##############################################

    def carcass_to_section(model,t,r,k):
        return model.zk[t,r,k] == model.z[t,r]
    model.A1Constraint = Constraint(model.T, model.R, model.K, rule = carcass_to_section)        # All sections of birds are cut in equal number (no inventroy at section level)

    def carcass_in_cutpattern(model,t,r,k):
        lst = [j0 for r0,k0,j0 in model.indx_rkj if r0 == r and k0 == k and j0 in model.Kj[k]]
        return sum(model.zkj[t,r,k,j] for j in lst) == model.zk[t,r,k]
    model.A2Constraint = Constraint(model.T, model.R, model.K, rule = carcass_in_cutpattern)     # Total number of cuts on section k of bird type r is the sum of total applicable cutting pattenrs on (r,k) combinations

    def cutting_pattern_count_gen(model,t,r,k,j):
        return model.zj[t,r,j] >= model.zkj[t,r,k,j]
    model.A3Constraint = Constraint(model.T, model.indx_rkj, rule = cutting_pattern_count_gen)   # Determining number of times cutting pattern J is applied (min value)

    def cutting_pattern_count_limiter(model,t,r,j):  # Will become redundant if z is in min(obj)
        return model.zj[t,r,j] <= sum(model.zkj[t,r,k,j] for k in model.Jk[j])
    model.A4Constraint = Constraint(model.T, model.indx_rj, rule = cutting_pattern_count_limiter) # Determining number of times cutting pattern J is applied (max value)

    def cutting_pattern_balancer1(model,t,r,k,j):
        return model.zkj[t,r,k,j] >= model.zj[t,r,j]
    model.A5_1Constraint = Constraint(model.T, model.indx_rkj, rule = cutting_pattern_balancer1)  # Cutting pattern of whole bird is equally applied on all the sections

    def cutting_pattern_balancer2(model,t,r,k,j):
        return model.zkj[t,r,k,j] <= model.zj[t,r,j]
    model.A5_2Constraint = Constraint(model.T, model.indx_rkj, rule = cutting_pattern_balancer2)   # Cutting pattern of whole bird is equally applied on all the sections

    def product_yield_eqn(model,t,r,k,j,p):
        all_items = set(model.RJp[r,j])       # >> All items possible from r,j
        products1 = set([p])                  # >> All sectional(non-whole bird) bird entities from cutting pattenr j
        products2 = set(model.KJp2[k,j])      # >> All whole bird entities from cutting pattern j
        products = products1.union(products2) # >> Combine two sets of possible products
        products = all_items.intersection(products)  # >> This step will any remove any p = -1 coming from indx_rjkp
        if not products:
            return Constraint.Skip
        else:
            return model.zkj[t,r,k,j] == sum(model.xpjr[t,p,j,r]/model.yd[p,j,r] for p in products)   # >> Prone to Numerical Error float=int ; Revision Required
    model.A7Constraint = Constraint(model.T, model.indx_rkjp, rule = product_yield_eqn)        # Conversion of a cut-up into a product group

    def fresh_part_production(model,t,p,r):
        return model.xpr[t,p,r] == sum(model.xpjr[t,p,j1,r] for p1,j1,r1 in model.indx_pjr if p1 == p and r1 == r)
    model.A8Constraint = Constraint(model.T, model.P, model.R, rule = fresh_part_production)           # Summation of product belonging to a bird type obtained from agreegation of all freshly produced lots

    ## Inventroy Balance Equations and Constraints ##########################################

    def expression_gen1(model,t,p,r):
        return  model.u_fresh[t,p,r] + model.x_freezing[t,p,r] + model.x_marination[t,p,r] + sum(model.a_flex[t,p,rng,r,1] for rng in model.wt_set2[r])
    model.fresh_inv_used = Expression(model.T, model.P, model.R, rule = expression_gen1)      # Quantity of Fresh Inventory used is equal to inv q used in freezing + q used in marination + q sold in fresh form

    def inventory_used_for_age(model,t,p,r):                                                  # Quantity of Fresh Inv used = Qunaity of Fresh Inv used for all ages
        return sum(model.il[t,p,r,l] for l in range(int(model.L[p,r]) + 1)) == model.fresh_inv_used[t,p,r]
    model.A9Constraint = Constraint(model.T, model.P, model.R, rule = inventory_used_for_age)

    def expression_gen2(model,t,p,r,l):
        if t == 0 or l == 0:                                                              # opening inv == 0 at age 0 (initial value without production)
            return model.initial_inv_fresh[p,r,l]
        else:
            if l == 1:
                return model.xpr[t-1,p,r] - model.il[t-1,p,r,0]       # Inventroy produced at t is aged == 0
            else:
                return model.fresh_inv_qoh[t-1,p,r,l-1] - model.il[t-1,p,r,l-1]
    model.fresh_inv_qoh = Expression(model.T, model.INV_Fresh, rule = expression_gen2)       # Expression to derive Fresh Inventory(opening) quantity_on_hand(qoh) by age

    def expression_gen3(model,t,p,r):
        if t == 0:
            return model.initial_inv_frozen[p,r]
        else:
            return model.inv_frozen[t-1,p,r] + model.x_freezing[t-1,p,r] - model.u_frozen[t-1,p,r] - sum(model.a_flex[t,p,rng,r,2] for rng in model.wt_set2[r])
    model.inv_frozen = Expression(model.T, model.P, model.R, rule = expression_gen3)            # Expression to derive Frozen Inventroy(opening) quantity on hand total without age

    def inv_requirement_balance1(model,t,p,r,l):
        return model.ifs[t,p,r,l] == model.fresh_inv_qoh[t,p,r,l]
    model.A10Constraint = Constraint(model.T, model.INV_Fresh, rule = inv_requirement_balance1)   # Fresh Inventory Variable which makes fresh_inv_qoh >= 0

    def inv_requirement_balance2(model,t,p,r):
        return model.ifz[t,p,r] == model.inv_frozen[t,p,r]
    model.A11Constraint = Constraint(model.T, model.P, model.R, rule = inv_requirement_balance2)   # Frozen Inventory Variable which makes inv_frozen >= 0

    def expression_gen4(model,t,p,r):
        return sum(model.fresh_inv_qoh[t,p,r,l1] for l1 in range(int(model.L[p,r])+1))         # Total Inventory(opening) Fresh == sum of inventory at all ages
    model.total_inv_fresh = Expression(model.T,model.P,model.R, rule = expression_gen4)

    def limit_inv_usage(model,t,p,r):                                                           # Inventory Usage <= Inital Available + produced
        return model.total_inv_fresh[t,p,r] + model.xpr[t,p,r] >= model.fresh_inv_used[t,p,r]
    model.A12Constraint = Constraint(model.T, model.P, model.R, rule = limit_inv_usage)

    def balance_inv_usage(model,t,p,r):                                                        # Conserve Inventory Quantity between two consecutive days
        if t == 0:                                                                              # inv_fresh for all ages + model.fresh_part_production[t,p,r] =< model.inv_usage[t,p,r])
            return Constraint.Skip
        else:
            return model.total_inv_fresh[t-1,p,r] + model.xpr[t-1,p,r] - model.fresh_inv_used[t-1,p,r] == model.total_inv_fresh[t,p,r]
    model.A12_1Constraint = Constraint(model.T, model.P, model.R, rule = balance_inv_usage)

    def freeze_expiring_inventory(model,t,p,r):                                         # Freeze Inv at the age = shelf life  >> need to check 'l' or 'l-1'
        if list(model.T)[-1] == t:  ### Temporary >> Fixture to numerical residuals (decimal) >>  if planning horizon is t then t+1 is the data req
            return Constraint.Skip  ###        +++++++++++++++++++++++++++++++++++++++++++++  Need Revision +++++++++++++++++++++++++++++++++
        max_life = model.L[p,r]
        return model.fresh_inv_qoh[t,p,r,max_life] - model.fresh_inv_used[t,p,r] <= 0
    model.A13Constraint = Constraint(model.T, model.P, model.R, rule = freeze_expiring_inventory)

    ### Demand Satisfaction Constraints

    def fresh_requirement_balance(model,t,p,r):
        return model.u_fresh[t,p,r] + model.v_fresh[t,p,r] == sum(model.sales_order[t,c,p,r,1,0] for c in model.C_priority)
    model.requirement_balance1 = Constraint(model.T, model.P, model.R, rule = fresh_requirement_balance)    # Sold + Unsold Fresh Without Marination

    def fresh_m_requirement_balance1(model,t,p,r):
        return model.um_fresh[t,p,r] + model.vm_fresh[t,p,r] == sum(model.sales_order[t,c,p,r,1,1] for c in model.C_priority)
    model.requirement_balance2 = Constraint(model.T, model.P, model.R, rule = fresh_m_requirement_balance1)   # Sold + Unsold Fresh with Marination

    def fresh_m_requirement_balance2(model,t,p,r):
        return model.um_fresh[t,p,r] == model.x_marination[t,p,r]
    model.requirement_balance3 = Constraint(model.T, model.P, model.R, rule = fresh_m_requirement_balance2)    # Marination process is Make to Order

    def frozen_requirement_balance(model,t,p,r):
        return model.u_frozen[t,p,r] + model.v_frozen[t,p,r] == sum(model.sales_order[t,c,p,r,2,0] for c in model.C_priority)
    model.requirement_balance4 = Constraint(model.T, model.P, model.R, rule = frozen_requirement_balance)    # Sold + Unsold Frozen Products without Marination

    def order_requirement_balance1(model,t,p,r):
        lst_p1 = set(model.order_group[t,1,p,r,1,0])
        lst_p2 = set(model.order_group[t,2,p,r,1,0])
        my_set = lst_p1.union(lst_p2)
        if my_set == set():
            return Constraint.Skip
        else:
            return model.u_fresh[t,p,r] == sum(model.order_qty_supplied[o] for o in my_set)
    model.requirement_balance5 = Constraint(model.T, model.P, model.R, rule = order_requirement_balance1)    # Sold fresh = total quantity contained in order

    def order_requirement_balance2(model,t,p,r):
        lst_p1 = set(model.order_group[t,1,p,r,1,1])
        lst_p2 = set(model.order_group[t,2,p,r,1,1])
        my_set = lst_p1.union(lst_p2)
        if my_set == set():
            return Constraint.Skip
        else:
            return model.um_fresh[t,p,r] == sum(model.order_qty_supplied[o] for o in my_set)
    model.requirement_balance6 = Constraint(model.T, model.P, model.R, rule = order_requirement_balance2)    # Sold fresh = total quantity contained in orders

    def order_requirement_balance3(model,t,p,r):
        lst_p1 = set(model.order_group[t,1,p,r,2,0])
        lst_p2 = set(model.order_group[t,2,p,r,2,0])
        my_set = lst_p1.union(lst_p2)
        if my_set == set():
            return Constraint.Skip
        else:
            return model.u_frozen[t,p,r] == sum(model.order_qty_supplied[o] for o in my_set)
    model.requirement_balance7 = Constraint(model.T, model.P, model.R, rule = order_requirement_balance3)    # Sold fresh = total quantity contained in orders

    def order_fulfillment_limiter(model,o):
        return model.order_qty_supplied[o] <= model.order_qty[o]
    model.requirement_balance8 = Constraint(model.O, rule = order_fulfillment_limiter)                # Max Quantity supplied in order <= sales order qty

    def flex_size_fulfillment1(model,t,p,rng,typ):
        return sum(model.a_flex[t,p,rng,r,typ] for r in model.wt_set1[rng]) == model.u_flex[t,p,rng,typ]
    model.fulfillment1 = Constraint(model.T,model.P,model.RNG, model.P_type, rule = flex_size_fulfillment1)

    def flex_size_fulfillment2(model,t,p,rng,typ):
        return model.u_flex[t,p,rng,typ] + model.v_flex[t,p,rng,typ] == sum(model.flex_sales_order[t,c,p,rng,typ,0] for c in model.C_priority)
    model.fullfillment2 = Constraint(model.T, model.P, model.RNG, model.P_type, rule = flex_size_fulfillment2)

    def order_requirement_balance4(model,t,p,rng):
        lst_p1 = set(model.flex_order_group[t,1,p,rng,1,0])
        lst_p2 = set(model.flex_order_group[t,2,p,rng,1,0])
        my_set = lst_p1.union(lst_p2)
        if my_set == set():
            return Constraint.Skip
        else:
            return model.u_flex[t,p,rng,1] == sum(model.flex_order_qty_supplied[o] for o in my_set)
    model.requirement_balance9 = Constraint(model.T, model.P, model.RNG, rule = order_requirement_balance4)

    def order_requirement_balance5(model,t,p,rng):
        lst_p1 = set(model.flex_order_group[t,1,p,rng,2,0])
        lst_p2 = set(model.flex_order_group[t,2,p,rng,2,0])
        my_set = lst_p1.union(lst_p2)
        if my_set == set():
            return Constraint.Skip
        else:
            return model.u_flex[t,p,rng,2] == sum(model.flex_order_qty_supplied[o] for o in my_set)
    model.requirement_balance10 = Constraint(model.T, model.P, model.RNG, rule = order_requirement_balance5)

    def flex_order_fulfillment_limiter(model,o):
        return model.flex_order_qty_supplied[o] <= model.flex_order_qty[o]
    model.requirement_balance11 = Constraint(model.flex_O, rule = flex_order_fulfillment_limiter)                # Max Quantity supplied in order <= sales order qty

    ## Capacity Constraints ###################################  (Checking on UOM Pending)
    """
    def capacity_gen1(model,t,p,r):
        return model.x_freezing[t,p,r] <= model.process_capacity['freezing']*24
    model.A14Constraint = Constraint(model.T, model.P, model.R, rule = capacity_gen1)

    def capacity_gen2(model,t,p,r):
        return model.x_marination[t,p,r] <= model.process_capacity['marination']*24
    model.A15Constraint = Constraint(model.T, model.P, model.R, rule = capacity_gen2)

    def capacity_gen3(model,t,j):
        return sum(model.zkj[t,r,k,j] for r,k,j1 in model.indx_rkj if j == j1) <= 24*model.cutting_capacity[j]
    model.A16Constraint = Constraint(model.T, model.J, rule  = capacity_gen3)
    """
    # Costing Expressions : selling_gains - Op Cost - Inv holding ##########################

    def expression_gen6(model):
        return sum(model.order_qty_supplied[o]*model.order_sp[o] for o in model.O)
        # return sum(model.u_fresh[t,p,r]*model.selling_price[p,r,1,0] + model.um_fresh[t,p,r]*model.selling_price[p,r,1,1] + model.u_frozen[t,p,r]*model.selling_price[p,r,2,0] for p in model.P for r in model.R)
    model.selling_gains = Expression(rule = expression_gen6)   # Calculated selling gains obtained from satisfied demand of SKU's

    def expression_gen7(model,t):
        return sum(sum(model.zkj[t,r,k,j] for r,k,j1 in model.indx_rkj if j == j1)*model.unit_cutting_cost[j] for j in model.J) + sum(model.x_freezing[t,p,r]*model.unit_freezing_cost + model.x_marination[t,p,r]*model.unit_marination_cost for p in model.P for r in model.R)
    model.operations_cost = Expression(model.T, rule = expression_gen7)   # Calculating total cost incurred in processing in cutup + freezing + marination process

    def expression_gen8(model,t):
        return sum(model.total_inv_fresh[t,p,r]*model.inventory_holding_cost[p,r,1] + model.ifz[t,p,r]*model.inventory_holding_cost[p,r,2] for p in model.P for r in model.R)
    model.holding_cost = Expression(model.T, rule = expression_gen8)        # Calculation total Cost Incurred to hold the imbalance inventory

    def expression_gen9(model):
        return A*model.selling_gains - B*(sum(model.operations_cost[t] for t in model.T) + sum(model.holding_cost[t] for t in model.T))
    model.profit_projected = Expression(rule = expression_gen9)            # Profit Equation

    ## Temporary Forcing Constraints (Testing) #########################

    # def force1(model):
    #     return model.zj[0,1,1] >= 100
    # model.F1Constraint = Constraint(rule = force1)

    # def force1(model):
    #     return sum(model.v_frozen[t,p,r] + model.vm_fresh[t,p,r] + model.v_fresh[t,p,r] for t in model.T for p in model.P for r in model.R) == 0
    # model.F1Constraint = Constraint(rule = force1)

    # def force11(model):
    #     return sum(model.x_freezing[t,p,r] for t in model.T for p in model.P for r in model.R) == 0
    # model.F11Constraint = Constraint(rule = force11)

    # def force12(model):
    #     return sum(model.x_marination[t,18,1] for t in model.T) == 10
    # model.F111Constraint = Constraint(rule = force12)

    # def stop_freezing(model):
    #     return sum(model.x_freezing[t,p,r] for t in model.T for p in model.P for r in model.R) == 0
    # model.F12Constraint = Constraint(rule = stop_freezing)

    ## Objective Function and Scenario Selection ############################################

    def obj_fcn(model):

        if scenario == 1:

            def carcass_availability(model,t,r):
                return model.z[t,r] <= model.H[t,r]
            model.A0Constraint = Constraint(model.T, model.R, rule = carcass_availability)      # Total Number of Birds Cut < Bird Available

            def fullfillment_policy1(model,t,c,p,r,typ,m,o):
                return model.coef_serv_L[t,c,p,r,typ,m]*model.order_qty[o] == model.order_qty_supplied[o]
            model.SC1_Constratint1 = Constraint(model.indx_o_filling, rule = fullfillment_policy1)     # Equality of serive level for all the orders

            def xy_relationship(model,t,c,p,r,typ,m,o):
                return model.BigM*model.coef_serv_L_indicator[t,c,p,r,typ,m] >= model.coef_serv_L[t,c,p,r,typ,m]
            model.binary_relationship = Constraint(model.indx_o_filling, rule = xy_relationship)  # Indicator = 1 if value > 0

            def fullfillment_policy2(model,t,c,p,r,typ,m,o):
                return model.coef_serv_L[t,1,p,r,typ,m] >= model.coef_serv_L_indicator[t,2,p,r,typ,m]
            model.SC1_Constraint2 = Constraint(model.indx_o_filling, rule = fullfillment_policy2)        ## For a product is a customer prior 2 is served then making sure all priority 1 customers are serviced first

            return -1*model.profit_projected
            # return sum(model.z[t,r] for t in model.T for r in model.R) + sum((3-t)*model.x_freezing[t,p,r] for t in model.T for p in model.P for r in model.R)

        elif scenario == 2:

            def produce_fresh_for_p1(model,t,p,r):
                return model.u_fresh[t,p,r] >= model.sales_order[t,1,p,r,1,0]
            model.SC3_Constraint1 = Constraint(model.T, model.P, model.R, rule = produce_fresh_for_p1) # Total Sales > Priority Fulfillment

            def produce_fresh_m_for_p1(model,t,p,r):
                return model.um_fresh[t,p,r] >= model.sales_order[t,1,p,r,1,1]
            model.SC3_Constraint2 = Constraint(model.T, model.P, model.R, rule = produce_fresh_m_for_p1)  # Total Sales > Priority Fulfillment

            def produce_frozen_for_p1(model,t,p,r):
                return model.u_frozen[t,p,r] >= model.sales_order[t,1,p,r,2,0]
            model.SC3_Constraint3 = Constraint(model.T, model.P, model.R, rule = produce_frozen_for_p1)   # Total Sales > Priority Fulfillment

            def produce_flex_for_p1(model,t,p,rng,typ):   # Both flex and frozen combined
                return model.u_flex[t,p,rng,typ] >= model.flex_sales_order[t,1,p,rng,typ,0]
            model.SC3_Constraint4 = Constraint(model.T, model.P, model.RNG, model.P_type, rule = produce_flex_for_p1) # Total sales > Priority fulfillment for flex typ

            def order_commitment(model,o):
                return model.order_qty_supplied[o] >= model.order_sla[o]
            model.SC3_Constraint5 = Constraint(model.O, rule = order_commitment)      # For each order >> Quantity supplied > committed Service Level

            def flex_order_commitment(model,fo):
                return model.flex_order_qty_supplied[fo] >= model.flex_order_sla[fo]
            model.SC3_Constraint6 = Constraint(model.flex_O, rule = flex_order_commitment)

            return -1*model.profit_projected
            # return sum(model.z[t,r] for t in model.T for r in model.R) + sum((3-t)*model.x_freezing[t,p,r] for t in model.T for p in model.P for r in model.R)
        else:
            raise AssertionError("Invalid Scenario Selection.\n\t\tThe available options are : 1, 2, 3\n\t\tPlease retry with a valid parameter\n\t\tError Code 200A")
            return 0
    model.objctve = Objective(rule = obj_fcn,sense = minimize)

    return model

if __name__=="__main__":
    print ("Pyomo : Concrete Model Generator Method \nImbalance Parts Optimization \nPROMPT : Invalid Execution! \nACTION : Supply Parameters!")
