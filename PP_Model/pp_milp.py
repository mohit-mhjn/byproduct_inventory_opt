from __future__ import division
from pyomo.environ import *

# MIP formulation from A MixedIntegerLinearProgramforOperationalPlanninginaMeat PlaningPlant
# V´ıctor M.Albornoz1, MarcelaGonz´alez-Araya2, Mat´ıas C.Gripe1 and SaraV.Rodr´ıguez3
# AVIOTA
# Rishikesh

model = AbstractModel()

# model.m = Param(within=NonNegativeIntegers)
# model.n = Param(within=NonNegativeIntegers)
#  F = Fresh and C = Frozen
model.T = Set()     # planning horizon
model.J = Set()     # cutting patterns
model.L = Set()     # shell life for fresh products
model.H = Param()         # min no of avaiable carcasses of type r in all planning horizon
model.K = Set()
model.Jk = Set(model.K)
model.R = Set()     # type of carcasses
model.P = Set()     # products
model.alpha = Param(model.R)    # proportion of carcasses of type r
model.yd = Param(model.P, model.J, model.R)   #yield of product i in pattern j on carcasses of type r
model.pf = Param(model.P)    # SP of fresh product
model.pc = Param(model.P)    # SP of frozen product
model.c = Param(model.J)     # operational cost of pattern j
model.ce = Param(model.J)     # operational cost of pattern j in OT
model.b = Param()     # freezing cost /kg
model.hf = Param()    # HC F
model.hc = Param()    # HC Frozen
model.sf= Param(model.P)     # penality for unstatisfed fresh
model.sc= Param(model.P)     # penality for unstatisfed frozen
model.F = Param()    # Freezing tunnel capacity at t
model.df = Param(model.P, model.T) # Demand of fresh products
model.dc = Param(model.P, model.T) # Demand of frozen products
model.tau = Param()         # Freezing Process duration
model.Wf = Param()     # warehouse capacity
model.Wc = Param()     # warehouse capacity
model.t = Param(model.J)    #cutting operation for pattern j
model.Ta = Param(model.T)    #avaiable work hours in each time period
model.Te = Param(model.T)       # avaiable overtime hours
model.delta = Param()         # auxilary pattern for better control the available carcass

## Extra Parameters to handle computed indices
model.Td = Set(ordered=True)

model.x = Var(model.P, model.T,domain = NonNegativeReals)   #total quantity of product i to be processed in period t
model.xf = Var(model.P, model.Td,model.Td,domain = NonNegativeReals)        #quantity of fresh product i to process in period t to be sold at t'
model.z = Var(model.J, model.R, model.T, domain = PositiveIntegers)   # no of times to use pattern j on carcass r in period t in RT
model.ze = Var(model.J, model.R, model.T, domain = PositiveIntegers)    # no of times to use pattern j on carcass r in period t in OT
model.xc = Var(model.P, model.Td,domain = NonNegativeReals)        # quantity of frozen product i to process in t
model.vf = Var(model.P, model.T,domain = NonNegativeReals)        # quantity of fresh product i to be sold in t
model.vc = Var(model.P, model.T,domain = NonNegativeReals)        # quantity of frozen product i to be sold in t
model.Ic = Var(model.P, model.Td,domain = NonNegativeReals)        # quantity of frozen product i to be hold in t
model.uf = Var(model.P, model.T,domain = NonNegativeReals)        # unsatisfied demand of fresh product i in t
model.uc = Var(model.P, model.T,domain = NonNegativeReals)        # unsatisfied demand of frozen product i in t
model.HA = Var(model.T,domain = PositiveIntegers)         # No of carcasses to be processed in period t

def objective_function(model):
    
    return sum(sum(model.pf[i]* model.vf[i,t]  + model.pc[i]* model.vc[i,t] for i in model.P) for t in model.T) - sum(sum(model.b*model.xc[i,t]  for i in model.P) for t in model.T)
    - sum(sum(model.hc * model.Ic[i,t]  for i in model.P) for t in model.T)
    - sum(sum(sum(model.hf * model.x[i,t,t+l] for i in model.P) for t in model.T) for l in model.L)
    - sum(sum(model.sf[i]*model.uf[i,t] - model.sc[i]* model.uc[i,t] for i in model.P) for t in model.T)
    - sum(sum(sum(model.c[j]*model.z[j,r,t] + model.ce[j]*model.ze[j,r,t] for j in model.J) for r in model.R) for t in model.T) 
model.OBJ = Objective(rule = objective_function, sense = minimize)



def carcass_availabiltiy(model,r, t):
        return sum(model.z[j,r,t] + model.ze[j, r, t] for j in model.Jk) == model.alpha[r]*model.HA[t]

model.A2Constraint = Constraint(model.R, model.T, rule = carcass_availabiltiy)

def carcass_limit(model):
        return model.delta*model.H <= sum(model.HA[t] for t in model.T) <= model.H
model.A3Constraint = Constraint(rule = carcass_limit)

def cutting_pattern(model,i,t):
        return model.x[i,t] == sum(sum(model.yd[i,j,r] * (model.z[j,r,t] + model.ze[j,r,t]) for j in model.Jk) for r in model.R )
model.A4Constraint = Constraint(model.P, model.T, rule = cutting_pattern)

def avaiable_daily_wh (model,t):
    return sum(sum(model.z[j,r,t]* model.t[j] for j in model.J) for r in model.R) <= model.Ta[t]

model.A5Constraint = Constraint(model.T, rule = avaiable_daily_wh)

def avaiable_daily_ot(model,t):
    return sum(sum(model.ze[j,r,t]* model.t[j] for j in model.J) for r in model.R) <= model.Te[t]

model.A6Constraint = Constraint(model.T, rule = avaiable_daily_ot)

def fresh_frozne_balance(model,i,t):

        return model.x[i,t] == sum(model.xf[i,t,t+l] for l in model.L) + model.xc[i,t]

model.A7Constraint = Constraint(model.P, model.T, rule = fresh_frozne_balance)

def fresh_product_sold(model,i,t):
        return model.vf[i,t] == sum(model.xf[i,t-l,t] for l in model.L)

model.A8Constraint = Constraint(model.P, model.T, rule = fresh_product_sold)

def frozen_product_sold(model,i,t):
    return model.vc[i,t] == model.Ic[i,t-1] + model.xc[i,t-model.tau] - model.Ic[i,t]

model.A9Constraint = Constraint(model.P,model.T, rule = frozen_product_sold)

def demand_frozen_product(model, i, t):
        return model.vc[i,t] + model.uc[i,t] == model.dc[i,t]
model.A10Constraint = Constraint(model.P, model.T, rule = demand_frozen_product)

def demand_fresh_product (model,i, t):
        return model.vf[i,t] + model.uf[i,t] == model.df[i,t]
model.A11Constraint = Constraint(model.P, model.T,  rule = demand_fresh_product)

def freezing_capacity(model, t):
    return sum(sum(model.xc[i,t] for t in range(t-model.tau, t) )for i in model.P) <= model.F

model.A12Constraint = Constraint(model.T, rule = freezing_capacity)

def fresh_product_wh_cap(model, t):
    return sum(sum(sum(model.xf[i,t-l,td] for td in range(t, t - l+1 )) for l in model.L)for i in model.P) <= model.Wf

model.A13Constraint = Constraint(model.T, rule = fresh_product_wh_cap)

def frozen_product_wh_cap(model, t):
    return sum(model.Ic[i,t] for i in model.P) <= model.Wc

model.A14Constraint = Constraint(model.T, rule = frozen_product_wh_cap)

