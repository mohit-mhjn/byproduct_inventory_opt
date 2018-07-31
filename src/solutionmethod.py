"""
This generic fucntion is used for solver selection for the model and respectively calling the solve method.
As the output the solve_model fuction will throw the response as the solved model object and result file of the variables

The selection is available for cplex and cbc to execute for both local and NEOS servers.
Use "initialization setting" to configure

The fucntion solve model consists of two labelled parameters : p_summary and p_log
p_summary : To print the output summary of problem and solution. (number of variables, number of constraints, non-zeros , matrix size, Termination Condition, Branching etc.)
p_log : To print the solver log while processing output (only for the local solver execution, Doesnt work with remote executiong of solver)

The output is an array of length 2
[solved_model_object, result_file]
note that solved_model_object belongs to pyomo model class
"""

def solve_model(model,p_summary = False, p_log = False):       # Custom Solve Method
    import datetime
    # Choose the best solution from the trial pool

    # trial_pool = {
    #                 'sample1':{'solution_file':None,'aspiration':None},
    #                 'sample2':{'solution_file':None,'aspiration':None},
    #                 'sample3':{'solution_file':None,'aspiration':None}
    #              }

    #model1 = copy.deepcopy(model)
    #model2 = copy.deepcopy(model)
    #model3 = copy.deepcopy(model)

    print ("success! \n loading solver......")

    mip_gap = 0
    solver_timeout = 300
    number_of_trials = 1
    solver_sh = 'cbc'     #initialization Setting
    engage_neos = False  #initialization Setting
    threads = 3
    j = 1
    timeout_arguments = {'cplex':'timelimit','cbc':'sec'}
    gap_arguments = {'cplex':'mipgap','cbc':'ratio'}   # Cplex Local Executable will take : "mip_tolerance_mipgap", mipgap is for neos version

    # NEOS Server Library Dependency
    # pyro4
    # suds
    # openopt

    while j < number_of_trials + 1:
        from pyomo.opt import SolverFactory, SolverManagerFactory
        #model.pprint()
        opt = SolverFactory(solver_sh, solver_io = 'lp')
        print ('\ninterfacing solver shell :',solver_sh)
        if engage_neos:
            solver_manager = SolverManagerFactory('neos')
        opt.options[timeout_arguments[solver_sh]]= solver_timeout
        opt.options[gap_arguments[solver_sh]] = mip_gap
        #opt.symbolic_solver_labels=True
        #
        #opt.enable = 'parallel'
        print ('\tsolver options >> \n\n\tTolerance Limits:\n\tmip_gap = %s \n\ttimeout = %s'%(str(mip_gap),str(solver_timeout)))
        print ("\nProcessing Trial Number :",j)
        print ("\nJob Triggered at :",str(datetime.datetime.now()))
        print ('\ngenerating solution ...... !! please wait !!    \n to interrupt press ctrl+C\n')
        try:
            if engage_neos:
                p_log = False
                results = solver_manager.solve(model,opt = opt, tee= True)
            else:
                opt.options['threads'] = threads
                if p_log:
                    opt.options['slog'] = 1
                results = opt.solve(model) # Method Load Solutions is not available in pyomo versions less than 4.x
        except:
            j = j+1
            mip_gap = (j-1)*mip_gap
            solver_sh = 'cplex'
            engage_neos = True
            continue
        #results.write(filename='results'+str(datetime.date.today())+'.json',format = 'json')
        #print (results)
        if str(results['Solver'][0]['Termination condition']) in ['infeasible','maxTimeLimit','maxIterations','intermediateNonInteger','unbounded']:
            j = j+1
            mip_gap = (j-1)*mip_gap
            solver_sh = 'cplex'
            engage_neos = True
            if j == number_of_trials + 1:
                #print (results['Problem'])
                #print (results['Solver'])
                raise AssertionError("Solver Failed with Termination Status : %s"%(str(results['Solver'][0]['Termination condition'])))
                exit(0)
            print ('Terminated by:',str(results['Solver'][0]['Termination condition']))
            print ("\n\nRetrying...\n\n")
        else:
            print ("SUCCESS: Solution Captured!")
            model.solutions.store_to(results)
            #post_process_results()
            break

    if p_summary:
        print (results['Problem'])
        print (results['Solver'])
    print ("\nSolution Retrived at:",str(datetime.datetime.now()))
    return [model,results]
