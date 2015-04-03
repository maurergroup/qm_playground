"""
"""

from qmp.solver.eigensolver import *

solver_type = {
        'scipy': scipy_solver,
        'alglib': alglib_solver,
        #'pimd': pimd_solver,
        }

def solver_init(data, potential):
    param = data.parameters
    solver = solver_type[param['solver']](data=data, potential=potential)

    return solver

def solver_help():
    string = 'Solver types include: \n'
    for k in solver_type.keys():
        string += key+'\n'
    return string
    
