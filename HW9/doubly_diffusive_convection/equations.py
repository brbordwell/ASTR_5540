import numpy as np
from dedalus2.public import *

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class thermosolutal_doubly_diffusive_convection:
    def __init__(self, domain):
        self.domain = domain
        
    def set_problem(self, Rayleigh_thermal, Rayleigh_solutal, Prandtl, Lewis):
        logger.info("Ra_T = {:g}, Ra_S = {:g}, Pr = {:g}, Lewis = {:g}".format(Rayleigh_thermal, Rayleigh_solutal, Prandtl, Lewis))
        
        problem = ParsedProblem( axis_names=['x', 'z'],
                                field_names=['u','u_z','w','w_z','T', 'T_z', 'S', 'S_z', 'P'],
                                param_names=['Ra_T', 'Ra_S', 'Pr', 'Lewis'])


        problem.add_equation("1/Pr*dt(w)  - (dx(dx(w)) + dz(w_z)) - Ra_T*T + Ra_S*S + dz(P) = -1/Pr*(u*dx(w) + w*w_z)")
        problem.add_equation("1/Pr*dt(u)  - (dx(dx(u)) + dz(u_z))                   + dx(P) = -1/Pr*(u*dx(u) + w*u_z)")
        problem.add_equation("dt(T) -       (dx(dx(T)) + dz(T_z)) = -u*dx(T) - w*T_z")
        problem.add_equation("dt(S) - Lewis*(dx(dx(S)) + dz(S_z)) = -u*dx(S) - w*S_z")
        problem.add_equation("dx(u) + w_z = 0")

        problem.add_equation("dz(u) - u_z = 0")
        problem.add_equation("dz(w) - w_z = 0")
        problem.add_equation("dz(T) - T_z = 0")
        problem.add_equation("dz(S) - S_z = 0")

        problem.add_left_bc( "S = 1")
        problem.add_right_bc("S = 0")
        problem.add_left_bc( "T = 1")
        problem.add_right_bc("T = 0")
        problem.add_left_bc( "u = 0")
        problem.add_right_bc("u = 0")
        problem.add_left_bc( "w = 0", condition="dx != 0")
        problem.add_left_bc( "P = 0", condition="dx == 0")
        problem.add_right_bc("w = 0")

        problem.parameters['Ra_T']  = Rayleigh_thermal
        problem.parameters['Ra_S']  = Rayleigh_solutal
        problem.parameters['Pr']    = Prandtl
        problem.parameters['Lewis'] = Lewis

        problem.expand(self.domain, order=1)

        return problem
