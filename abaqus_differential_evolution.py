import numpy
from differential_evolution import DifferentialEvolution

'''
TARGET PROBLEM:
Optimise an arbitrary number of bolts on a fully-supported square membrane
such that the deflection of the membrane is minimised.
'''

class AbaqusDifferentialEvolution(DifferentialEvolution):
    '''
    Use Abaqus/Standard to calculate the cost function for the
    Differential Evolution algorithm. The population consists of node objects
    rather than co-ordinates. The input variables are the number and type of
    bolts.
    '''
    def __init__(self, quantity_to_minimise, number_of_bolts, bolt_type):
        '''
        Initialise the problem
        '''
        super(AbaqusDifferentialEvolution, self).__init__()
        self.absolute_bounds = True

    def is_planar(self, node_based_surface):
        '''
        Check if a node-based surface is planar
        (used to determine valid surface pairs)
        '''
        pass

    def create_local_csys(self):
        '''
        Create a coordinate system to describe the mid-plane between the two
        mating surfaces in 2D.
        '''
        pass

    def local_to_global(self):
        '''
        Transform coordinates from the local 2D system to the global 3D system.
        '''

    def get_bounding_vectors(self):
        '''
        Get the lower and upper bounds for the problem in the local 2D
        coordinate system
        '''
        pass

    def create_connector(self, coordinates):
        '''
        Create a connector: a CONN3D2 element with zero length, and two
        distributing couplings to the mating surfaces.
        '''
        pass
