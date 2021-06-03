from math import sqrt
import numpy as np
import random
import copy
from scipy.stats import kendalltau
random.seed(0)
np.random.seed(0)

def solve_3_var_system(a,b,c):
    '''
    x*y = a ; y*z = b ; x*z = c
    '''
    u = a*b*c
    #print(u)
    if(abs(u)<1e-8):
        print(u)
        return [0,0,0]
    s = u//abs(u)
    sab = (a*b)//abs(a*b)
    sbc = (b*c)//abs(b*c)
    sca = (c*a)//abs(c*a)
    sa,sb,sc = s*sbc, s*sca, s*sab
    
    u = abs(u)
    a,b,c = abs(a),abs(b),abs(c)
    
    return [sa*sqrt(u/b),sb*sqrt(u/c),sc*sqrt(u/a)]

def solve_3_var_system_sum(a,b,c):
    t = (a+b+c)/2
    return [t-b,t-c,t-a]
