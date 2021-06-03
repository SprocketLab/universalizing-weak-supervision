import numpy as np
import copy 
import random 
from ranking_utils import *
import scipy
from math import exp,log
from collections import defaultdict
import itertools

class Mallows:
    def __init__(self,rank_utils,theta):
        
        self.rank_utils  = rank_utils
        d = len(rank_utils.items)
        
        dist_counts = rank_utils.get_dist_counts()
        
        Z_theta = exp(self.log_partition_fun(theta,d))
        
        dist_probs  =  np.array([ (dist_counts[i]/Z_theta)*exp(-theta*i) for i in range(len(dist_counts))])
        #print(sum(dist_probs))
        dist_probs[dist_probs<1e-20] = 1e-20
        #print(dist_probs)
        self.dist_counts = dist_counts
        self.dist_probs = dist_probs
        self.theta = theta
        self.dict_dist_perm = None
        pass
    
    def swap_pair(self,ranking,i,j):
        temp = ranking[i]
        ranking[i] = ranking[j]
        ranking[j] = temp
    
    def get_random_perm_at_dist_k(self,center_pi,k):
        d = len(center_pi)
        s = copy.deepcopy(center_pi)
        #print(type(s))
        s = s.permutation
        pairs_avlbl = []
        
        pairs_taken = [[0 for i in range(d)] for j in range(d)]
        
        for i in range(0,d-1):
            pairs_avlbl.append((s[i],s[i+1]))
            
        
        cur_k = 0
        #print(s)
        pairs_history_count = defaultdict(int)
        
        pi_star_0 = center_pi.permutation[0]
        
        C = self.rank_utils.dist_counts
        
        #print(k,C[d-1][k],C[d][k])
        #ng = C[d-1][k]
        #n_tot = C[d][k]
        #pg = ng/n_tot
        #p_top = (1-pg)#/(d-1)
        #p_oth =  pg#/()
        
        while(cur_k<k):
            
            probs = []
            k_ = cur_k + 1
            #print(k_,d,C[d-1][k_],C[d][k_])
            #print(C)
            #p_top = ((1-C[d-1][k_]/C[d][k_]) -(1-C[d-1][k_-1]/C[d][k_-1]))
            #print(k,p_top)
            #n_o = 0
            
            #for x,y in pairs_avlbl:
            #    if(x== pi_star_0 or y == pi_star_0):
            #        probs.append(p_top)
            #    else:
            #        probs.append(p_oth)
                    #n_o+=1
            #if(n_o>0):
            #    p_others = (1-sum(probs))/n_o
            #    for i in range(len(probs)):
            #        if(not probs[i]>0):
            #            probs[i]=p_others
            #print(probs)
            #print(pairs_avlbl)
            #print(sum(probs))
            
            
            for x,y in pairs_avlbl:
                probs.append(pairs_history_count[str(x)+','+str(y)])
            #print('b',probs)
            probs = np.array(probs)
            probs = np.exp(-probs)
            probs = probs/np.sum(probs)
            #print(probs)
            #print(pairs_avlbl)
            #print(probs)
            
            x = np.random.multinomial(1,probs)
            sp_idx = np.argmax(x)
            sp = pairs_avlbl[sp_idx]
            
            #sp = random.sample(pairs_avlbl,1)[0]
            
            for x,y in pairs_avlbl:
                pairs_history_count[str(x)+','+str(y)]+=1
                pairs_history_count[str(y)+','+str(x)]+=1
            
            #print('sp',sp)
            #print('pairs_avlbl',pairs_avlbl)
            i = s.index(sp[0])
            j = s.index(sp[1])
            self.swap_pair(s,i,j)
            cur_k+=1
            
            pairs_taken[sp[0]][sp[1]]=1
            pairs_taken[sp[1]][sp[0]]=1
            
            pairs_avlbl = []
            
            for i in range(0,d-1):
                if(pairs_taken[s[i]][s[i+1]]==0):
                    pairs_avlbl.append((s[i],s[i+1]))
                    
            #print(s)
            
        return s
    
    def build_perm_dict(self,center_pi):
        d = len(center_pi)
        dict_dist_perm = defaultdict(list)
        perms = itertools.permutations(center_pi.permutation)
        p = (d*(d-1))/2
        for perm in perms:
            perm_r = Ranking(perm,self.rank_utils)
            dist = int(self.rank_utils.kendall_tau_distance(perm_r,center_pi,normalize=False))
            dict_dist_perm[dist].append(list(perm_r))
            
        self.dict_dist_perm = dict_dist_perm
        
    def sample(self,center_pi):
        
        #theta = self.theta
        if(len(center_pi) <10):
            self.build_perm_dict(center_pi)
            #for k in self.dict_dist_perm.keys():
            #    print(k,len(self.dict_dist_perm[k]))
        
        
        err = []
        center_pi_0 = copy.deepcopy(center_pi)
        
        p = len(self.rank_utils.unique_pairs)
        
        # get a distance from mallows,
        f = np.nonzero(np.random.multinomial(1,self.dist_probs))[0][0]
        
        '''
        f = np.random.exponential(1.0/theta,1)*p # *p
        f0 = int(f)
        f = min(f0,len(self.rank_utils.unique_pairs))
        '''
        if(not self.dict_dist_perm is None):
            best_s = random.sample(self.dict_dist_perm[f],1)[0]
            #print(best_s.permutation)
        else:
            if(f>p/2):
                center_pi = center_pi.reverse()#list(reversed(center_pi))
                f = p -f 

            best_s = self.get_random_perm_at_dist_k(center_pi,f)

        best_s = Ranking(best_s,self.rank_utils)

        err.append(abs(f-self.rank_utils.kendall_tau_distance(best_s,center_pi_0)))
        
        return best_s, np.mean(err),f
    
    def estimate_accuracy(self,n_samples,center_pi):
        d = len(center_pi)
        y = center_pi[0]
        samples = [self.sample(center_pi)[0] for i in range(n_samples)]
        Y_ = [ samples[i].permutation[0]==y for i in range(n_samples)]
        return sum(Y_)/n_samples
    
    @staticmethod
    def find_eq_theta_to_p0(d,p_0):
        r_utils = RankingUtils(d)

        D = r_utils.build_dist_counts()
        C = [D[d-1][i] for i in range(((d-1)*(d-2))//2 +1)]
        
        def A1(theta):
            return sum([C[i]*exp(-i*theta) for i in range(len(C))])

        def A1_jac(theta):
            return sum([-i*C[i]*exp(-i*theta) for i in range(len(C))])
    
        def fun(theta):
            #print(theta)
            log_z_theta = Mallows.log_partition_fun(theta,d)
            A1_theta = A1(theta)
            return 0.5*(log(A1_theta) -log_z_theta - log(p_0))**2

        def jac(theta):
            et = exp(-theta)
            log_z_theta = Mallows.log_partition_fun(theta,d)
            A1_theta = A1(theta)
            A1_jac_theta = A1_jac(theta)
            return (log(A1_theta) -log_z_theta - log(p_0))*( A1_jac_theta/A1_theta -mlw.d_log_partition(theta,d))

        theta_0 = 0.01
        out = scipy.optimize.minimize_scalar(fun,bracket=(0.01,10),bounds = (0.01,10),tol=1e-50,
                                             method='bounded',options={'disp':False})
        #print(out)
        theta = out['x']
        p0_tilde = A1(theta)/exp(Mallows.log_partition_fun(theta,d))
        
        return theta, p0_tilde 

    @staticmethod
    def log_partition_fun(theta,d):
        out =0.0
        Zo = 1.0
        for k in range(1,d):
            #Zo *= (1-np.exp(-theta*(k+1)))/(1-np.exp(-theta))
            out+= log((1-exp(-(k+1)*theta))/(1-exp(-theta)))
        return out #np.log(Zo)
    
    @staticmethod
    def d_log_partition_1(theta,d):
        return (Mallows.log_partition_fun(theta+1e-5,d)-Mallows.log_partition_fun(theta,d))/1e-5
    
    @staticmethod
    def d_log_partition(theta,d):
        out = 0.0
        for k in range(1,d):
            out+=  (k+1)/(exp((k+1)*theta) -1 ) - 1/(exp(theta)-1)
        return out
    
    @staticmethod
    def d_d_log_partition(theta,d):
        out = 0.0

        for k in range(1,d):
            a = exp((k+1)*theta)
            b = exp(theta)
            out+=  -((k+1)**2*a)/((a-1 )**2) + b/((b-1)**2)
        return out

    @staticmethod
    def estimate_theta(d,mu_hat):
        '''
            mu_hat = \hat{E}[d_\tau(\pi,\pi^*)]
        '''
        theta_0 = 11
        
        def f(theta):
            return 0.5*(-Mallows.d_log_partition(theta,d) - mu_hat)**2
        def jac(theta):
            return -(-Mallows.d_log_partition(theta,d) - mu_hat)*(Mallows.d_d_log_partition(theta,d))
        
        out = scipy.optimize.minimize(f, x0=[theta_0], jac=jac,bounds=[(0.1,10.0)],tol=1e-18)
        
        return out.x[0]
