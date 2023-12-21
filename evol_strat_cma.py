import numpy as np
import time

#ES : take as entry a problem-correlated starting mean (a simple python list) an stdr dev (a float),
#     the number of supplementary population (int), the blackbox f (f : R^(len(x_0)) -> R)) and fitness criterium (float)
#The fitness criterium represents the last difference in terms of the 2-norm between the successive estimated minimums
#You can add as many offspring as you want, the initial number is a heuristical best compromise between speed and reliability
#CAREFUL : f takes here an array of shape (n,1), not (n,), so instead of [x,y] use [[x],[y]]

def cma_es(x_0, sigma_0, pop_sup, func, fit_crit):

    #dimension, starting mean and stdr dev, bonus sample size
    n = len(x_0)
    x_mean = x_0
    sigma = sigma_0

    #stopping criteria if taking too much time
    val_stop = 300*(n**2)
    update_wait = 0

    #sample size of offsprings and chosen ones for best fit
    lbd = int(4+np.floor(3*np.log(n)) + pop_sup)

    mu = int(np.floor(lbd/2))

    #weigths initializing
    w_p = np.zeros(lbd)
    for i in range(0, lbd):
        w_p[i] = np.log((lbd+1)/2) - np.log(i+1)

    #setting intermediate constants for calculating parameters
    a_cov = 2    #a_cov can be modified but it's theorically unecessary to do so

    sum_pos = 0
    sum_pos_2 = 0
    for i in range(0,mu):
        sum_pos += w_p[i]
        sum_pos_2 += w_p[i]**2
    
    mueff = (sum_pos**2)/sum_pos_2

    sum_neg = 0
    sum_neg_2 = 0
    for i in range(mu,lbd):
        sum_neg += w_p[i]
        sum_neg_2 += w_p[i]**2

    mueff_neg = (sum_neg**2)/sum_neg_2

    #parameters
    #step size sigma control
    c_sig = (mueff+2)/(n+mueff+5)
    d_sig = 1+2*max(0, (np.sqrt((mueff-1)/(n+1))-1)) + c_sig

    #cov matrix adaptation

    c_c = (4+mueff/n)/(n + 4 + 2*mueff/n)
    c_1 = a_cov/((n+1.3)**2 + mueff)
    c_mu = min(1-c_1, a_cov*(0.25 + mueff + 1/mueff -2)/((n+2)*(n+2)+a_cov*mueff*0.5))

    #weigths
    a_mu = 1+(c_1/c_mu)
    a_eff = 1+2*mueff_neg/(mueff+2)
    a_posd = (1 -c_1 -c_mu)/n*c_mu

    weigths = np.zeros(lbd)
    for i in range(0,mu):
        weigths[i] = w_p[i]/sum_pos
    for i in range(mu,lbd):
        weigths[i] = (min(a_mu, a_eff, a_posd))*w_p[i]/(-sum_neg)

    sum_weights = 0
    for e in weigths:
        sum_weights += e

    #easier when written this way
      
    def norm2(A):
        return np.linalg.norm(A,2)

    
    #initialization
    g = 0                   #number of generations
    val_count = 0           #number of valuations of f
    gen_step = max(1,int(np.floor(1/(10*n*(c_1+c_mu)))))


    B = np.identity(n)      #B,D = np.linalg.eig(C)
    D = np.identity(n)

    C = np.identity(n)      #cov matrix

    pc = np.zeros((n,1))        #evolution paths, for rank-one update
    ps = np.zeros((n,1))        #and sigma

    chi_n = np.sqrt(n)*(1 -1/(4*n) +1/(21*(n**2)))

    error_relative = 1

    print("initiate ok: n (dim of the problem), lbd (offspring number), mu (selection number over lbd) = ", n, lbd, mu)

    #learning

    while (val_count<val_stop) and (error_relative>fit_crit) and (g <= 1000):

        g += 1

        print("offspring generation: ", g)

        x_old = np.copy(x_mean)


        #New sample of search points x_k in arr_x obtained from previous gen's parameters

        arr_z = np.random.normal(0, 1, (lbd, n, 1))
        
        arr_y = np.zeros((lbd,n,1))                            
        for k in range(0,lbd):                                  
            arr_y[k] = B @ (D @ arr_z[k])

        arr_x = x_mean + sigma*arr_y

        #Selection

        f_eval = np.zeros(lbd)
        for k in range(0,lbd):
            f_eval[k] = func(arr_x[k])
            val_count += 1


        sorted_indices = np.argsort(f_eval)

        #Recombination

        y_w = np.zeros((n,1))
        x_mean = np.zeros((n,1))
        for k in range(0,mu):
            y_w += weigths[k]*arr_y[sorted_indices[k]]
            x_mean += weigths[k]*arr_x[sorted_indices[k]]

        #update of the parameters

        C_y_w = np.zeros((n,1))
        for i in range(0,mu):
            C_y_w += weigths[i]*(B @ arr_z[sorted_indices[i]])


        #step size
        ps = (1-c_sig)*ps + np.sqrt(c_sig*(2-c_sig)*mueff)*C_y_w
        sigma = sigma*np.exp((c_sig/d_sig)*((norm2(ps)/chi_n)-1))

        #cov mat adapt
        hsig = norm2(ps)/np.sqrt(1-(1-c_sig)**(2*(g+1)))/chi_n < 1.4+2/(n+1) #here for preventing C
        dhsig = (1-hsig)*c_c*(2 - c_c)                                       #to get too large with ps

        pc = (1-c_c)*pc + hsig*np.sqrt(c_c*(2-c_c)*mueff)*y_w

        weigths_update = np.zeros(lbd)
        for k in range(0,mu):
            weigths_update[k] = weigths[k]
        for k in range(mu,lbd):
            w = weigths[k]*(n)/(norm2(np.dot(B,arr_z[sorted_indices[k]])))**2
            weigths_update[k] = w
        
        R_1 = np.zeros((n,n))
        for k in range(0,lbd):
            R_1 += weigths_update[k]*(arr_y[sorted_indices[k]] @ (arr_y[sorted_indices[k]].T))

        C = (1+c_1*dhsig-c_1-c_mu*sum_weights)*C + c_1*(pc @ (pc.T)) + c_mu*R_1

        #then B and D (not too quick)
        if val_count - update_wait > gen_step:


            update_wait = val_count
            C = np.triu(C) + np.triu(C,1).T

            D1,B = np.linalg.eigh(C)

            D1 = np.real(D1)
            B = np.real(B)
            D = np.diag(np.where(D1 > 0, np.sqrt(D1), 0))

        error_relative = norm2(x_mean-x_old)

    return x_mean