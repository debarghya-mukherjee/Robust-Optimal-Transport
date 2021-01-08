import numpy as np 
import Adam  


def e_dist(A, B):
    A_n = (A**2).sum(axis=1).reshape(-1,1)
    B_n = (B**2).sum(axis=1).reshape(1,-1)
    inner = np.matmul(A, B.T)
    return A_n - 2*inner + B_n


############# The main function #################



def ROBOT(x, mu_true, m = 1, maxiter = 50000, eps = 0.2, sh_eps = 0.01, eta = 0.002, lambda_val = 5):
    n, d = x.shape
    iter = 0
    mu = -1 * np.ones(d)
    v_tilde = np.zeros(n)
    C = 1
    emp_dist = np.ones(n)/n
    v = v_tilde 
    adam_optimizer = Adam.Adam(lr= eta)
    while iter < maxiter:
        for _ in range(100):
            z1 =  np.random.normal(0, 1, (1, d))
            z = z1 + mu 
            cost_vector = e_dist(x, z)
            good_set = np.squeeze(cost_vector < 2 * lambda_val)
            cost_vector[cost_vector > 2 * lambda_val] = 2 * lambda_val 
            cost_vector = np.squeeze(cost_vector)
            v1 = (v_tilde - cost_vector)/sh_eps 
            exp_smooth_vec = np.exp(v1 - v1.max())
            exp_smooth_vec /= exp_smooth_vec.sum()
            v_grad = emp_dist  - exp_smooth_vec
            v_tilde = v_tilde + C * v_grad
            v = (1/(iter//10000 + 1)) * v_tilde + (iter//10000/(iter//10000 + 1)) * v
        v1 = (v - cost_vector)/sh_eps
        pi_opt_now = np.exp(v1 - v1.max())
        pi_opt_now /= pi_opt_now.sum()
        if good_set.sum() == 0:
            mu_grad = np.zeros(d) + 1e-6
        else:
            mu_grad = (2 * mu + 2 * z1) * pi_opt_now[good_set].sum() - 2 * x[good_set, :].T @ pi_opt_now[good_set]
        
        update = adam_optimizer.get_update(grad = mu_grad)
        mu = mu - update
        if iter % 1000 == 0: 
            print('current estimation error', np.linalg.norm(mu))
        iter += 1
        
    return mu



############ An application ################


n = 1000
d = 5

eps = 0.2
mu_true = 0
sigma_true = np.eye(d)
mu_cont = 0.2


contprop = np.int(n * eps)

### generate the contaminated dataset  

x = np.zeros((n, d)) 
x[0:(n - contprop), :] = np.random.normal(mu_true, 1, ((n - contprop), d))
x[(n - contprop):, :] = np.random.normal(mu_cont, 1, (contprop, d))


### Estimate mean using ROBOT 

mu_hat = ROBOT(x, mu_true, m = 1, eta = 0.002, lambda_val = 0.5, maxiter = 100000)    
np.save('filename.npy', np.linalg.norm(mu_hat))


