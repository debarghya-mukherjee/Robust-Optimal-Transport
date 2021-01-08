import numpy as np
import ot
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def e_dist(A, B):
    A_n = (A**2).sum(axis=1).reshape(-1,1)
    B_n = (B**2).sum(axis=1).reshape(1,-1)
    inner = np.matmul(A, B.T)
    return A_n - 2*inner + B_n



############ Creating pure mnist and contaminated mnist+fmnist dataset ############

(mnist, _), (_, _) = tf.keras.datasets.mnist.load_data()
(fashion_mnist, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
np.save('mnist.npy', mnist)


eps = 0.2 # Contamination proportion 
samples_no = 10000
samples_fmnist_no = np.int(samples_no * eps)
samples_mnist_no = samples_no - samples_fmnist_no



mnist_idx_choice = np.random.choice(60000, samples_mnist_no)
fmnist_idx_choice = np.random.choice(60000, samples_fmnist_no)


X = np.zeros((samples_no, 784))
X[:samples_mnist_no, :] = mnist[mnist_idx_choice, :, :].reshape(samples_mnist_no, 784)
X[samples_mnist_no:, :] = fashion_mnist[fmnist_idx_choice, :, :].reshape(samples_fmnist_no, 784)
X = X/255


X_label = np.zeros(samples_no)
X_label[samples_mnist_no:] = 1

Y = mnist[np.random.choice(60000, samples_no)].reshape(-1, 784)/255


######### Selection of lambda 

iter = 30
lambda_vec = np.zeros(iter)
for j in range(iter):
    sample_batch = 5000
    
    Y1 = Y[np.random.choice(Y.shape[0], sample_batch, replace=False)]
    Y2 = Y[np.random.choice(Y.shape[0], sample_batch, replace=False)]
    
    cost_now = e_dist(Y1, Y2)
    
    Pival, Pi = ot.emd2(np.ones(sample_batch)/sample_batch, np.ones(sample_batch)/sample_batch, cost_now, processes=4, numItermax=100000, log=False, return_matrix=True)
    cost_now[Pi['G'] > 1e-6].max()
    lambda_vec[j] = np.percentile(cost_now[Pi['G'] > 1e-6], 99)
    
lambda_val = lambda_vec.mean()


########## detection of outlier 

cost_matrix = e_dist(X, Y)
cost_matrix_new = np.copy(cost_matrix)
cost_matrix[cost_matrix > lambda_val] = lambda_val
Pival, Pi = ot.emd2(np.ones(samples_no)/samples_no, np.ones(samples_no)/samples_no, cost_matrix, processes=4, numItermax=100000, log=False, return_matrix=True)
Pimat = Pi['G']
s1 = np.zeros(samples_no)
for i in range(samples_no):
    s1[i] = Pimat[i, np.where(cost_matrix_new[i, :] > lambda_val)[0]].sum()
    
Xhat_label = np.zeros(samples_no)
Xhat_label[np.where(s1 > 1e-6)[0]] = 1

accuracy = 1 - np.linalg.norm(X_label - Xhat_label, 1)/samples_no

########## Plotting the selected outliers  

plt.hist((1/samples_no - s1), bins = 100)
plt.title('ROBOT')

n_col = 30
n_row = 5
idx = np.random.choice(np.where(s1 > 1e-6)[0], 150, replace=False)
outliers = X[idx].reshape(-1, 28, 28)
fig = plt.figure(figsize=(n_row, n_col))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_row, n_col),  # creates 2x2 grid of axes
                 axes_pad=0.,  # pad between axes in inch.
                 )
i = 0
for ax in grid:
    img = outliers[i]
    ax.imshow(img, 'gray')
    ax.set_axis_off()
    i += 1
    
plt.show()