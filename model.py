import itertools
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from sklearn.cluster import spectral_clustering
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm




class NetworkSSC:
    
    
    # Initialization
    def __init__(self, adata, lambda1_lst, lambda2_lst, lambda3 = 1, alpha = 100, n_pcs = 100, n_partition = 5000, tol1 = 1e-3, tol2 = 1e-3, kmax = 1e3, seed = 42):
        self.adata = adata
        self.labels = adata.obs['labels']
        self.lambda1_lst = lambda1_lst
        self.lambda2_lst = lambda2_lst
        self.lambda3 = lambda3
        self.alpha = alpha
        self.n_pcs = n_pcs
        self.n_partition = n_partition
        self.tol1 = tol1
        self.tol2 = tol2
        self.kmax = kmax
        self.seed = seed
    
    
    # Build network
    def build_network(self, adata):
        
        # load data
        genes = np.array(adata.var.index.tolist())
        network = pd.read_table('HomoSapiens_binary_hq.txt')
        network = network.loc[:, ['Gene_A', 'Gene_B']]
        
        # extract network
        network_ind = []
        for i in range(network.shape[0]):
            if network.iloc[i, 0] in genes and network.iloc[i, 1] in genes:
                network_ind.append(i)
        network = network.iloc[network_ind, :]
        network = network[network['Gene_A'] != network['Gene_B']]
        
        # compute adjacency matrix
        adj = np.zeros((len(genes), len(genes)))
        for i in range(network.shape[0]):
            gene1 = network.iloc[i, 0]
            gene2 = network.iloc[i, 1]
            gene1_ind = np.where(gene1 == genes)[0][0]
            gene2_ind = np.where(gene2 == genes)[0][0]
            adj[gene1_ind, gene2_ind] = 1
        adj += adj.T - np.diag(np.diag(adj))
        adj[np.where(adj == 2)] = 1
        
        # compute degree matrix
        rows, cols = np.where((adj == 1) | (adj == 2))
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        for n in range(0, adj.shape[0]):
            G.add_node(n)
        G.add_edges_from(edges)
        dist = list(nx.all_pairs_shortest_path_length(G))
        m = len(dist)
        degree, dist_matrix = np.zeros((m, m)), np.zeros((m, m))
        print("-" * 20, end = '')
        print("Constructing Network", end = '')
        print("-" * 20)
        for i in tqdm(range(m)):
            values = np.array(list(dist[i][1].values()))
            degree[i, i] = sum(values == 1)
            values = np.array(list(dist[i][1].values()))
            position = zip([i] * len(dist[i][1].keys()), list(dist[i][1].keys()))
            position = np.array(list(position))
            dist_matrix[position[:, 0], position[:, 1]] = list(dist[i][1].values())
        dist_matrix[np.where(dist_matrix == 0)] = 1000
        dist_matrix = dist_matrix + np.diag(-np.diag(dist_matrix))
        
        # normalize Laplacian matrix
        degree = degree + np.diag([0.01] * degree.shape[0])
        L = np.identity(degree.shape[0]) - np.dot(np.dot(np.linalg.inv(degree ** 0.5), adj), np.linalg.inv(degree ** 0.5))
        
        return np.transpose(adata.X), L
    
    
    # Loss function
    def loss(self, X, L, Q, Z, lambda1, lambda2):
        f1 = 0.5 * (np.linalg.norm(np.dot(Q, X) - np.dot(np.dot(Q, X), Z))) ** 2
        f2 = lambda1 * np.linalg.norm(Z, ord = 1)
        f3 = 0.5 * lambda2 * np.trace(np.dot(np.dot(Q, L), np.transpose(Q)))
        return f1 + f2 + f3
    
    
    # Compute matrix C
    def C(self, X, Q, Z, mu):
        G = np.dot(np.transpose(-np.dot(Q, X)), np.dot(Q, X) - np.dot(np.dot(Q, X), Z))
        C = Z - G / mu
        return C
    
    
    # Optimization
    def optimize(self, X, L, lambda1, lambda2):
        
        # initialization
        np.random.seed(self.seed)
        Q = PCA(n_components = self.n_pcs).fit(np.transpose(X)).components_
        Z = np.dot(np.linalg.inv(np.dot(np.transpose(np.dot(Q, X)), np.dot(Q, X)) + self.lambda3 * np.identity(X.shape[1])), np.dot(np.transpose(np.dot(Q, X)), np.dot(Q, X)))
        Z = Z - np.diag(np.diag(Z))
        mu = self.alpha * (np.linalg.norm(np.dot(Q, X), ord = 'nuc')) ** 2
        
        # iteration
        itr = 0
        
        while True:
            
            # store initial values
            Z0 = Z
            Q0 = Q
            
            # Update Z
            Z = np.maximum(self.C(X, Q, Z, mu) - lambda1/mu, 0) + np.minimum(self.C(X, Q, Z, mu) + lambda1/mu, 0)
            Z = Z - np.diag(np.diag(Z))
            
            # Update Q
            H = X - np.dot(X, Z)
            M = np.dot(H, np.transpose(H)) + lambda2 * L
            Q = np.transpose(scipy.linalg.eigh(M, eigvals = (0, self.n_pcs - 1))[1])

            # Update mu
            mu = self.alpha * (np.linalg.norm(np.dot(Q, X), ord = 'nuc')) ** 2

            # end iteration
            itr += 1
            if np.linalg.norm(Z - Z0) / np.linalg.norm(Z0) <= self.tol1 and abs(self.loss(X, L, Q, Z, lambda1, lambda2) - self.loss(X, L, Q0, Z0, lambda1, lambda2)) <= self.tol2:
                break
            elif itr > self.kmax:
                print('-' * 10 + 'lambda_1 = ' + str(lambda1) + ', lambda_2 = ' + str(lambda2) + ', Warning: Not Converge!' + '-' * 10)
                break
        
        # return symmetric matrix W
        W = 0.5 * (abs(Z) + abs(np.transpose(Z)))
        return W, Q
    
    
    # Parameter tuning
    def tune(self):
        
        # initialization
        candidate = pd.DataFrame(self.expandgrid(self.lambda1_lst, self.lambda2_lst))
        candidate.columns = ['lambda_1', 'lambda_2']
        cluster_output = {}
        nmi = []
        ari = []
        
        # iteration
        for j in tqdm(range(candidate.shape[0])):
            
            lambda1 = candidate.iloc[j, 0]
            lambda2 = candidate.iloc[j, 1]               
            
            # partition the dataset based on dimension
            if self.adata.shape[0] < self.n_partition:
                X, L = self.build_network(self.adata)
                W, Q = self.optimize(X, L, lambda1, lambda2)
            else:
                divide_num = self.adata.shape[0] // self.n_partition + 1
                group_num = self.adata.shape[0] // divide_num
                W = np.zeros((self.adata.shape[1], self.adata.shape[1]))
                Q = []
                for i in range(divide_num):
                    if i == divide_num - 1:
                        adata_part = self.adata[(group_num * i):self.adata.shape[0] , :]
                    else:
                        adata_part = self.adata[(group_num * i):(group_num * (i + 1)) , :]
                    X_temp, L_temp = self.build_network(adata_part)
                    W_temp, Q_temp = self.optimize(X_temp, L_temp, lambda1, lambda2)
                    W += W_temp
                    Q.append(Q_temp)
                W = W / divide_num
            
            para_result = spectral_clustering(W, n_clusters = len(np.unique(self.labels)), random_state = self.seed)
            cluster_output[j] = para_result
            nmi.append(normalized_mutual_info_score(self.labels, para_result))
            ari.append(adjusted_rand_score(self.labels, para_result))
            score_result = pd.concat([candidate, pd.DataFrame(nmi), pd.DataFrame(ari)], axis = 1)
            score_result.columns = ['lambda_1', 'lambda_2', 'NMI', 'ARI']

        # return result
        return cluster_output, score_result, Q
    
    
    # Extract the largest score
    def extract_max(self, score_result, name):
        return score_result.iloc[np.where(score_result[name] == max(score_result[name]))[0], :]
    
    
    # Expand grids
    def expandgrid(self, *itrs):
        product = list(itertools.product(*itrs))
        return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}