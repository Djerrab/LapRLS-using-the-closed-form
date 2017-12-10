def LapRLS_closedform(X_labeled,y_labels,X_unlabeled,lbda,lbda_u):
    y_labels = 2*y_labels - 1
    l = len(y_labels)
    u = len(X_unlabeled)
    X = np.concatenate((X_labeled,X_unlabeled))
    W = kneighbors_graph(X, n_neighbors=5, mode='connectivity')
    K = rbf_kernel(X)
    
    D = np.diag(np.sum(W,axis=0))
    L = D-W
    Y = np.concatenate((y_labels,np.zeros(u)))
    J = np.diag(np.concatenate((np.ones(l), np.zeros(u))))
    alpha_star = np.linalg.solve(J.dot(K) + lbda*l*np.eye(l+u) + lbda_u*l/((u+l)**2)*L.dot(K),Y)
    
    def f(X_test):
        return alpha_star.dot(rbf_kernel(X,X_test))
    
    return f
