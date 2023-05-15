
import numpy as np
import time
from numpy.linalg import norm



def minxent_gradient(q: np.ndarray, G: np.ndarray, eta: np.ndarray, lambda_: np.ndarray, maxiter: list):
    startTime = time.time()
    q = q.astype(float)
    G = G.astype(float)
    eta = eta.astype(float)
    lambda_ = lambda_.astype(float)
    G_full = G.copy()
    eta_full = eta.copy()
    G = G_full[1:, :]
    eta = eta_full[:, 1:]
    dimG = G.shape
    ncons = dimG[0]  # POV number of constrains wihout the natural constraints
    nproba = dimG[1]  # POV number of probabilites to find
    q = q.reshape(nproba, 1)
    lambda_ = lambda_.reshape(ncons, 1)
    eta = eta.reshape(ncons, 1)

    n_stop_iter = len(maxiter)
    max_iter_general = maxiter[n_stop_iter - 1]
    iter_general = 0
    Lagrangians = []
    Estimates = []
    Duration = np.zeros(n_stop_iter)
    # pdb.set_trace() # POV-debug

    compteur_max_iter = 1
    common_ratio_descending = 1 / 2
    common_ratio_ascending = 2.0
    # lambda0 = np.log(q.T.dot(np.exp(-G.T.dot(lambda_))))

    while True:
        iter_general += 1

        # lambda0 = np.log(np.sum(q * np.exp(-lambda_old.dot(G[1:, :])))) Exp_neg_Gt_lambdaPOV
        Exp_neg_Gt_lambda = np.exp(-G.T.dot(lambda_))  # POV exp(-Gt.lambda)
        lambda0 = np.log(q.T.dot(Exp_neg_Gt_lambda))
        pk = (q * Exp_neg_Gt_lambda) / (q.T.dot(Exp_neg_Gt_lambda))
        level_objective = lambda0 + np.sum(lambda_ * eta)
        f_old = eta - G.dot(pk)

        # f_old = fk(q=q, G=G, eta=eta, lambda_=lambda_old)

        # dev_ent = np.dot(q * np.exp(-lambda0) * np.exp(-lambda_old.dot(G[1:, :])), G)
        # pg = q * np.exp(-lambda0) * np.exp(-lambda_old.dot(G[1:, :]))

        alpha_ascent = 1.0
        alpha_descent = 1.0
        alpha = 1.0
        alpha_old = alpha
        lambda_old = lambda_.copy()
        lambda_ = lambda_.copy()
        lambda_new = lambda_.copy()
        level_objective_new = level_objective
        test_descent = 0
        test_ascent = 0

        while True:
            lambda_new = lambda_old - alpha * f_old
            lambda0 = np.log(q.T.dot(np.exp(-G.T.dot(lambda_new))))
            level_objective_new = lambda0 + np.sum(lambda_new * eta)

            # level_objective_new = objective(q=q, G=G, eta=eta, lambda_=lambda_new)
            if level_objective_new > level_objective:
                alpha_descent *= common_ratio_descending
                alpha_old = alpha
                alpha = alpha_descent
                test_descent = 1

            else:
                level_objective = level_objective_new
                lambda_ = lambda_new.copy()
                alpha_ascent *= common_ratio_ascending
                alpha_old = alpha
                alpha = alpha_ascent
                test_ascent = 1

            # pdb.set_trace() # POV-debug
            # stop if ascend AND descend
            if test_descent * test_ascent > 0.5:
                break

            # laisser valeurs arbitraires et ajouter warning
            if alpha < 1e-06:
                break
            if alpha > 1e2:
                break

        # test convergence à mettre en dur
        if norm(lambda_ - lambda_old) < 1e-08:
            break

        # test nombre d'itérations
        if iter_general > max_iter_general:
            break

        # if iter_general == maxiter[compteur_max_iter - 1]:
        #   break

    endTime = time.time()
    duration_iter = endTime - startTime
    # lambda0 = np.log(np.sum(q * np.exp(-lambda_.dot(G[1:, :]))))

    Exp_neg_Gt_lambda = np.exp(-G.T.dot(lambda_))  # POV exp(-Gt.lambda)
    lambda0 = np.log(q.T.dot(Exp_neg_Gt_lambda))
    pk = (q * Exp_neg_Gt_lambda) / (q.T.dot(Exp_neg_Gt_lambda))
    # pi_solve = q * np.exp(-lambda0) * np.exp(-lambda_.dot(G[1:, :]))
    Lagrangians.append([lambda0, lambda_.tolist()])
    test_pierreolivier = G.dot(pk) - eta

    return pk.tolist(), lambda_.tolist()