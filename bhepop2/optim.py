
import numpy as np
import time
from numpy.linalg import norm
from .utils import log
import logging as lg


def minxent_gradient(q: np.ndarray, G: np.ndarray, eta: np.ndarray, lambda_: np.ndarray, maxiter: int):
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

    iter_general = 0
    Lagrangians = []

    common_ratio_descending = 0.5
    common_ratio_ascending = 2.0

    lambda_old = None

    dist = 1

    while iter_general <= maxiter and dist >= 1e-08:
        iter_general += 1
        log("Mixent gradient iteration : " + str(iter_general), lg.DEBUG)

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
        lambda_old = lambda_.copy()
        lambda_ = lambda_.copy()
        lambda_new = lambda_.copy()
        level_objective_new = level_objective
        did_ascent = False
        did_descent = False

        while not (did_ascent and did_descent):
            lambda_new = lambda_old - alpha * f_old
            lambda0 = np.log(q.T.dot(np.exp(-G.T.dot(lambda_new))))
            level_objective_new = lambda0 + np.sum(lambda_new * eta)

            # level_objective_new = objective(q=q, G=G, eta=eta, lambda_=lambda_new)
            if level_objective_new > level_objective:
                alpha_descent *= common_ratio_descending
                alpha = alpha_descent
                did_descent = True

            else:
                level_objective = level_objective_new
                lambda_ = lambda_new.copy()
                alpha_ascent *= common_ratio_ascending
                alpha = alpha_ascent
                did_ascent = True

            # fail to converge
            if alpha < 1e-06 or alpha > 1e2:
                log("Leaving gradient descent due to high alpha value", lg.DEBUG)
                break

        dist = norm(lambda_ - lambda_old)

    endTime = time.time()
    duration_iter = endTime - startTime
    # lambda0 = np.log(np.sum(q * np.exp(-lambda_.dot(G[1:, :]))))

    Exp_neg_Gt_lambda = np.exp(-G.T.dot(lambda_))  # POV exp(-Gt.lambda)
    lambda0 = np.log(q.T.dot(Exp_neg_Gt_lambda))
    pk = (q * Exp_neg_Gt_lambda) / (q.T.dot(Exp_neg_Gt_lambda))
    # pi_solve = q * np.exp(-lambda0) * np.exp(-lambda_.dot(G[1:, :]))
    Lagrangians.append([lambda0, lambda_.tolist()])
    test_pierreolivier = G.dot(pk) - eta

    return pk.tolist(), lambda_
