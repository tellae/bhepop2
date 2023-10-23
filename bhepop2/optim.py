import numpy as np
from numpy.linalg import norm
from .utils import log
import logging as lg


def minxent_gradient(
    q: np.ndarray, matrix: np.ndarray, eta: np.ndarray, lambda_: np.ndarray, maxiter: int
):
    # remove first constraint
    # TODO : check that first constraint is density constraint instead of discarding it
    matrix = matrix.copy()[1:, :]
    eta = eta.copy()[:, 1:]

    # get array size
    ncons, nproba = matrix.shape
    q = q.reshape(nproba, 1)
    lambda_ = lambda_.reshape(ncons, 1)
    eta = eta.reshape(ncons, 1)

    # remove ?
    Lagrangians = []

    # move to const ?
    common_ratio_descending = 0.5
    common_ratio_ascending = 2.0

    # loop until maximum iteration number or convergence
    iter_general = 0
    dist = 1
    while iter_general <= maxiter and dist >= 1e-08:
        iter_general += 1

        # lambda0 = np.log(np.sum(q * np.exp(-lambda_old.dot(G[1:, :])))) Exp_neg_Gt_lambdaPOV
        Exp_neg_Gt_lambda = np.exp(-matrix.T.dot(lambda_))  # POV exp(-Gt.lambda)
        lambda0 = np.log(q.T.dot(Exp_neg_Gt_lambda))
        pk = (q * Exp_neg_Gt_lambda) / (q.T.dot(Exp_neg_Gt_lambda))
        level_objective = lambda0 + np.sum(lambda_ * eta)
        f_old = eta - matrix.dot(pk)

        alpha_ascent = 1.0
        alpha_descent = 1.0
        alpha = 1.0
        lambda_old = lambda_.copy()
        lambda_ = lambda_.copy()
        did_ascent = False
        did_descent = False

        while not (did_ascent and did_descent):
            lambda_new = lambda_old - alpha * f_old
            lambda0 = np.log(q.T.dot(np.exp(-matrix.T.dot(lambda_new))))
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
            if alpha < 1e-06 or alpha > 1e6:
                log(
                    "Leaving gradient descent due to extreme alpha value : {}".format(alpha),
                    lg.DEBUG,
                )
                break

        dist = norm(lambda_ - lambda_old)

    # log("Leaving mixent after {} iterations".format(str(iter_general)), lg.DEBUG)
    Exp_neg_Gt_lambda = np.exp(-matrix.T.dot(lambda_))  # POV exp(-Gt.lambda)
    lambda0 = np.log(q.T.dot(Exp_neg_Gt_lambda))
    pk = (q * Exp_neg_Gt_lambda) / (q.T.dot(Exp_neg_Gt_lambda))

    # remove ?
    # pi_solve = q * np.exp(-lambda0) * np.exp(-lambda_.dot(G[1:, :]))
    Lagrangians.append([lambda0, lambda_.tolist()])
    test_pierreolivier = matrix.dot(pk) - eta
    # log("test PO : " + str(test_pierreolivier/eta), 10)
    return pk.tolist(), lambda_
