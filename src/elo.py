import numpy as np
from scipy.optimize import minimize
from collections import deque


def elo_prob(r_i, r_j):
    """
    Elo / Bradley–Terry win probability
    """
    return 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))


def check_connectivity(P):
    """
    Check whether the comparison graph is connected.
    P[i, j] is observed if not NaN.
    """
    N = P.shape[0]
    graph = [[] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i != j and not np.isnan(P[i, j]):
                graph[i].append(j)
                graph[j].append(i)

    visited = [False] * N
    queue = deque([0])
    visited[0] = True

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    return all(visited)


def neg_log_likelihood(R, P, n):
    """
    Negative log-likelihood for Elo / Bradley–Terry model.

    R : (N,) Elo parameters
    P : (N, N) win-rate matrix, NaN if missing
    n : number of comparisons per observed pair
    """
    N = len(R)
    eps = 1e-12
    nll = 0.0

    for i in range(N):
        for j in range(N):
            if i == j or np.isnan(P[i, j]):
                continue

            p_model = elo_prob(R[i], R[j])

            w_ij = n * P[i, j]
            w_ji = n * (1.0 - P[i, j])

            nll -= (
                w_ij * np.log(p_model + eps)
                + w_ji * np.log(1.0 - p_model + eps)
            )

    return nll


def fit_elo_mle(P, n, init_rating=0.0):
    """
    Fit Elo scores via MLE from a win-rate matrix.

    Parameters
    ----------
    P : np.ndarray (N, N)
        Win-rate matrix (win + 0.5 * draw), NaN if not compared.
    n : int
        Number of comparisons per observed pair.
    init_rating : float
        Initial value for all Elo parameters.

    Returns
    -------
    R : np.ndarray (N,)
        Estimated Elo scores with mean = 0.
    """
    P = np.asarray(P)
    N = P.shape[0]

    if not check_connectivity(P):
        raise ValueError(
            "Comparison graph is not connected; Elo scores are not identifiable."
        )

    R0 = np.full(N, init_rating)

    constraint = {
        "type": "eq",
        "fun": lambda R: np.mean(R)
    }

    result = minimize(
        neg_log_likelihood,
        R0,
        args=(P, n),
        method="SLSQP",
        constraints=constraint,
        options={
            "maxiter": 2000,
            "ftol": 1e-9,
            "disp": False
        }
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    return result.x


if __name__ == "__main__":

    P = np.array([
        [np.nan, 30, 60],
        [70, np.nan, 55],
        [40, 45, np.nan]]) / 100

    n = 2097  # This is the size of test set in our paper

    R_centered = fit_elo_mle(P, n)
    R_elo = R_centered + 1000  # conventional Elo scale

    print("Elo estimation results:")
    for i in range(len(R_centered)):
        print(
            f"Method {i}: "
            f"Elo(centered) = {R_centered[i]:7.2f}, "
            f"Elo = {R_elo[i]:7.1f}"
        )
