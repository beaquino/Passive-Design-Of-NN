import picos
import numpy as np

def checkindexes(W,nu_l=-0.05, delta_l=4.74):
    print("Passivity Shrinking Started\n")
    (nl,nc) = np.shape(W)
    k1 = 0
    k2 = 1
    rate = 1
    step = 0.1
    n = np.maximum(nl,nc)
    if nc!=nl:
        Wsl = np.zeros((n,n))
    else:
        Wsl=np.zeros((nl,nc))
    Wsl[:nl,:nc]=W 
    Wsl = rate * Wsl
    print(np.shape(Wsl))
    problem = picos.Problem()
    tau = picos.RealVariable('tau', (n, 1))
    index_matrix = picos.diag((-delta_l, -nu_l)) + np.diag(np.ones(1) / 2, -1) + np.diag(np.ones(1) / 2, 1)
    Plaux = picos.kron(index_matrix, np.identity(n))
    Rvaux = np.concatenate((np.concatenate((np.zeros((n, n)), Wsl,), axis=1), np.concatenate((np.identity(n), np.zeros((n, n))), axis=1)),axis=0)
    T2aux = picos.SymmetricVariable("T2aux", (n, n))
    Laux = picos.diag(picos.RealVariable('Laux', (n, 1)))
    Baux = (-2 * k1 * k2 * (T2aux + Laux) & (k1 + k2) * (T2aux + Laux)) // ((k1 + k2) * (T2aux + Laux) & -2 * (T2aux + Laux))
    Maux = Plaux - Rvaux.T * Baux * Rvaux
    problem.add_constraint(Maux >> 0)
    problem.add_constraint(T2aux >> 0)
    problem.add_constraint(Laux >> 0)

    solution = problem.solve(primals=False)
    #print("Solution Found\n")

    while solution.claimedStatus != 'optimal':
        if rate <= 2*step:
            step = step / 10
        rate = rate - step;
        if nc!=nl:
            Wsl = np.zeros((n,n))
        else:
            Wsl=np.zeros((nl,nc))
        Wsl[:nl,:nc]=W 
        Wsl = rate * Wsl
        Plaux = picos.kron(index_matrix, np.identity(n))
        Rvaux = np.concatenate((np.concatenate((np.zeros((n, n)), Wsl,), axis=1),np.concatenate((np.identity(n), np.zeros((n, n))), axis=1)), axis=0)
        T2aux = picos.SymmetricVariable("T2aux", (n, n))
        Laux = picos.diag(picos.RealVariable('Laux', (n, 1)))
        Baux = (-2 * k1 * k2 * (T2aux + Laux) & (k1 + k2) * (T2aux + Laux)) // ((k1 + k2) * (T2aux + Laux) & -2 * (T2aux + Laux))
        Maux = Plaux - Rvaux.T * Baux * Rvaux
        problem.reset()
        problem.add_constraint(Maux >> 0)
        problem.add_constraint(T2aux >> 0)
        problem.add_constraint(Laux >> 0)
        solution = problem.solve(primals=False)
        print('Current shrinking factor: {:.4f}'.format(rate))
    return rate

def checkmatrix(nu_l, delta_l, W):
    n = np.size(W, 0)
    Wsl = W
    k1 = 0
    k2 = 1
    problem = picos.Problem()
    tau = picos.RealVariable('tau', (n, 1))
    index_matrix = picos.diag((-delta_l, -nu_l)) + np.diag(np.ones(1) / 2, -1) + np.diag(np.ones(1) / 2, 1)
    Plaux = picos.kron(index_matrix, np.identity(n))
    Rvaux = np.concatenate((np.concatenate((np.zeros((n, n)), Wsl,), axis=1), np.concatenate((np.identity(n), np.zeros((n, n))), axis=1)),axis=0)
    T2aux = picos.SymmetricVariable("T2aux", (n, n))
    Laux = picos.diag(picos.RealVariable('Laux', (n, 1)))
    Baux = (-2 * k1 * k2 * (T2aux + Laux) & (k1 + k2) * (T2aux + Laux)) // ((k1 + k2) * (T2aux + Laux) & -2 * (T2aux + Laux))
    Maux = Plaux - Rvaux.T * Baux * Rvaux
    problem.add_constraint(Maux >> 0)
    problem.add_constraint(T2aux >> 0)
    problem.add_constraint(Laux >> 0)

    solution = problem.solve(primals=False)

    if solution.claimedStatus == 'optimal':
        return 0
    else:
        return 1
