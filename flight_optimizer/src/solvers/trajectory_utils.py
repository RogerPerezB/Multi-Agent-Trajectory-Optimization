import numpy as np
from scipy.special import comb

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class PolynomialBasis:
    def __init__(self, order, num_steps, dt):
        self.n = order
        self.num_steps = num_steps
        self.dt = dt
        self.time_points = np.linspace(0, num_steps * dt, num_steps + 1)
        
    def bernstein_poly(self, t, i, n):
        T = self.time_points[-1]
        return comb(n, i) * ((t/T)**i) * ((1 - t/T)**(n-i))

    def get_P_matrix(self):
        P = np.zeros((self.num_steps + 1, self.n + 1))
        for t_idx, t in enumerate(self.time_points):
            for i in range(self.n + 1):
                P[t_idx, i] = self.bernstein_poly(t, i, self.n)
        return P

    def get_derivative_matrices(self):
        P = self.get_P_matrix()
        # Velo
        D1 = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for i in range(1, self.num_steps):
            D1[i, i+1] = 1 / (2 * self.dt)
            D1[i, i-1] = -1 / (2 * self.dt)
        D1[0, 0] = -1/self.dt; D1[0, 1] = 1/self.dt
        D1[-1, -1] = 1/self.dt; D1[-1, -2] = -1/self.dt
        
        # Acceleration
        D2 = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for i in range(1, self.num_steps):
            D2[i, i+1] = 1 / (self.dt**2)
            D2[i, i] = -2 / (self.dt**2)
            D2[i, i-1] = 1 / (self.dt**2)
            
        return D1 @ P, D2 @ P

    def build_min_accel_matrices(self):
        P = self.get_P_matrix()
        P_vel, P_accel = self.get_derivative_matrices()
        Q = P_accel.T @ P_accel
        
        # Constraints: [Start_Pos, End_Pos, Start_Vel, End_Vel]
        A_eq = np.zeros((4, self.n + 1))
        A_eq[0, :] = P[0, :]
        A_eq[1, :] = P[-1, :]
        A_eq[2, :] = P_vel[0, :]
        A_eq[3, :] = P_vel[-1, :]
        
        return P, Q, A_eq

    def build_batch_matrices(self, num_planes, rho, use_gpu=False):
        # Small matrices
        P, Q, A_eq = self.build_min_accel_matrices()
        
        I_N = np.eye(num_planes)
        
        Q_batch = np.kron(I_N, Q)
        P_batch = np.kron(I_N, P)
        A_batch = np.kron(I_N, A_eq)
        
        LHS_single = Q + rho * (P.T @ P)
        LHS_batch = np.kron(I_N, LHS_single)
        
        n_vars = LHS_batch.shape[0]
        n_con = A_batch.shape[0]
        
        # KKT System
        KKT_batch = np.zeros((n_vars + n_con, n_vars + n_con))
        KKT_batch[:n_vars, :n_vars] = LHS_batch
        KKT_batch[:n_vars, n_vars:] = A_batch.T
        KKT_batch[n_vars:, :n_vars] = A_batch
        
        if use_gpu and HAS_GPU:
            return (cp.asarray(KKT_batch), cp.asarray(P_batch), n_vars, n_con)
        else:
            return (KKT_batch, P_batch, n_vars, n_con)