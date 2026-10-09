import numpy as np
import time
from src.solvers.base_solver import BaseSolver
from src.solvers.trajectory_utils import PolynomialBasis

class CPUSequentialSolver(BaseSolver):

    def __init__(self, config):

        self.rho = config.get('rho')
        self.alpha = config.get('alpha')
        self.max_iter = config.get('max_iter')
        self.order = config.get('poly_order')
        
        self.safe_dist = config.get('safe_dist')
        self.obs_buffer = config.get('obs_buffer')

        self.matrices_built = False
        self.KKT_inv = None
        self.P = None
        self.n_vars = 0
        self.n_con = 0

    def _precompute_matrices(self, steps, dt):
        poly = PolynomialBasis(self.order, steps, dt)
        P, Q, A_eq = poly.build_min_accel_matrices()
        LHS = Q + self.rho * (P.T @ P)
        
        self.n_vars = LHS.shape[0]
        self.n_con = A_eq.shape[0]
        
        KKT = np.zeros((self.n_vars + self.n_con, self.n_vars + self.n_con))
        KKT[:self.n_vars, :self.n_vars] = LHS
        KKT[:self.n_vars, self.n_vars:] = A_eq.T
        KKT[self.n_vars:, :self.n_vars] = A_eq
        
        self.KKT_inv = np.linalg.inv(KKT)
        self.P = P
        self.matrices_built = True

    def solve(self, scenario):
        print(f"SOLVER MODE: CPU SEQUENTIAL")
        start_time = time.time()
        
        steps = scenario.steps
        dt = scenario.dt
        
        if not self.matrices_built:
            self._precompute_matrices(steps, dt)
            
        fixed_trajectories = []
        
        for p_idx, plane in enumerate(scenario.planes):
            current_traj = np.zeros((steps + 1, 3))
            for t in range(steps + 1):
                interp = t / steps
                current_traj[t, :] = (1 - interp) * plane.start + interp * plane.goal
                
            ref_traj = current_traj.copy()
            
            lambda_obs = np.zeros((len(scenario.obstacles), steps+1, 3))
            lambda_dynamic = np.zeros((len(fixed_trajectories), steps+1, 3))
            
            for k in range(self.max_iter):
                safe_targets = current_traj.copy()
                
                # Obstacle avoidance
                for o_idx, obs in enumerate(scenario.obstacles):
                    diff = current_traj - obs.position
                    dist = np.linalg.norm(diff, axis=1)
                    
                    # buffer from other objects
                    buffer = self.obs_buffer
                    mask = dist < (obs.radius + buffer)
                    
                    if np.any(mask):
                        unsafe_dists = dist[mask][:, None]
                        unsafe_dists[unsafe_dists < 1e-6] = 1e-6
                        
                        norm_diff = diff[mask] / unsafe_dists
                        # add buffer to radius of object
                        safe_pts = obs.position + norm_diff * (obs.radius + buffer)
                        
                        safe_targets[mask] = safe_pts
                        lambda_obs[o_idx][mask] += self.rho * (current_traj[mask] - safe_pts)
                
                # Dynamic Obstacle Avoidance 
                for prev_i, prev_traj in enumerate(fixed_trajectories):
                    diff = current_traj - prev_traj
                    dist = np.linalg.norm(diff, axis=1)
                    
                    mask = dist < self.safe_dist
                    if np.any(mask):
                        unsafe_dists = dist[mask][:, None]
                        unsafe_dists[unsafe_dists < 1e-6] = 1e-6
                        
                        direction = diff[mask] / unsafe_dists
                        correction = direction * (self.safe_dist - unsafe_dists)
                        
                        safe_targets[mask] += correction
                        lambda_dynamic[prev_i][mask] += self.rho * correction 
                
                # QP Update 
                for axis in range(3):
                    sum_lambda = np.sum(lambda_obs[:, :, axis], axis=0)
                    if len(fixed_trajectories) > 0:
                        sum_lambda += np.sum(lambda_dynamic[:, :, axis], axis=0)
                    
                    target_vec = self.rho * safe_targets[:, axis] - sum_lambda
                    target_vec += self.alpha * ref_traj[:, axis]
                    
                    linear_term = -(self.P.T @ target_vec)
                    
                    b_eq = np.zeros(4)
                    b_eq[0] = plane.start[axis]
                    b_eq[1] = plane.goal[axis]
                    b_eq[2] = 0.0 
                    b_eq[3] = 0.0 
                    
                    rhs = np.zeros(self.n_vars + self.n_con)
                    rhs[:self.n_vars] = linear_term
                    rhs[self.n_vars:] = b_eq
                    
                    sol = self.KKT_inv @ rhs
                    coeffs = sol[:self.n_vars]
                    current_traj[:, axis] = self.P @ coeffs

            # Adds plane path to trajectories, treating them as non dynamic
            fixed_trajectories.append(current_traj)
            plane.trajectory = current_traj
            
        end_time = time.time()
        print(f"Sequential Optimization Complete. Time: {end_time - start_time:.4f} seconds.")