import numpy as np
import time
from src.solvers.base_solver import BaseSolver
from src.solvers.trajectory_utils import PolynomialBasis, HAS_GPU

if HAS_GPU:
    import cupy as cp
else:
    import numpy as cp 
class GPUSolver(BaseSolver):

    def __init__(self, config):
        self.rho = config.get('rho')
        self.alpha = config.get('alpha')
        self.max_iter = config.get('max_iter')
        self.order = config.get('poly_order')
        
        self.safe_dist = config.get('safe_dist')
        self.obs_buffer = config.get('obs_buffer')              
        
        self.device_mode = config.get('device', 'cpu')
        
        if self.device_mode == 'gpu' and HAS_GPU:
            self.use_gpu = True
        else:
            self.use_gpu = False
            if self.device_mode == 'gpu':
                print("GPU requested but not found. Falling back to CPU.")

    def solve(self, scenario):
        if self.use_gpu:
            print(f"SOLVER MODE: GPU JOINT OPTIMIZATION")
            xp = cp
        else:
            print(f"SOLVER MODE: JOINT OPTIMIZATION (NumPy Fallback)")
            xp = np
            
        start_time = time.time()

        # Setup
        num_planes = len(scenario.planes)
        steps = scenario.steps
        dt = scenario.dt
        poly = PolynomialBasis(self.order, steps, dt)
        
        # Build Matrices 
        total_penalty = self.rho + self.alpha
        KKT, P_batch, n_vars, n_con = poly.build_batch_matrices(num_planes, total_penalty, self.use_gpu)
        
        # Inversion
        KKT_inv = xp.linalg.inv(KKT)
        
        # Prepare Trajectories
        ref_traj_cpu = np.zeros((num_planes, steps + 1, 3))
        for i, plane in enumerate(scenario.planes):
            for t in range(steps + 1):
                interp = t / steps
                ref_traj_cpu[i, t, :] = (1 - interp) * plane.start + interp * plane.goal

        # Upload to GPU
        ref_traj = xp.asarray(ref_traj_cpu) if self.use_gpu else ref_traj_cpu
        current_traj = xp.copy(ref_traj)
        
        lambda_obs = xp.zeros((num_planes, len(scenario.obstacles), steps+1, 3))
        
        # Pre-load obstacle data to GPU
        obs_positions = xp.array([o.position for o in scenario.obstacles])
        obs_radii = xp.array([o.radius for o in scenario.obstacles])

        # Optimization Loop
        for k in range(self.max_iter):
            
            safe_targets = xp.copy(current_traj)
            for o_idx in range(len(scenario.obstacles)):
                pos = obs_positions[o_idx]
                rad = obs_radii[o_idx]
                
                # Check collision distance
                diff = current_traj - pos
                dist = xp.linalg.norm(diff, axis=2) 
                mask = dist < (rad + self.obs_buffer)
                
                if xp.any(mask):
                    unsafe_dist = dist[mask][:, None]
                    unsafe_dist[unsafe_dist < 1e-6] = 1e-6 
                    
                    # Project to surface (Radius + Buffer)
                    norm_diff = diff[mask] / unsafe_dist
                    safe_pts = pos + norm_diff * (rad + self.obs_buffer)
                    
                    safe_targets[mask] = safe_pts
                    lambda_obs[:, o_idx][mask] += self.rho * (current_traj[mask] - safe_pts)


            diff_matrix = current_traj[:, None, :, :] - current_traj[None, :, :, :]
            dist_matrix = xp.linalg.norm(diff_matrix, axis=3)
            
            conflict_mask = (dist_matrix < self.safe_dist) & (dist_matrix > 1e-6)
            
            if xp.any(conflict_mask):

                norm_diffs = xp.zeros_like(diff_matrix)
                valid_mask = dist_matrix > 1e-6
                norm_diffs[valid_mask] = diff_matrix[valid_mask] / dist_matrix[valid_mask][:, None]
                
                overlaps = xp.maximum(0, self.safe_dist - dist_matrix)
                
                identity_mask = xp.eye(num_planes, dtype=bool)[:, :, None]
                overlaps[identity_mask * xp.ones((1,1,steps+1), dtype=bool)] = 0
                

                total_push = xp.sum((overlaps[:, :, :, None] * 0.5) * norm_diffs, axis=1)
                
                
                safe_targets += total_push              # Apply Push
                
            
            for axis in range(3):

                sum_lambda_obs = xp.sum(lambda_obs[:, :, :, axis], axis=1)
                
                target_vec = self.rho * safe_targets[:, :, axis] - sum_lambda_obs
                target_vec += self.alpha * ref_traj[:, :, axis]
                
                flat_target = target_vec.reshape(-1)
                linear_term = -(P_batch.T @ flat_target)
                
                # Boundary Conditions
                b_eq_batch = xp.zeros(num_planes * 4)
                for i in range(num_planes):
                    b_eq_batch[i*4]     = float(scenario.planes[i].start[axis])
                    b_eq_batch[i*4+1]   = float(scenario.planes[i].goal[axis])
                    b_eq_batch[i*4+2]   = 0.0 
                    b_eq_batch[i*4+3]   = 0.0
                
                # GPU Matrix-Vector Multiplication
                rhs = xp.zeros(n_vars + n_con)
                rhs[:n_vars] = linear_term
                rhs[n_vars:] = b_eq_batch
                
                sol = KKT_inv @ rhs
                
                #  Extract Results
                coeffs_batch = sol[:n_vars]
                traj_axis = P_batch @ coeffs_batch
                current_traj[:, :, axis] = traj_axis.reshape(num_planes, steps+1)

        # Retrieve Results
        if self.use_gpu:
            final_traj = cp.asnumpy(current_traj) 
        else:
            final_traj = current_traj

        for i, plane in enumerate(scenario.planes):
            plane.trajectory = final_traj[i]
            
        end_time = time.time()
        print(f"Joint Optimization Complete. Time: {end_time - start_time:.4f} seconds.")