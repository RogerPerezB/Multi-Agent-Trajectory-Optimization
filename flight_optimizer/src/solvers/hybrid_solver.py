import numpy as np
import time
from src.solvers.base_solver import BaseSolver

try:
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
    HAS_LOCAL_TOOLS = True
except ImportError:
    HAS_LOCAL_TOOLS = False

class HybridSolver(BaseSolver):
    def __init__(self, config):
        self.samples = config.get('samples')
        self.buffer = config.get('obs_buffer')
        self.safe_dist = config.get('safe_dist')

    def generate_primitives(self, plane, steps):

        candidates = []
        
        dist = np.linalg.norm(plane.goal - plane.start)
        scale = dist * 0.5  
        
        straight = np.linspace(plane.start, plane.goal, steps + 1)
        candidates.append(straight)
        
        midpoint = (plane.start + plane.goal) / 2
        

        deviations = [
            [0, scale, 0],          # Left
            [0, -scale, 0],         # Right
            [0, 0, scale/2],        # Up 
            [0, 0, -scale/2],       # Down
            [0, scale, scale/2],
            [0, -scale, scale/2],
            [0, scale, -scale/2],
            [0, -scale, -scale/2],
            [0, scale * 1.5, 0],
            [0, -scale * 1.5, 0],
            [0, 0, scale], 
        ]
        
        for dev in deviations:
            control_point = midpoint + np.array(dev)
            path = np.zeros((steps + 1, 3))
            for i in range(steps + 1):
                t = i / steps
                # Quadratic Bezier
                path[i] = (1-t)**2 * plane.start + 2*(1-t)*t * control_point + t**2 * plane.goal
            candidates.append(path)
            
        return candidates

    def calculate_cost(self, trajectory, obstacles):
        # Smoothness
        acc_vecs = trajectory[2:] - 2*trajectory[1:-1] + trajectory[:-2]
        smoothness = np.sum(acc_vecs**2)
        
        # Obstacle Penalty
        obs_penalty = 0
        for pos in trajectory:
            for obs in obstacles:
                if np.linalg.norm(pos - obs.position) < (obs.radius + self.buffer):
                    obs_penalty += 1e9 
                    
        return smoothness + obs_penalty

    def check_interaction(self, traj_a, traj_b):
        dists = np.linalg.norm(traj_a - traj_b, axis=1)
        if np.any(dists < self.safe_dist):
            return True
        return False

    def solve(self, scenario):
        if not HAS_LOCAL_TOOLS:
            raise RuntimeError("Missing dependencies.")

        print(f"SOLVER MODE: MULTI-AGENT LOCAL ANNEALING")
        start_time = time.time()
        
        num_planes = len(scenario.planes)
        
        # Generate All Candidates
        all_candidates = []
        for p in scenario.planes:
            all_candidates.append(self.generate_primitives(p, scenario.steps))
            
        num_candidates = len(all_candidates[0])
        print(f"Total Binary Variables: {num_planes * num_candidates}")
        
        bqm = dimod.BinaryQuadraticModel('BINARY')
        constraint_penalty = 1e12 
        
        for p_idx in range(num_planes):
            for c_idx in range(num_candidates):
                traj = all_candidates[p_idx][c_idx]
                cost = self.calculate_cost(traj, scenario.obstacles)
                
                var_name = f'p{p_idx}_c{c_idx}'
                bqm.add_variable(var_name, cost - constraint_penalty)
                
           
            for c1 in range(num_candidates):
                for c2 in range(c1 + 1, num_candidates):
                    u = f'p{p_idx}_c{c1}'
                    v = f'p{p_idx}_c{c2}'
                    bqm.add_interaction(u, v, 2 * constraint_penalty)

        #  Inter-Agent Collision Penalties
        collision_count = 0
        
        for p1 in range(num_planes):
            for p2 in range(p1 + 1, num_planes):
                for c1 in range(num_candidates):
                    traj1 = all_candidates[p1][c1]
                    for c2 in range(num_candidates):
                        traj2 = all_candidates[p2][c2]
                        
                        if self.check_interaction(traj1, traj2):
                            u = f'p{p1}_c{c1}'
                            v = f'p{p2}_c{c2}'
                            
                            bqm.add_interaction(u, v, constraint_penalty)
                            collision_count += 1
                            
        print(f"Found {collision_count} conflicting pairs.")

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=self.samples)
        
        best_sample = sampleset.first.sample
        
        
        for p_idx in range(num_planes):
            selected = -1
            for c_idx in range(num_candidates):
                if best_sample[f'p{p_idx}_c{c_idx}'] == 1:
                    selected = c_idx
                    break
            
            if selected != -1:
                scenario.planes[p_idx].trajectory = all_candidates[p_idx][selected]
            else:
                print(f"Plane {p_idx}: Constraint violation (No single path).")
                scenario.planes[p_idx].trajectory = all_candidates[p_idx][0]

        end_time = time.time()
        print(f"Hybrid Optimization Complete. Time: {end_time - start_time:.4f} seconds.")