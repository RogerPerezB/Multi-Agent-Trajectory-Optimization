import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cli_utils import parse_arguments, handle_scenario_generation, get_scenario_name
import src.global_params as G
from src.simulation import SimulationScenario

def main():
    args = parse_arguments()

    if handle_scenario_generation(args):
        print(f"Generated new scenario.")
    else:
        print("Using existing config.")

    try:
        scenario_name = get_scenario_name(args.config)
    except Exception as e:
        print(f"Error reading config: {e}")
        return
    
    # creates the configuration for the simulation
    config = {
        'rho': G.RHO,
        'alpha': G.ALPHA,
        'safe_dist': G.SAFE_DIST,
        'obs_buffer': G.OBS_BUFFER,
        'poly_order': G.POLY_ORDER,
        'samples': G.SAMPLES_HYBRID,
        'max_iter': G.MAX_ITER_GPU, 
        'device': 'cpu'
    }

    if args.solver == 'cpu':
        print(">>> Mode: CPU SEQUENTIAL Solver")
        from src.solvers.cpu_solver import CPUSequentialSolver
        config['max_iter'] = G.MAX_ITER_CPU
        solver = CPUSequentialSolver(config)
        
    elif args.solver == 'gpu':
        print(">>> Mode: GPU PARALLEL Solver")
        from src.solvers.gpu_solver import GPUSolver
        config['device'] = 'gpu'
        config['max_iter'] = G.MAX_ITER_GPU
        solver = GPUSolver(config)
        
    elif args.solver == 'hybrid':
        print(">>> Mode: HYBRID QUBO Solver")
        from src.solvers.hybrid_solver import HybridSolver
        solver = HybridSolver(config)

    # Run Simulation
    try:
        sim = SimulationScenario.from_yaml(args.config, scenario_name)
        print(f"Scenario: {len(sim.planes)} planes, {len(sim.obstacles)} obstacles.")
        
        solver.solve(sim)
        
        # Handling Output
        res_dir = os.path.join("data", "results")
        os.makedirs(os.path.join(res_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(res_dir, "logs"), exist_ok=True)
        
        csv_path = os.path.join(res_dir, "logs", f"{args.solver}_data.csv")
        img_path = os.path.join(res_dir, "plots", f"{args.solver}_result.png")
        
        sim.export_to_csv(csv_path, safe_dist=G.SAFE_DIST)
        
        # Title formatting
        title = f"({args.solver.upper()}) {len(sim.planes)} Airplanes | {int(sim.planes[0].goal[0])}m Dimensions | {len(sim.obstacles)} Obstacles"         
        sim.visualize(show=True, save_path=img_path, custom_title=title)

    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()