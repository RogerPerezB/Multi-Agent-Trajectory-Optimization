import argparse
import os
import yaml
from src.scripts.generate_scenarios import generate_random_crossing, save_custom_run, read_existing_params

# Parses command line arguments to run desired simulation
def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-Agent Trajectory Optimization CLI")
    
    parser.add_argument('num_planes', nargs='?', default=None)
    parser.add_argument('sim_size', nargs='?', default=None)
    parser.add_argument('num_obstacles', nargs='?', default=None)
    
    parser.add_argument('--config', default='configs/scenarios.yaml', help="Path to scenario config")
    parser.add_argument('--solver', default='gpu', choices=['cpu', 'gpu', 'hybrid'], help="Solver backend to use")
    
    return parser.parse_args()

# Checks for desisired scenario 
def handle_scenario_generation(args):
    if not any([args.num_planes, args.sim_size, args.num_obstacles]):
        return False

   # Checks for placeholder 
    defaults = (6, 100, 4)
    if os.path.exists(args.config):
        try:
            defaults = read_existing_params(args.config)
        except Exception:
            pass 

    n = int(args.num_planes) if args.num_planes and args.num_planes != '-' else defaults[0]
    s = int(args.sim_size) if args.sim_size and args.sim_size != '-' else defaults[1]
    o = int(args.num_obstacles) if args.num_obstacles and args.num_obstacles != '-' else defaults[2]

    print(f"Generating Scenario: {n} Planes | {s}m Box | {o} Obstacles")
    
    planes, obstacles = generate_random_crossing(n, s, o)
    save_custom_run(planes, obstacles, filename=args.config)
    return True

def get_scenario_name(config_path):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return list(data['scenarios'].keys())[0]