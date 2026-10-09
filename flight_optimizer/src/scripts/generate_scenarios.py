import numpy as np
import yaml
import os
import random

# Generate custom scenario 
def generate_random_crossing(num_planes, room_size, num_obstacles, fixed_radius=None):
    planes = []
    obstacles = []
    
    # Generate Planes
    for i in range(num_planes):
        sy = random.uniform(0, room_size)
        sz = random.uniform(room_size * 0.2, room_size * 0.8)
        start = [0.0, sy, sz]
        
        gy = random.uniform(0, room_size)
        gz = random.uniform(room_size * 0.2, room_size * 0.8)
        goal = [float(room_size), gy, gz]
        
        planes.append({
            'id': i,
            'start': start,
            'goal': goal
        })

    # Generate Obstacles (middle of the box)
    min_obs_bound = room_size * 0.2
    max_obs_bound = room_size * 0.8
    
    for _ in range(num_obstacles):
        ox = random.uniform(min_obs_bound, max_obs_bound)
        oy = random.uniform(0, room_size)
        oz = random.uniform(0, room_size)
        
        if fixed_radius:
            rad = random.uniform(fixed_radius * 0.9, fixed_radius * 1.1)
        else:
            rad = random.uniform(20.0, 50.0)
        
        obstacles.append({
            'position': [ox, oy, oz],
            'radius': rad
        })

    return planes, obstacles

def read_existing_params(filename='configs/scenarios.yaml'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist")
        
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    
    scenario_name = list(data['scenarios'].keys())[0]
    scen = data['scenarios'][scenario_name]
    
    num_planes = len(scen['planes'])
    num_obstacles = len(scen.get('obstacles', []))
    
    if num_planes > 0:
        sim_size = int(scen['planes'][0]['goal'][0])
    else:
        sim_size = 100
        
    return num_planes, sim_size, num_obstacles

def save_custom_run(planes, obstacles, filename='configs/scenarios.yaml'):
    scenario_name = 'custom_run'
    if planes:
        room_size = planes[0]['goal'][0]
    else:
        room_size = 100.0

    raw_steps = int(room_size / 2.5)
    steps = max(20, min(raw_steps, 200))
    duration = steps * 0.5
    
    print(f"Custom Run: Size {room_size}m => {steps} Steps, {duration}s Duration")
    new_data = {
        'scenarios': {
            scenario_name: {
                'duration': float(duration),
                'steps': int(steps),
                'planes': planes,
                'obstacles': obstacles
            }
        }
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        yaml.dump(new_data, f, default_flow_style=None)
    
    return scenario_name