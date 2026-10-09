import os
import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt
import src.global_params as G

class Airplane:
    def __init__(self, uid, start, goal):
        self.id = uid
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.trajectory = None 

class Obstacle:
    def __init__(self, pos, radius):
        self.position = np.array(pos, dtype=float)
        self.radius = float(radius)

class SimulationScenario:
    def __init__(self, name, duration, steps, planes, obstacles):
        self.name = name
        self.duration = duration
        self.steps = steps
        self.dt = duration / steps
        
        self.planes = [Airplane(p['id'], p['start'], p['goal']) for p in planes]
        self.obstacles = [Obstacle(o['position'], o['radius']) for o in obstacles]

    # Loads the scenario from pre-made shared configuration 
    @classmethod
    def from_yaml(cls, path, name):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if name not in data['scenarios']:
            raise ValueError(f"!!! Scenario missing !!!")
            
        conf = data['scenarios'][name]
        return cls(name, conf['duration'], conf['steps'], 
                   conf['planes'], conf.get('obstacles', []))

    # Exports the trajectory data to a CSV file, with other important variables
    def export_to_csv(self, filename, safe_dist=G.SAFE_DIST):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Time", "Plane_ID", "X", "Y", "Z", 
                             "Collision_Obs", "Collision_Agent", "Min_Dist"])
            
            for t in range(self.steps + 1):
                time = t * self.dt
                
                positions = {p.id: p.trajectory[t] for p in self.planes if p.trajectory is not None}

                for p in self.planes:
                    if p.trajectory is None: continue
                    
                    pos = p.trajectory[t]
                    obs_col = "None"
                    agent_col = "None"
                    min_dist = 9999.9

                    # Check obstacles
                    for o in self.obstacles:
                        if np.linalg.norm(pos - o.position) < o.radius:
                            obs_col = str(o.position)
                            break
                    
                    # Check aiplanes
                    for oid, opos in positions.items():
                        if oid == p.id: continue
                        d = np.linalg.norm(pos - opos)
                        if d < min_dist: min_dist = d
                        if d < safe_dist: agent_col = str(oid)
                    
                    writer.writerow([t, f"{time:.2f}", p.id, 
                                     f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                                     obs_col, agent_col, f"{min_dist:.4f}"])

    # creates the plot to visualize the simulation
    def visualize(self, show=True, save_path=None, custom_title=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for o in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = o.position[0] + o.radius * np.cos(u) * np.sin(v)
            y = o.position[1] + o.radius * np.sin(u) * np.sin(v)
            z = o.position[2] + o.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
            
        colors = plt.cm.jet(np.linspace(0, 1, len(self.planes)))
        for i, p in enumerate(self.planes):
            ax.scatter(*p.start, color=colors[i], marker='o')
            ax.scatter(*p.goal, color=colors[i], marker='x')
            
            if p.trajectory is not None:
                ax.plot(p.trajectory[:,0], p.trajectory[:,1], p.trajectory[:,2], 
                        color=colors[i], label=f"P{p.id}")

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(custom_title if custom_title else f"Scenario: {self.name}")
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        if show:
            plt.show()