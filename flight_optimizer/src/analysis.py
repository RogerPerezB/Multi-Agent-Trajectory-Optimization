import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import global_params as G

class SolverAnalyzer:
    def __init__(self, data_dir="data/results"):
        self.base_dir = data_dir
        self.logs_dir = os.path.join(data_dir, "logs")
        self.analysis_dir = os.path.join(data_dir, "analysis")
        
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        self.results = {}

    def load_data(self):
        csv_files = glob.glob(os.path.join(self.logs_dir, "*_data.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.logs_dir}")
            return

        for f in csv_files:
            solver_name = os.path.basename(f).replace("_data.csv", "").upper()
            self.results[solver_name] = pd.read_csv(f, keep_default_na=False) 
            print(f"Loaded data for {solver_name}...")

    def calculate_metrics(self, df, solver_name):
        metrics = {}
        
        agent_col_str = df['Collision_Agent'].astype(str)
        agent_colls = len(df[~agent_col_str.isin(['None', 'nan', ''])])
        
        obs_col_str = df['Collision_Obs'].astype(str)
        obs_colls = len(df[~obs_col_str.isin(['None', 'nan', ''])])
        clean_min = df['Min_Dist'].replace(9999.9, np.nan).min()   
        min_dist = round(clean_min, 4) if pd.notna(clean_min) else 0.0

        metrics['Collisions (Obs)'] = obs_colls
        metrics['Collisions (Agent)'] = agent_colls
        metrics['Min Separation [m]'] = round(min_dist, 4) if pd.notna(min_dist) else 0.0
        
        # Efficiency
        planes = df.groupby('Plane_ID')
        total_dist = 0
        total_displacement = 0
        
        for p_id, p_data in planes:
            coords = p_data[['X', 'Y', 'Z']].values
            diffs = np.diff(coords, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            total_dist += np.sum(dists)
            
            if len(coords) > 0:
                straight_line = np.linalg.norm(coords[-1] - coords[0])
                total_displacement += straight_line
            
        avg_path_len = total_dist / len(planes) if len(planes) > 0 else 0
        detour_ratio = total_dist / total_displacement if total_displacement > 0 else 1.0
        
        metrics['Avg Path Length [m]'] = round(avg_path_len, 2)
        metrics['Detour Ratio'] = round(detour_ratio, 3)
        
        # Smoothness
        total_smoothness = 0
        for p_id, p_data in planes:
            coords = p_data[['X', 'Y', 'Z']].values
            if len(coords) < 3: continue
            
            acc_vecs = coords[2:] - 2*coords[1:-1] + coords[:-2]
            acc_mags_sq = np.sum(acc_vecs**2, axis=1)
            total_smoothness += np.sum(acc_mags_sq)
            
        metrics['Smoothness Cost'] = round(total_smoothness / len(planes), 2) if len(planes) > 0 else 0
        
        return metrics

    def run_comparison(self):
        self.load_data()
        
        comparison_data = []
        for solver, df in self.results.items():
            m = self.calculate_metrics(df, solver)
            m['Solver'] = solver
            comparison_data.append(m)
            
        if not comparison_data:
            print("No data to analyze.")
            return
            
        summary = pd.DataFrame(comparison_data)
        summary = summary.sort_values(by='Solver')
        
        cols = ['Solver', 'Collisions (Agent)', 'Min Separation [m]', 
                'Avg Path Length [m]', 'Detour Ratio', 'Smoothness Cost']
        summary = summary[cols]
        
        print("\n--- COMPARATIVE ANALYSIS REPORT ---")
        print(summary.to_string(index=False))
        
        report_path = os.path.join(self.analysis_dir, "comparison_report.csv")
        summary.to_csv(report_path, index=False)
        print(f"\nReport saved to {report_path}")
        
        self.plot_separate_charts(summary)

    def plot_separate_charts(self, df):
        solvers = df['Solver']
        color_map = {'CPU': '#d62728', 'GPU': '#2ca02c', 'HYBRID': '#1f77b4'}
        colors = [color_map.get(s, 'gray') for s in solvers]

        # SAFETY 
        plt.figure(figsize=(8, 6))
        plt.bar(solvers, df['Min Separation [m]'], color=colors)
        plt.title('Safety Margin ', fontsize=14)
        plt.ylabel('Min Separation [m]', fontsize=12)
        plt.axhline(y=2.0, color='black', linestyle='--', label='Safety Limit (2.0m)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.analysis_dir, "metric_safety.png"), dpi=300)
        plt.close()

        # EFFICIENCY 
        plt.figure(figsize=(8, 6))
        plt.bar(solvers, df['Detour Ratio'], color=colors)
        plt.title('Path Efficiency ', fontsize=14)
        plt.ylabel('Detour Ratio', fontsize=12)
        plt.ylim(bottom=1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.analysis_dir, "metric_efficiency.png"), dpi=300)
        plt.close()

        # SMOOTHNESS CHART
        plt.figure(figsize=(8, 6))
        plt.bar(solvers, df['Smoothness Cost'], color=colors)
        plt.title('Control Effort', fontsize=14)
        plt.ylabel('Sum Squared Acceleration', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.analysis_dir, "metric_smoothness.png"), dpi=300)
        plt.close()
        print(f"Charts saved: {self.analysis_dir}")

if __name__ == "__main__":
    analyzer = SolverAnalyzer()
    analyzer.run_comparison()