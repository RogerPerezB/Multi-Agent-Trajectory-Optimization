# Airplane settings
RHO = 1000.0        # Penalty weight for breaking constraints
ALPHA = 1.0         # Reference weight to make planes choose the straight path

# Path settings
SAFE_DIST = 3.0     # Minimum distance to another plane in m
OBS_BUFFER = 2.0    # Extra buffer to add to obstacle radius to make it safe

# Solver specific variables
POLY_ORDER = 7      # Degree for the Bernstein polynomial
MAX_ITER_GPU = 150
MAX_ITER_CPU = 100 
SAMPLES_HYBRID = 1000  # Number of annealing reads