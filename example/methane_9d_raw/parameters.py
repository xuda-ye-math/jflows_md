"""Full-size methane Boltzmann generator parameters."""

MOLECULE = "methane"
FORMULA = "CH4"
DIMENSION = 9
TEMPERATURE_KELVIN = 300.0
SEED = 0

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

# Full methane configuration inherited from the successful 9D c50 baseline.
VALID_SIZE = 100000
POOL_SIZE = 0
BATCH_SIZE = 5000
STEPS_TOTAL = 500
LR = 1e-3
LR_WARMUP = 25
U_CLIP = 1e3
G_CLIP = 1e2

LADDER = 8
MC_DT = 1e-3
MC_STEPS_1 = 20    # leapfrog steps of each intermediate SMC level, MALA steps of the last and of the QT rows
MC_STEPS_2 = 100   # MALA steps of the QT pool temper and of the stage advance
MC_IMAGE_RADIUS = 3
CHUNKS = 32
SCREEN_FRACTION = 1e-4   # top fraction of the importance weights screened in training

MELT = 1.0
OPT_ALPHA = 1e-2
OPT_STEPS = 200
COEFF_THETA = 1.0   # weight of the mixture variation
COEFF_ALPHA = 0.5   # QT proportion of the mixture
COEFF_QT = 0.0      # partial importance resampling of the QT pool (0: none)
CHECKPOINT = True
INITIALIZE_FROM_IDENTITY = True

BG_PARAM = {
    "t_safe": 0.25,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.5,
    "tau_valid": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

# The monitor prints every tenth step; the driver saves ESS for every step.
MONITOR_EVERY = 10
