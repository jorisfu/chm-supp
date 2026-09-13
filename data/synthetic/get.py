import numpy as np

def get_dataset():
    ##
    ## CONFIG
    ##

    DATA_SHAPE = (100, 100) # (rows, cols)
    RNG = np.random.default_rng(1337)

    MEAN_UP = 1
    MEAN_DOWN = -1
    VAR_UP = 0.5
    VAR_DOWN = 0.5
    VAR_TOTAL = 4

    ##
    ## Generation
    ##

    mean_down = RNG.normal(MEAN_DOWN, VAR_DOWN, size=DATA_SHAPE[0] // 2)
    mean_up = RNG.normal(MEAN_UP, VAR_UP, size=DATA_SHAPE[0] // 2)
    mean_du = np.concatenate([mean_down, mean_up])
    mean_ud = np.concatenate([mean_up, mean_down])

    cov = np.eye(DATA_SHAPE[0], DATA_SHAPE[1]) * VAR_TOTAL
    d_du = RNG.multivariate_normal(mean_du, cov, DATA_SHAPE[0] // 2)
    d_ud = RNG.multivariate_normal(mean_ud, cov, DATA_SHAPE[0] // 2)
    d = np.concatenate([d_du, d_ud]).transpose()

    # Flat cluster assignments
    true_fcluster_cols = [0] * (DATA_SHAPE[1] // 2) + [1] * (DATA_SHAPE[1] // 2)

    col_gm = {"Column": {str(i): "C_DU" for i in range(DATA_SHAPE[0]//2)} | {str(i): "C_UD" for i in range(DATA_SHAPE[0]//2, DATA_SHAPE[0])}}

    return d, true_fcluster_cols, col_gm
