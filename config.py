
class Config(object):
    # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 9

    # train config
    EPOCHS = 25
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "./logs"
    BASE_LR = 1e-2
    BETA = (0.9, 0.999)
    EPS = 1e-08


    # step lr
    step_size = 10

    # cosine_annealing
    CYCLE_INTER = 10
    CYCLE_NUM = 3 
