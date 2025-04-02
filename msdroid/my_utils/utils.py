def set_seed(seed = 666):
    import random
    import numpy as np
    import torch
    print('Set seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)