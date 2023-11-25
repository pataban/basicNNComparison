

# general
VERBOSE = 3
TRAIN_SIZE = 1000
VAL_SIZE = 100
TEST_SIZE = 100
EPOCHS = 100
BATCH_SIZE = 512
PATIENCE = 5


class ManualMLP:
    NEURON_COUNT = [400, 10]  # 784,500,300,150,10       #[784,500,250,10]
    WEIGHTS_RANDOM_MAX = 0.1
    BIAS_RANDOM_MAX = 1.0
    BIAS_RANDOM_NEGATIVE = True
    LEARNING_SPEED = 0.001
    SUCCESS_MIN = 0.8
    SUCCESS_EARLY_STOP = -0.1
    BATCH_SIZE = 32
    PROGRESS_CHECK = 10


class MnistMLP:
    NEURON_COUNT = 128


class MnistCNN:
    FILTER = 1
    KERNEL = 3
    STRIDES = 1
    POOL_SIZE = 2
    POOL_STRIDE = 2
    DENSE_COUNT = 128


class Cifar10CNN:
    LABELS = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
              4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    FILTER = 9
    KERNEL = 3
    STRIDES = 1
    POOL_SIZE = 9
    POOL_STRIDE = 1
    DENSE_COUNT = 512
    """
    POOL_STRIDE=9
    DENSE_COUNT=150
    """
