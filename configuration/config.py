MNIST_DS_SAMPLE_IMAGE_SIZE = 28
MNIST_DS_SAMPLE_REDUCED_IMAGE_SIZE = 0

# mnist handwriting
MNIST_HW_LABEL_TRAINING_SET_PATH = "../input_raw_data/mnist/mnist_hw/train-labels-idx1-ubyte"
MNIST_HW_IMAGE_TRAINING_SET_PATH = "../input_raw_data/mnist/mnist_hw/train-images-idx3-ubyte.gz"

MNIST_HW_LABEL_TEST_SET_PATH = "../input_raw_data/mnist/mnist_hw/t10k-labels-idx1-ubyte"
MNIST_HW_IMAGE_TEST_SET_PATH = "../input_raw_data/mnist/mnist_hw/t10k-images-idx3-ubyte.gz"

# mnist fashion
MNIST_FASHION_LABEL_TRAINING_SET_PATH = "../input_raw_data/mnist/mnist_fashion/train-labels-idx1-ubyte"
MNIST_FASHION_IMAGE_TRAINING_SET_PATH = "../input_raw_data/mnist/mnist_fashion/train-images-idx3-ubyte.gz"

MNIST_FASHION_LABEL_TEST_SET_PATH = "../input_raw_data/mnist/mnist_fashion/t10k-labels-idx1-ubyte"
MNIST_FASHION_IMAGE_TEST_SET_PATH = "../input_raw_data/mnist/mnist_fashion/t10k-images-idx3-ubyte.gz"



class dataset_type:
    mnist_hw = 0
    mnist_fashion = 1


mnist_fashion_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                             'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

BATCH_SIZE = 32


PCA_ACCURACY_ARRAY = [90]
