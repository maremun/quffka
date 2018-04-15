#   encoding: utf-8
#   download_mnist.py

from numpy import DataSource, frombuffer


IMAGE_URLS = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
              'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz']

LABEL_URLS = ['http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
              'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']


def download(temp_dir, *urls):
    for url in urls:
        ds = DataSource('../datasets/MNIST')
        with ds.open(url) as f:
            array = frombuffer(f.read(), dtype='uint8')
        yield array


def load_mnist(temp_dir='../datasets/MNIST'):
    arrays = download(temp_dir, *IMAGE_URLS, *LABEL_URLS)
    xtrain = next(arrays)[16:].reshape((60000, 784))
    xtest = next(arrays)[16:].reshape((10000, 784))
    ytrain = next(arrays)[8:]
    ytest = next(arrays)[8:]
    return (xtrain, ytrain, xtest, ytest)
