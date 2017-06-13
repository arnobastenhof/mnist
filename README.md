# mnist

Introduction
------------
A C++ implementation of a feed-forward neural network for training on the MNIST
database of handwritten digits. Intended as a private exercise to consolidating
some of the knowledge I gained from Andrew Ng's Machine Learning offering on
Coursera. Currently still a work in progress.

Requirements
------------
* Armadillo >=7.800.2
* Doxygen
* GCC or Clang (C++14-compatible)
* GNU Make

Installation
------------
Download and install version 7.800.2 or later of Armadillo, as per the
instructions on their website:
```
http://arma.sourceforge.net/download.html
```
On the developer's machine, running Lubuntu 16.10, the following steps were
taken to this end. On the command line, type
```
sudo apt-get install cmake libopenblas-dev libarpack++2-dev
```
Next, navigate to the directory where you would like to download the
Armadillo sources, and type
```
wget http://sourceforge.net/projects/arma/files/armadillo-7.800.2.tar.xz
tar xf armadillo-7.800.2.tar.xz
cd armadillo-7.800.2.tar.xz
cmake .
make
sudo make install
```
Next, install Doyxgen. Again, on Lubuntu 16.10, one may proceed as follows:
```
sudo apt-get install graphviz doxygen
```
The MNIST database will have to be downloaded separately. Create a directory
where you would like to store it, and from there type
```
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *
```
Clone or download the sources, e.g., through typing
```
wget https://github.com/arnobastenhof/mnist/archive/master.zip
unzip master.zip
```
From the project root, type
```
make
```

Usage
-----
After running `make`, an executable file called `main` will have been created
in `build/`. When executing it, supply the path to the MNIST database as a
command line parameter. The program will output its accuracy on the test set
for a number of user-supplied network parameters (e.g., the learning rate,
regularization parameter, ...).

TODO
----
Some of the changes still needed to be realized are as follows.
- Add Doxygen documentation
- Make number of hidden layers to use configurable.
- Switch Make for CMake
- Use Google Test for unit tests.
