# fashion-mnist-tf
Multi-class classification for Fashion-MNIST in tensorflow

Assignment 3 code for Deep Learning, CS60010.

MNIST data provides us very high accuracy with simple models, so we will be using fashion-MNIST.

The neural network has 3 hidden layers, with 50 epochs/iterations. Refer to [Report](Report.pdf) for more details.

## Performance.

* Training accuracy: 96.83% (Max accuracy in an iteration: 100%)

* Testing accuracy: 89.46%

* Loss as a function of iterations

 ![](https://imgur.com/EL465EB.jpg)

* Accuracy as a function of iterations

 ![](https://imgur.com/LIs5a8D.jpg)

## Layers

We apply Logistic Regression at every hidden layer. Here are the results:

* Layer 1: 88.87%
* Layer 2: 89.33%
* Layer 3: 89.46%

The first layer seems to provide enough accuracy, which proves further layers might not be needed.

## Usage

`python train.py --train`

Run training, save weights into `weights/` folder.

`python train.py --train iter=5`

Run training with specified number of iterations. Default iterations are 50.

`python train.py --test`

Load precomputed weights and report test accuracy.

`python train.py --layer=1`

Run Logistic Regression on hidden layer's output and report the accuracy. Allowed options : 1, 2, 3.

## Code structure

* [`data_loader`](data_loader.py) is used to load data from zip files in `data` folder.
* [`module`](module.py) defines the neural network parameters, and network related code.
* [`train`](train.py) handles input and states the model.


## License

The MIT License (MIT) 2018 - [Kaustubh Hiware](https://github.com/kaustubhhiware).
