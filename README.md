# handwritten digit recognition

Training and testing of neural network model on MNIST data (kaggle)

---
# Project structure

```
handwritten-digit-recognition
├── data
│   └── digit-recognizer.zip                # zip with MNIST dataset
├── notebook
│   ├── __init__.py
│   └── handwritten_digit_recognition.ipynb # notebook for run in colab
├── src
│   ├── analytics
│   │   ├── __init__.py
│   │   └── statistic.py
│   ├── configuration
│   │   ├── __init__.py
│   │   ├── environment.yml                 # for environment installation
│   │   ├── literals.py
│   │   ├── requirements.txt                # for environment installation
│   │   └── config.json                     # main configuration
│   ├── models
│   │   ├── __init__.py
│   │   ├── logs
│   │   │   └── fit                         # dir with train tensorboard logs
│   │   └── cnn.h5                          # place fo save model
│   ├── tensorflow_models
│   │   ├── __init__.py
│   │   ├── architecture.py                # tensorflow model compile script
│   │   ├── cuda_config.py                 # tensorflow cuda configuration
│   │   └── training.py                    # tensorflow model train script
│   ├── dataset.py
│   ├── run_test.py                        # run tensorflow model testing
│   └── run_train.py                       # run tensorflow model training
└── README.md
```

---
# Installation

`cd/d handwritten-digit-recognition/src`

### For Cuda 
_Comment out the top part and uncomment the bottom part file (`environment.yml` or `requirements.txt`)_

### Using conda

`conda env create --file configuration/environment.yml`

### Using pip

`pip install -r configuration/requirements.txt`

---
# Run train

`python run_train.py`

---
# Run test

`python run_test.py`