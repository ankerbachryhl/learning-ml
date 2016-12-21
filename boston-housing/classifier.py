import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston

X, y = mglearn.datasets.load_extended_boston()
boston = load_boston()
