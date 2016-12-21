import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Cancer.keys(): \n{}".format(cancer.DESCR))
