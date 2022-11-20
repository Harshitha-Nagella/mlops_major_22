import sys, os, math, random
from sklearn import datasets

#To find the utils.utils package
testdir = os.path.dirname(__file__)

srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from utils.utils import create_splits,preprocess,flatten_images
import utils
sys.path.insert(1, '/Users/harshi/Desktop/IITJ/Sem_4/mlops_major_22/utils/utils.py')




def test_equality():

    assert 1==1


def test_same_split():
    flattened_images, digits = flatten_images()
    rescale_factor = 1
    test_size = 0.2
    validation_size_from_test_size = 0.5
    rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
    random_state = 42

    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size,random_state) # 5 different splits
    X_train_1, X_test_1, X_validation_1, y_train_1, y_test_1, y_validation_1 = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size,random_state) # 5 different splits

    assert X_train == X_train_1
    assert X_test == X_test_1
    assert X_validation == X_validation_1

def test_differnt_split():
    flattened_images, digits = flatten_images()
    rescale_factor = 1
    test_size = 0.2
    validation_size_from_test_size = 0.5
    rescaled_images = preprocess(flattened_images, rescale_factor = rescale_factor)
    random_state = 0
    random_state_1 = 42
    X_train, X_test, X_validation, y_train, y_test, y_validation = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size,random_state) # 5 different splits
    X_train_1, X_test_1, X_validation_1, y_train_1, y_test_1, y_validation_1 = create_splits(rescaled_images, digits.target, test_size, validation_size_from_test_size,random_state_1) # 5 different splits
    assert X_train != X_train_1
    assert X_test != X_test_1
    assert X_validation != X_validation_1
