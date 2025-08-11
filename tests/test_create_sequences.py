import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lstm65 import create_sequences


def test_create_sequences_sufficient_data():
    data = np.array([[1], [2], [3], [4], [5], [6]])
    look_back = 3
    X, Y = create_sequences(data, look_back)
    expected_X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    expected_Y = np.array([4, 5, 6])
    assert np.array_equal(X, expected_X)
    assert np.array_equal(Y, expected_Y)


@pytest.mark.parametrize(
    "data",
    [np.array([[1], [2], [3]]), np.array([[1], [2], [3], [4], [5]])],
)
def test_create_sequences_insufficient_data(data):
    look_back = 5
    X, Y = create_sequences(data, look_back)
    assert X.size == 0
    assert Y.size == 0
