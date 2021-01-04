"""
Copyright (c) 2016-2020 The scikit-optimize developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import pytest
import numbers
import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises_regex
from skopt.space import LogN, Normalize
from skopt.space.transformers import StringEncoder, LabelEncoder, Identity


@pytest.mark.fast_test
def test_logn2_integer():

    transformer = LogN(2)
    for X in range(2, 31):
        X_orig = transformer.inverse_transform(transformer.transform(X))
        assert_array_equal(int(np.round(X_orig)), X)


@pytest.mark.fast_test
def test_logn10_integer():

    transformer = LogN(2)
    for X in range(2, 31):
        X_orig = transformer.inverse_transform(transformer.transform(X))
        assert_array_equal(int(np.round(X_orig)), X)


@pytest.mark.fast_test
def test_integer_encoder():

    transformer = LabelEncoder()
    X = [1, 5, 9]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)

    transformer = LabelEncoder(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)

    X = ["a", "b", "c"]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)

    transformer = LabelEncoder(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)


@pytest.mark.fast_test
def test_string_encoder():

    transformer = StringEncoder()
    X = [1, 5, 9]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), ['1', '5', '9'])
    assert_array_equal(transformer.inverse_transform(['1', '5', '9']), X)

    X = ['a', True, 1]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), ['a', 'True', '1'])
    assert_array_equal(transformer.inverse_transform(['a', 'True', '1']), X)

    X = ["a", "b", "c"]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)


@pytest.mark.fast_test
def test_identity_encoder():

    transformer = Identity()
    X = [1, 5, 9, 9, 5, 1]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)

    X = ['a', True, 1, 'a', True, 1]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)

    X = ["a", "b", "c", "a", "b", "c"]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)


@pytest.mark.fast_test
def test_normalize_integer():
    transformer = Normalize(1, 20, is_int=True)
    assert transformer.transform(19.8) == 1.0
    assert transformer.transform(20.2) == 1.0
    assert transformer.transform(1.2) == 0.0
    assert transformer.transform(0.9) == 0.0
    assert_raises(ValueError, transformer.transform, 20.6)
    assert_raises(ValueError, transformer.transform, 0.4)

    assert transformer.inverse_transform(0.99) == 20
    assert transformer.inverse_transform(0.01) == 1
    assert_raises(ValueError, transformer.inverse_transform, 1. + 1e-6)
    assert_raises(ValueError, transformer.transform, 0. - 1e-6)
    transformer = Normalize(0, 20, is_int=True)
    assert transformer.transform(-0.2) == 0.0
    assert_raises(ValueError, transformer.transform, -0.6)


@pytest.mark.fast_test
def test_normalize():
    transformer = Normalize(1, 20, is_int=False)
    assert transformer.transform(20.) == 1.0
    assert transformer.transform(1.) == 0.0
    assert_raises(ValueError, transformer.transform, 20. + 1e-6)
    assert_raises(ValueError, transformer.transform, 1.0 - 1e-6)
    assert_raises(ValueError, transformer.inverse_transform, 1. + 1e-6)
    assert_raises(ValueError, transformer.transform, 0. - 1e-6)