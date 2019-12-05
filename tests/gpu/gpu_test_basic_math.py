import unittest
import cupy as np  # !! CUPY !!
from dezero import Variable
from dezero.utils import check_backward


class TestAdd(unittest.TestCase):
    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 + x1
        res = y.data
        expected = np.array([2, 4, 6])
        self.assertTrue(np.array_equal(res, expected))

    def test_datatype(self):
        # np.float64ではなく、0次元のndarrayを返すかをテスト
        x = Variable(np.array(2.0))
        y = x ** 2
        self.assertFalse(np.isscalar(y))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x: x + y
        self.assertTrue(check_backward(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x + y
        self.assertTrue(check_backward(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda y: x + y
        self.assertTrue(check_backward(f, x))


class TestMul(unittest.TestCase):
    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 * x1
        res = y.data
        expected = np.array([1, 4, 9])
        self.assertTrue(np.array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x: x * y
        self.assertTrue(check_backward(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x * y
        self.assertTrue(check_backward(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda y: x * y
        self.assertTrue(check_backward(f, x))


class TestDiv(unittest.TestCase):
    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 / x1
        res = y.data
        expected = np.array([1, 1, 1])
        self.assertTrue(np.array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x: x / y
        self.assertTrue(check_backward(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x / y
        self.assertTrue(check_backward(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x / y
        self.assertTrue(check_backward(f, x))