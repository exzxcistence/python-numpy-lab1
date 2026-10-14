from python_basics import count_vowels
from grader_contracts.python_basics import TextInput
from python_basics import count_vowels, has_unique_characters
from python_basics import count_vowels, has_unique_characters, count_one_bits
from grader_contracts.python_basics import PositiveIntegerInput
from python_basics import count_vowels, has_unique_characters, count_one_bits, multiplicative_persistence
from python_basics import mse
from grader_contracts.python_basics import VectorPairInput
from python_basics import prime_factorization
from python_basics import pyramid
from python_basics import is_balanced_number



def test_count_vowels():
    assert count_vowels(TextInput(value="hello")) == 2
    assert count_vowels(TextInput(value="Python")) == 1
    assert count_vowels(TextInput(value="AEIOU")) == 5
    assert count_vowels(TextInput(value="bcdfg")) == 0
    assert count_vowels(TextInput(value="")) == 0
    assert count_vowels(TextInput(value="aAaA")) == 4
    assert count_vowels(TextInput(value="education")) == 5



def test_has_unique_characters():
    assert has_unique_characters(TextInput(value="abcde")) is True
    assert has_unique_characters(TextInput(value="hello")) is False
    assert has_unique_characters(TextInput(value="python")) is True
    assert has_unique_characters(TextInput(value="aabb")) is False
    assert has_unique_characters(TextInput(value="")) is True
    assert has_unique_characters(TextInput(value="a")) is True
    assert has_unique_characters(TextInput(value="Aa")) is True
    assert has_unique_characters(TextInput(value="1123")) is False



def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(value=1)) == 1
    assert count_one_bits(PositiveIntegerInput(value=2)) == 1
    assert count_one_bits(PositiveIntegerInput(value=5)) == 2
    assert count_one_bits(PositiveIntegerInput(value=7)) == 3
    assert count_one_bits(PositiveIntegerInput(value=10)) == 2
    assert count_one_bits(PositiveIntegerInput(value=255)) == 8
    assert count_one_bits(PositiveIntegerInput(value=256)) == 1


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(value=39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(value=4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(value=999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(value=10)) == 1
    assert multiplicative_persistence(PositiveIntegerInput(value=25)) == 2
    assert multiplicative_persistence(PositiveIntegerInput(value=1)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(value=77)) == 4



def test_mse():
    assert abs(mse(VectorPairInput(
        predicted=[1, 2, 3],
        expected=[2, 4, 6]
    )) - 14 / 3) < 1e-9

    assert mse(VectorPairInput(
        predicted=[1, 2, 3],
        expected=[1, 2, 3]
    )) == 0.0

    assert mse(VectorPairInput(
        predicted=[5],
        expected=[2]
    )) == 9.0

    assert mse(VectorPairInput(
        predicted=[0, 0],
        expected=[3, 4]
    )) == 12.5

    assert mse(VectorPairInput(
        predicted=[-2, -4],
        expected=[2, 0]
    )) == 16.0


def test_prime_factorization():
    assert prime_factorization(
        PositiveIntegerInput(value=86240)
    ) == "(2**5)(5)(7**2)(11)"

    assert prime_factorization(
        PositiveIntegerInput(value=12)
    ) == "(2**2)(3)"

    assert prime_factorization(
        PositiveIntegerInput(value=7)
    ) == "(7)"

    assert prime_factorization(
        PositiveIntegerInput(value=100)
    ) == "(2**2)(5**2)"

    assert prime_factorization(
        PositiveIntegerInput(value=2)
    ) == "(2)"

    assert prime_factorization(
        PositiveIntegerInput(value=64)
    ) == "(2**6)"

    assert prime_factorization(
        PositiveIntegerInput(value=1)
    ) == ""



def test_pyramid():
    assert pyramid(PositiveIntegerInput(value=1)) == 1
    assert pyramid(PositiveIntegerInput(value=5)) == 2
    assert pyramid(PositiveIntegerInput(value=14)) == 3
    assert pyramid(PositiveIntegerInput(value=30)) == 4
    assert pyramid(PositiveIntegerInput(value=55)) == 5

    assert pyramid(
        PositiveIntegerInput(value=2)
    ) == "It is impossible"

    assert pyramid(
        PositiveIntegerInput(value=10)
    ) == "It is impossible"

    assert pyramid(
        PositiveIntegerInput(value=29)
    ) == "It is impossible"

    assert pyramid(
        PositiveIntegerInput(value=31)
    ) == "It is impossible"


def test_is_balanced_number():
    assert is_balanced_number(
        PositiveIntegerInput(value=1234006)
    ) is True

    assert is_balanced_number(
        PositiveIntegerInput(value=123456)
    ) is False

    assert is_balanced_number(
        PositiveIntegerInput(value=12321)
    ) is True

    assert is_balanced_number(
        PositiveIntegerInput(value=123321)
    ) is True

    assert is_balanced_number(
        PositiveIntegerInput(value=123)
    ) is False

    assert is_balanced_number(
        PositiveIntegerInput(value=121)
    ) is True

    assert is_balanced_number(
        PositiveIntegerInput(value=11)
    ) is True

    assert is_balanced_number(
        PositiveIntegerInput(value=1)
    ) is True

    assert is_balanced_number(
        PositiveIntegerInput(value=1001)
    ) is True

    assert is_balanced_number(
        PositiveIntegerInput(value=1002)
    ) is False
