"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = "aeiou"
    count = 0

    for char in text.lower():
        if char in vowels:
            count += 1

    return count



def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))



def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    binary = bin(number)
    return binary.count("1")



def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while number >= 10:
        product = 1

        for digit in str(number):
            product *= int(digit)

        number = product
        steps += 1

    return steps




def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    total_error = 0

    for p, e in zip(predicted, expected):
        total_error += (p - e) ** 2

    return total_error / len(predicted)




def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2

    while divisor * divisor <= number:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power > 0:
            if power == 1:
                factors.append(f"({divisor})")
            else:
                factors.append(f"({divisor}**{power})")

        divisor += 1

    if number > 1:
        factors.append(f"({number})")

    return "".join(factors)




def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    level = 0

    while total < cube_count:
        level += 1
        total += level ** 2

    if total == cube_count:
        return level

    return "It is impossible"




def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    length = len(digits)

    middle = length // 2

    if length % 2 == 0:
        left = digits[:middle - 1]
        right = digits[middle + 1:]
    else:
        left = digits[:middle]
        right = digits[middle + 1:]

    left_sum = sum(int(digit) for digit in left)
    right_sum = sum(int(digit) for digit in right)

    return left_sum == right_sum
