import random
import typing

from Nullity.NullityNetworkMini import relu


def check(test: typing.List[tuple]):
    i = 0
    correct = 0
    for num, correct_value in test:
        try:
            assert relu(num) == correct_value
            print(f"Тест {i + 1} пройден")
            correct += 1
        except AssertionError:
            print(f"Тест {i + 1} не пройден. relu({num}) != {correct_value}")
        i += 1
    return correct, len(test)


def generate_data(num: int):
    return [(a := random.uniform(-1000, 1000), a if a > 0 else 0) for _ in range(num)]


if __name__ == "__main__":
    tests = generate_data(10000)
    a = check(tests)
    print(f"Тестов пройдено {a[0]}/{a[1]}, {a[0] / a[1] * 100:.2f}%")
