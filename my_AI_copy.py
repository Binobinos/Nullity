from Nullity.NullityNetworkMini import *


def is_prime(number: int):
    if number <= 1:
        return False
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True


def generate_data(num_samples, start):
    data = []
    for number in range(start, start + num_samples + 1):
        # Normalize the input to [0, 1]
        normalized_number = (number - start) / (start + num_samples)
        input_vector = [normalized_number, number % 2, number % 3]  # [нормализованное число, четность, 0]
        is_even = int(number % 2 == 0)
        is_prime_num = int(is_prime(number))
        output = [is_even, 1 - is_even, is_prime_num]
        data.append((input_vector, output))
    return data


def train(network: Neuralnetwork,
          epochs: int = 100,
          learning_rate: float = 0.01,
          num_samples: int = 100
          ):
    # Инициализация весов и смещений
    epochs_data = {}
    start = 0
    data = generate_data(num_samples, start)
    last_error = 0
    epoch_size = 50
    for epoch in range(epochs):
        total_error = 0.0
        for input_vec, target in data:
            # Прямой проход
            output = network.neuron(input_vec)

            # Расчет ошибки
            total_error += mse(output[0], target)
            print(output)
            output_error = [output[0][i] - target[i] for i in range(len(output[0]))]
            print(output_error)
            # Градиенты для weights_2
            for i in range(3):  # Выходные нейроны
                for j in range(3):  # Скрытые нейроны
                    grad = output_error[i] * sigmoid(output[i], types="derivative") * hidden[j]
                    weights_2[i][j] -= learning_rate * grad

            # Вычисляем ошибку скрытого слоя
            hidden_error = [0.0] * 3
            for i in range(3):
                hidden_error[i] = sum(
                    output_error[k] * sigmoid(output[k], types="derivative") * weights_2[k][i]
                    for k in range(3)
                )

            # Градиенты для weights_1
            for i in range(3):  # Скрытые нейроны
                for j in range(3):  # Входные нейроны
                    grad = hidden_error[i] * relu(hidden[i], types="derivative") * input_vec[j]
                    weights_1[i][j] -= learning_rate * grad  # для weights_1
            biases[0] -= learning_rate * sum(hidden_error)  # для скрытого слоя
            biases[1] -= learning_rate * sum(output_error)  # для выходного слоя"""

        epochs_data[str(epoch + 1)] = total_error / len(data)
        if epoch % epoch_size == 0:
            learning_rate *= 0.2
            num_samples *= 1.5
            start += int(num_samples)
            data = generate_data(int(num_samples), start)
            error = total_error / len(data)
            print(
                f"\rЭпоха {epoch}, Средняя ошибка: {error:.4f}, Размер данных {len(data)}, Общая ошибка {total_error}"
                f", Начальная точка {start}, Разница ошибок в {epoch_size} эпохах "
                f"{(round(last_error, 6) - round(error, 6)):.6f}")
            last_error = error

    return network, epochs_data


if __name__ == "__main__":
    network = Neuralnetwork()
    arh = [3, 1, 3]
    for neiron in arh:
        network.add_layer(neiron, "RELU")
    network.calculating_weights()
    network, epochs_data = train(network,
                                epochs=1,
                                learning_rate=0.01,
                                num_samples=1)
