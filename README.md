
# Nullity


### Installation


```bash
git clone https://github.com/Binobinos/Nullity.git
```
Установите зависимости

```bash
pip install -r requirements.txt
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from Nullity.NullityNetworkMini import Neuralnetwork, mse

# Инициализация
network = Neuralnetwork()
# Добавление входного слоя
network.add_layer(2, "RELU")
# Добавление скрытого слоя
network.add_layer(3, "RELU")
# Добавление выходного слоя
network.add_layer(2, "SIGMOID")
# Инициализация весов
network.calculating_weights()

input_vector = [1, 2]
output_vector = [2, 1]
# прямой проход
result = network.neuron(input_vector)

# Ошибка 
error = mse(result[-1], output_vector)
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests


### License

MIT