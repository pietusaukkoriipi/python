from dataclasses import dataclass

import numpy
import math
import time

import ml.network as network
import ml.forwardPropagation as forwardPropagation

@dataclass
class Hyperparameters:
    learningRate: float = 0.001
    momentumCoefficient: float = 0.9
    minibatchSize: int = 8


def main():
    hyperparameters = Hyperparameters()

    data = [[3, 1.5],
            [2, 1],
            [4, 1.5],
            [3, 1],
            [3.5, 0.5],
            [2, 0.5],
            [5.5, 1],
            [1, 1]]
    classifications = [[1], [0], [1], [0], [1], [0], [1], [0]]

    hiddenLayerAmount = 1
    hiddenLayerNodeAmount = 5

    layers = []

    layers.append(network.Layer(2, None))           # input layer, amount of inputs
    for _ in range(hiddenLayerAmount):
        layers.append(network.Layer(hiddenLayerNodeAmount, layers[-1]))
    layers.append(network.Layer(1, layers[-1]))     # output layer, amount of outputs, previous layer

    errors = []

    data = numpy.array(data)
    print(data)
    data = normalizeAndScale(data)
    print(data)

    for _ in range(3000):
        dataPointIndex = numpy.random.randint(len(data))
        point = data[dataPointIndex]
        targets = classifications[dataPointIndex]

        assignWeights(layers)
        assignInputs(layers[0], point)

        forwardPropagation.run(layers)

        giveStartingErrors(layers[-1], targets)
        backpropagate(layers, hyperparameters)

        error = layers[-1].nodes[-1].value - targets[-1]
        errors.append(abs(error))
        if 0:
            print("data point and classification: ", point, targets)
            print("activated output node: ", layers[-1].nodes[1].value)
            print("squared error in answers: ", error ** 2)
        if len(errors) % 100 == 0:
            print("average error in the last 100: ", sum(errors[-100:]) / 100)

        resetNodes(layers)


def normalizeAndScale(dataset):
    dataset = dataset.transpose()
    for parameterIndex, parameters in enumerate(dataset):
        dataset[parameterIndex] = parameters - parameters.mean()
        dataset[parameterIndex] = parameters / parameters.max()
    dataset = dataset.transpose()
    return dataset


def assignWeights(layers):
    for layerIndex, layer in enumerate(layers[:-1]):
        for nodeIndex, node in enumerate(layer.nodes):
            node.weightChance(layer, layers[layerIndex + 1], nodeIndex)
    return


def assignInputs(inputLayer, point):
    for nodeIndex, node in enumerate(inputLayer.nodes[1:]):
        node.value = point[nodeIndex]
    return



def giveStartingErrors(layer, targets):
    for nodeIndex, node in enumerate(layer.nodes[1:]):      # skip bias
        node.error = 2 * (node.value - targets[nodeIndex])
    return


def backpropagate(layers, hyperparameters):
    giveErrors(layers)
    modifyWeights(layers, hyperparameters)
    return

def giveErrors(layers):
    for layer in layers[-2::-1]:     # go from second last to second layer
        for node in layer.nodes:
            node.error = 0
            weightSum = 0   # turha
            """for weight in node.weights:
                weightSum += abs(weight.value)"""
            for weight in node.weights:
                node.error = giveError(node, weight, weightSum)
    return
def giveError(node, weight, weightSum):
    return node.error + (weight.toNode.error * weight.value)

def modifyWeights(layers, hyperparameters):
    for layer in layers[-2::-1]:  # go from second last to first layer
        for node in layer.nodes:
            for weight in node.weights:
                weight.value = modifyWeight(node, weight, hyperparameters)
    return
def modifyWeight(node, weight, hyperparameters):
    return weight.value + weightChange(node, weight, hyperparameters)
def weightChange(node, weight, hyperparameters):
    weight.change = (weight.change * hyperparameters.momentumCoefficient) - ((node.value * getPredictionDerivative(weight.toNode.value) * (2 * weight.toNode.error)) * hyperparameters.learningRate)
    return weight.change
def getPredictionDerivative(activatedPrediction):
    # return activatedPrediction * (1-activatedPrediction)    # sigmoid
    if activatedPrediction > 0:     # ReLU
        return 1
    else:
        return 0
    """if activatedPrediction > 0:     # ELU
        return 1
    else:
        return 0.1 + activatedPrediction"""



def resetNodes(layers):
    for layer in layers:
        for node in layer.nodes[1:]:
            node.value = 0
    return





