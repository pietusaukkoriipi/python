import random

import numpy


class Layer:
    def __init__(self, nodeAmount, previousLayer):
        self.nodes = self.makeNodes(nodeAmount)
        if type(previousLayer) == type(self):
            self.weights = self.makeWeights(previousLayer)
        else:
            self.weights = []

    def makeNodes(self, nodeAmount):
        nodes = []
        nodes.append(Node(0.01))   # bias
        for _ in range(nodeAmount):
            nodes.append(Node(0))
        return nodes

    def makeWeights(self, previousLayer):
        weights = []
        for previousLayerNode in previousLayer.nodes:
            tempWeights = []
            for node in self.nodes[1:]:
                tempWeights.append(Weight(previousLayerNode, node))  # skipping bias
            weights.append(tempWeights)
        return weights

class Node:
    def __init__(self, nodeValue):
        self.value = nodeValue
        self.weights = set()
        self.error = 0

    def weightChance(self, layer, nextLayer, nodeIndex):
        if numpy.random.randn() > 0:
            self.weights.add(self.giveWeight(layer, nextLayer, nodeIndex))
        return

    def giveWeight(self, layer, nextLayer, nodeIndex):
        return random.choice(nextLayer.weights[nodeIndex])


class Weight:
    def __init__(self, fromNode, toNode):
        self.value = 0.01 * numpy.random.randn()
        self.fromNode = fromNode
        self.toNode = toNode
        self.change = 0
