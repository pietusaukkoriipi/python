import numpy

def run(layers):
    makePredictions(layers)
    return

def makePredictions(layers):
    for layerIndex, layer in enumerate(layers[:-1]):    # skip last layer
        predictNextLayer(layer)
        activateNodes(layers[layerIndex + 1])
    return


def predictNextLayer(layer):
    for node in layer.nodes:
        for weight in node.weights:
            weight.toNode.value += node.value * weight.value
    return


def activateNodes(layer):
    for node in layer.nodes[1:]:    # skip bias
        node.value = activate(node.value)
    return

def activate(predictedValue):
    return max(0, predictedValue)     # ReLU
    # return 1 / (1 + numpy.exp(-predictedValue))     # sigmoid
    # if predictedValue > 0:      # ELU
    #    return predictedValue
    # else:
    #    return
