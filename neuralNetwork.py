# TODO: save labeled training to own data and train as a whole



import ml.networks as networks


import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
# import torch.nn.functional as F

import torch.optim as optim

class Hyperparameters:
    learningRate: float = 0.001
    momentumCoefficient: float = 0.9
    minibatchSize: int = 1      # don't touch
    # hiddenLayerAmount: int = 1
    # hiddenLayerNodeAmount: int = 5
    epochAmount: int = 1
    labelAmount: int = 10
    trainingDataAmount: int = 25000
    # labelingTrainingDataAmount: int = 10000
    # attributeAmount: int = 32*32*3
    # lossFunction = torch.nn.MSELoss()
    lossFunction = nn.CrossEntropyLoss()
    confidenceRequirement: float = 0.95
    confidentNetworksNeeded: int = 5
    reTrain: bool = False



def main():
    print("start")

    trainLoader, labelTrainLoader,  testLoader = getLoaders()

    # nets = [networks.Net(), networks.NetDoubleDense()]

    nets = getAllNets()

    optimizers = getOptimizers(nets)

    if Hyperparameters.reTrain:
        nets = training(trainLoader, nets, optimizers)
        saveNets(nets)
    else:
        nets = loadNets(nets)

    # nets = labelingTraining(labelTrainLoader, nets, optimizers)

    testing(testLoader, nets)



def getLoaders():
    print("getting loaders")
    # data load and transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)

    trainSetIndices = list(range(0, Hyperparameters.trainingDataAmount))
    labelTrainSetIndices = list(range(Hyperparameters.trainingDataAmount, len(trainSet)))



    trainingSet = torch.utils.data.Subset(trainSet, trainSetIndices)

    labelingTrainSet = torch.utils.data.Subset(trainSet, labelTrainSetIndices)

    trainLoader = torch.utils.data.DataLoader(trainingSet,
                                              batch_size=Hyperparameters.minibatchSize,
                                              shuffle=True, num_workers=2)
    labelTrainLoader = torch.utils.data.DataLoader(labelingTrainSet,
                                                   batch_size=Hyperparameters.minibatchSize,
                                                   shuffle=True, num_workers=2)



    testSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=Hyperparameters.minibatchSize,
                                             shuffle=False, num_workers=2)

    return trainLoader, labelTrainLoader, testLoader

def getAllNets():
    print("getting all nets")
    nets = []
    nets.append(networks.Net())
    nets.append(networks.NetDoubleDense())
    nets.append(networks.NetHalfDense())
    nets.append(networks.NetDoubleWide())
    nets.append(networks.NetHalfWide())
    nets.append(networks.NetDoubleBoth())
    nets.append(networks.NetHalfBoth())
    nets.append(networks.NetDoubleWideHalfDense())
    nets.append(networks.NetHalfWideDoubleDense())
    return nets

def getOptimizers(nets):
    print("getting optimizers")
    optimizers = []
    for netIndex in range(len(nets)):
        optimizers.append(optim.SGD(nets[netIndex].parameters(),
                                    lr=Hyperparameters.learningRate,
                                    momentum=Hyperparameters.momentumCoefficient))
    return optimizers

def training(trainLoader, nets, optimizers):
    print("training")
    for epoch in range(Hyperparameters.epochAmount):  # loop over the dataset X times
        running_loss = 0.0

        for dataIndex, data in enumerate(trainLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            for netIndex, net in enumerate(nets):
                # zero the parameter gradients
                optimizers[netIndex].zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = Hyperparameters.lossFunction(outputs, labels)
                loss.backward()
                optimizers[netIndex].step()

                running_loss += loss.item()

            # print statistics
            if dataIndex % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {dataIndex + 1:5d}] loss: {running_loss / (2000*len(nets)):.3f}')
                running_loss = 0.0
    return nets

def saveNets(nets):
    print("saving nets")
    for netIndex, net in enumerate(nets):
        PATH = "./cifar_net" + str(netIndex) + ".pth"
        print(PATH)
        torch.save(net.state_dict(), PATH)

def loadNets(nets):
    print("loading nets")
    for netIndex, net in enumerate(nets):
        PATH = "./cifar_net" + str(netIndex) + ".pth"
        print(PATH)
        net.load_state_dict(torch.load(PATH))
    return nets


def labelingTraining(labelTrainLoader, nets, optimizers):
    print("labeling training starts")
    softmax = nn.Softmax(dim=1)
    for epoch in range(1):  # loop over the dataset X times
        confidentLabelAmount = 0

        for dataIndex, data in enumerate(labelTrainLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            confidentLabel = 0
            confidentLabelAgreed = 0

            outputs = []

            for netIndex, net in enumerate(nets):
                # zero the parameter gradients
                optimizers[netIndex].zero_grad()

                # forward
                outputs.append(net(inputs))
                # print(torch.max(softmax(outputs)))
                if torch.max(softmax(outputs[netIndex])) > Hyperparameters.confidenceRequirement:
                    # print("maximi ", softmax(outputs))
                    if confidentLabel == torch.argmax(outputs[netIndex]):      # at least 2 networks must agree
                        confidentLabel = torch.tensor([confidentLabel])  # making label list format
                        confidentLabelAgreed += 1
                        # print("networks agreed")
                        # print("net index ", netIndex)
                        # print("data index ", dataIndex)
                    else:
                        confidentLabel = torch.argmax(outputs[netIndex])
                        confidentLabelAgreed = 0
                    # print("confidernt label ", confidentLabel)
                    # print("real label ", labels)



            if confidentLabelAgreed >= Hyperparameters.confidentNetworksNeeded:
                # print("enough networks agreed")
                confidentLabelAmount += 1
                for netIndex, net in enumerate(nets):
                    loss = Hyperparameters.lossFunction(outputs[netIndex], confidentLabel)
                    loss.backward()
                    optimizers[netIndex].step()




        print("confident label amount = ", confidentLabelAmount)


    return nets


def testing(testLoader, nets):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for netIndex, net in enumerate(nets):
        # testing
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testLoader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %'
              + " networkIndex:" + str(netIndex))

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testLoader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
