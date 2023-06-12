import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from os.path import join
from typing import Callable
from torch.utils.data import DataLoader

def saveCheckpoint(state: dict, path: str, fileName: str="checkpoint.pth.tar", console: bool = True):
    """Saves the state of the model.
    * :var:`state` should be a dictionary containing the model state dictionary with the key :str:`state_dict` and the
    optimizer state dictionary with the key :str:`optimizer`.
    """
    if console:
        print("=> Saving checkpoint")
    torch.save(state, join(path, fileName))

def saveCheckpointMetadata(metadata: dict, path: str, fileName: str='checkpointData.json'):
    """Saves the checkpoint metadata.
    """
    f = open(join(path, fileName),'w')
    cpdJSON = json.dumps(metadata)
    f.write(cpdJSON)
    f.close()


def loadCheckpoint(path: str, model: nn.Module, optimizer: torch.optim = None, fileName: str="checkpoint.pth.tar", console: bool = True):
    """Loads the saved model.
    * :var:`state` should be a dictionary containing the model state dictionary with the key :str:`state_dict` and the
    optimizer state dictionary with the key :str:`optimizer`.
    """
    if console:
        print("=> Loading checkpoint")
    checkpoint = torch.load(join(path, fileName))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

def loadCheckpointMetadata(path: str, fileName: str="checkpointData.json") -> dict:
    """Returns a dictionary with the saved metadata.
    """
    f = open(join(path, fileName), 'r')
    data = json.load(f)
    f.close()
    return data


def savePredictionExample(saver: Callable,loader: DataLoader, model: nn.Module, path: str, device: str):
    """Calls the instance example saving function :func:`saver` should have 3 attributes in the next order:
    * :var:`inputs`: torch tensor with the inputs.
    * :var:`targets`: torch tensor with the targets.
    * :var:`predictions`: torch tensor with the predictions.
    * :vat:`path`: path to the location of the file, including the filename.
    """
    model.eval()
    for _, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)    
        saver(x, y, preds, path)
        break
    model.train()


def plotTrainingStats(data: dict, show: bool = True, savePath: str = None):
    """Plots a graph with the validation and training data
    """
    trainLoss = data['traiLossPerEpoch']
    valLoss = data['valLossPerEpoch']

    plt.plot(trainLoss)
    plt.plot(valLoss)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='upper right')

    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()