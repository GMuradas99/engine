import os
import torch
import torch.nn as nn

from tqdm import tqdm
from os.path import join
from typing import Callable
from torch.utils.data import DataLoader

from .utils import(
    loadCheckpoint,
    saveCheckpoint,
    savePredictionExample,
    loadCheckpointMetadata,
    saveCheckpointMetadata,
)

def trainOneEpoch(loader: DataLoader, model: nn.Module, optimizer: torch.optim, lossFunction: Callable, scaler, device: str, 
                  console: bool = True) -> float:
    """Trains for one epoch and returns the average loss of all instances.
    * :var:`lossFunction` takes the predictions as the first argument and the targets as the second.
    """
    if console:
        loop = tqdm(loader)
    else:
        loop = loader
    avgLoss, counter = 0, 0
    for _, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = lossFunction(predictions,targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        if console:
            loop.set_postfix(loss=loss.item())

        # Update loss
        avgLoss += loss.item()
        counter += 1

    return avgLoss/counter

def validate(loader: DataLoader, model: nn.Module, lossFunction: Callable, device: str, console: bool = True) -> float:
    """Returns the validation loss.
    * :var:`lossFunction` takes the predictions as the first argument and the targets as the second.
    """
    if console:
        print("Validating...")
        loop = tqdm(loader)
    else:
        loop = loader
    model.eval()
    avgLoss, counter = 0, 0
    with torch.no_grad():
        for _, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device=device)

            # Predicting
            predictions = model(data)
            loss = lossFunction(predictions,targets)

            # Update tqdm loop
            if console:
                loop.set_postfix(loss=loss.item())

            # Update loss
            avgLoss += loss.item()
            counter += 1
    return avgLoss/counter

def train(trainLoader: DataLoader, validationLoader: DataLoader, model: nn.Module, epochs: int,
          optimizer: torch.optim, lossFunction: Callable, scaler, device: str, loadLastCheckpoint: bool = False,
          checkpointPath: str = 'checkpoint', exampleSaverFunction: Callable = None, exampleSavingPath: str = 'examples',
          console: bool = True):
    """Trains the model for the selected amount of epochs.
    * boolean :var:`loadLastCheckpoint` indicates if the model is to be trained from scratch or continue training.
    """
    
    # Creating checkpoint and example folders (if they don't exist already)
    if not os.path.isdir(checkpointPath):
        os.mkdir(checkpointPath)
    if exampleSaverFunction is not None:
        if not os.path.isdir(exampleSavingPath):
            os.mkdir(exampleSavingPath)

    # Checkpoint Data
    lastEpoch = 0
    traiLossPerEpoch = []
    valLossPerEpoch = []

    # Loading previous checkpoint
    if loadLastCheckpoint:
        loadCheckpoint(checkpointPath, model, optimizer)
        metaData = loadCheckpointMetadata(checkpointPath)
        lastEpoch = metaData['lastEpoch']
        traiLossPerEpoch = metaData['traiLossPerEpoch']
        valLossPerEpoch = metaData['valLossPerEpoch']

    # Main loop
    for epoch in range(lastEpoch, epochs):
        if console:
            print(f'------   EPOCH {epoch+1}   ------')

        # Training one epoch
        trainLoss = trainOneEpoch(trainLoader, model, optimizer, lossFunction, scaler, device, console=console)
        traiLossPerEpoch.append(trainLoss)

        # Validating
        valLoss = validate(validationLoader, model, lossFunction, device, console=console)
        valLossPerEpoch.append(valLoss)

        # Saving Checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        saveCheckpoint(checkpoint, checkpointPath, console=console)
        lastEpoch = epoch+1
        checkpointData = {
            'lastEpoch': lastEpoch,
            'traiLossPerEpoch': traiLossPerEpoch,
            'valLossPerEpoch': valLossPerEpoch,
        }
        saveCheckpointMetadata(checkpointData, checkpointPath)

        # Save example
        if exampleSaverFunction is not None:
            if not os.path.isdir(join(exampleSavingPath, f'epoch_{epoch+1}')):
                os.mkdir(join(exampleSavingPath, f'epoch_{epoch+1}'))
            savePredictionExample(exampleSaverFunction, validationLoader, model, join(exampleSavingPath, f'epoch_{epoch+1}', 'example.png'), device)

    if console:
        print("Training complete.")