import cv2
import torch
import torchvision
import numpy as np

def saveExample(images: torch.tensor, target: torch.tensor, predictions: torch.tensor, path: str):
    """Saves 5 images from the validation set with the following color coding:
    * Green: True Positive
    * Red: False Negative
    * Blue: False Positive
    * Image Pixel: True Negative
    """
    imgsToShow = 5

    # Activation function
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    target = target.unsqueeze(1)

    # Getting grid images
    gridImages = torchvision.utils.make_grid(images[:imgsToShow]).cpu().numpy()
    gridImages = np.moveaxis(gridImages, 0, -1)
    gridTargets = torchvision.utils.make_grid(target[:imgsToShow]).cpu().numpy()
    gridTargets = np.moveaxis(gridTargets, 0, -1)
    gridPredictions = torchvision.utils.make_grid(predictions[:imgsToShow]).cpu().numpy()
    gridPredictions = np.moveaxis(gridPredictions, 0, -1)
    
    blankCanvass = np.zeros(gridImages.shape, dtype=np.uint8)

    img = gridImages.copy()
    img *= 255
    img = img.astype(np.uint8)
    oneDTarget = gridTargets.copy()
    oneDPrediction = gridPredictions.copy()
    oneDTarget = cv2.cvtColor(oneDTarget, cv2.COLOR_BGR2GRAY)
    oneDPrediction = cv2.cvtColor(oneDPrediction, cv2.COLOR_BGR2GRAY)

    maskTarget = oneDTarget == 1.
    maskPrediction = oneDPrediction == 1.

    blankCanvass[maskTarget & maskPrediction] = [0,255,0]
    blankCanvass[maskTarget & ~maskPrediction] = [255,0,0]
    blankCanvass[~maskTarget & maskPrediction] = [0,0,255]
    img[maskTarget & maskPrediction] = [0,255,0]
    img[maskTarget & ~maskPrediction] = [255,0,0]
    img[~maskTarget & maskPrediction] = [0,0,255]
    
    imV = cv2.vconcat([img, blankCanvass])

    cv2.imwrite(path, imV)