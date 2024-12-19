"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

import os
import random
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import utils
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


def seed_everything(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True


class Lab09_op:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything()

    def load_data_ex1(self, dpath: Union[Path, str, None] = None):
        data = []
        if dpath is None:
            dpath = Path("/home/woody/mrvl/mrvl100h/Lab09/data/dti")

        for fname in (dpath).iterdir():
            if fname.suffix == ".csv":
                data.append(pd.read_csv(fname, header=None).values)

        train_data = np.concatenate(data[:3], axis=0)
        test_data = data[-1]
        return train_data, test_data

    def load_data_ex2(self, dpath: Union[Path, str, None] = None):
        entire_data = []
        if dpath is None:
            dpath = Path("/home/woody/mrvl/mrvl100h/Lab09/data/recon_classification")

        for data_split in ["train", "val"]:
            images, labels = [], []
            for root, dir, files in os.walk(dpath / data_split):
                root = Path(root)
                for file in files:
                    if file.endswith(".png"):
                        image = cv2.imread(str(root / file))
                        image = utils.rgb2gray(image)
                        [nR, nC] = image.shape
                        image = image.reshape(1, nR, nC)
                        label = root.stem
                        images.append(image)
                        labels.append(label)
            entire_data.append(images)
            entire_data.append(labels)

        x_train, y_train, x_val, y_val = entire_data

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.fit_transform(y_val)

        return np.array(x_train), y_train, np.array(x_val), y_val

    def ex1_extract_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from the data to generate input data and label data.
        The input data contains features to be used for training
        The label data contains lable information for each sample.

        Args:
            data (np.ndarray): Data to extract features from.

        Returns:
            input_data (np.ndarray):    Input data for the model. [nSamples, nFeatures]
            labels_data (np.ndarray):   Label data. [0 (Thalamus), 1 (CC), 2 (Cortical WM)]

        """
        # Your code here ...

        input_data = None
        labels_data = None

        return input_data, labels_data

    def ex1_get_nSamples(self, input_data: np.ndarray) -> int:
        """
        Get the number of samples for each class.

        Args:
            input_data (np.ndarray): Input data. [nSamples, nFeatures]

        Returns:
            nSamples (np.ndarray): Number of samples in input_data

        """
        # Your code here ...

        nSamples = None
        return nSamples

    def ex1_get_nFeatures(self, input_data: np.ndarray) -> int:
        """
        Get the number of features.

        Args:
            input_data (np.ndarray): Input data. [nSamples, nFeatures]

        Returns:
            nFeatures (np.ndarray): Number of features in input_data

        """
        # Your code here ...

        nFeatures = None
        return nFeatures

    def ex1_get_nLabels(self, labels: np.ndarray) -> int:
        """
        Get the number of labels.

        Args:
            labels (np.ndarray): Label data. [nSamples]

        Returns:
            nLabels (np.ndarray): Number of labels in labels

        """
        # Your code here ...

        nLabels = None
        return nLabels

    def ex1_normalization(self, train_input: np.ndarray, test_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize each feature of train and test datasets using the maximum value of the train dataset.

        Args:
            train_input (np.ndarray):   Training input data. [nSamples, nFeatures]
            test_input (np.ndarray):    Test input data. [nSamples, nFeatures]

        Returns:
            train_input_norm (np.ndarray):   Normalized training input data. [nSamples, nFeatures]
            test_input_norm (np.ndarray):    Normalized test input data. [nSamples, nFeatures]

        """

        # Your code here ...

        return train_input_norm, test_input_norm

    def ex1_split_train_val(
            self, train_input: np.ndarray, train_labels: np.ndarray, train_ratio: float = 0.8, **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:        
        """
        Split the training data into training and validation sets.

        Args:
            train_input (np.ndarray):   Training input data. [nSamples, nFeatures]
            train_labels (np.ndarray):  Training label data. [nSamples]
            train_ratio (float):          Ratio of the training set.

        Returns:
            x_train (np.ndarray):       Training input data. [nTrainSamples, nFeatures]
            y_train (np.ndarray):       Training label data. [nTrainSamples]
            x_val (np.ndarray):         Validation input data. [nValSamples, nFeatures]
            y_val (np.ndarray):         Validation label data. [nValSamples]

        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        ex1_get_nSamples = kwargs.get("ex1_get_nSamples", self.ex1_get_nSamples)

        nSamples = None
        nTrainSamples = None

        # Split train_input into training and validation sets
        # Your code here ...

        return x_train, y_train, x_val, y_val

    def ex1_gen_dataloader(
        self,
        input_data: np.ndarray,
        label_data: np.ndarray,
        batch_size: int = 1024,
        shuffle: bool = True,
    ) -> data_utils.DataLoader:
        """
        Get dataloader for the input and label data.
        The input_data tensor should be converted to a FloatTensor.
        The label_data tensor should be converted to a LongTensor.
        Use data_utils.TensorDataset and data_utils.DataLoader.
        Transfer the data to the device.
        Make sure the dataloader is flexible to the shuffle option.

        Args:
            input_data (np.ndarray):    Input data. [nSamples, nFeatures]
            label_data (np.ndarray):    Label data. [nSamples]
            batch_size (int):           Batch size
            shuffle (bool):             Shuffle the data

        Returns:
            dataloader (DataLoader):    DataLoader for the input and label data
        """
        # Your code here ...

        dataloader = None

        return dataloader

    def ex1_build_model(self, nFeatures: int, nLabels: int, nHiddenFeatures: int, nLayers: int) -> nn.Sequential:
        """
        Build a simple feed-forward fully-connected neural network model.
        The model should have nLayers hidden layers with nHiddenFeatures neurons.
        Use ReLU activation functions for the hidden layers.
        The final layer should have nLabels neurons.
        The model should be moved to the device.

        Define the model as a nn.Sequential model. One easy way to define the model is to use a list of layers and pass it to nn.Sequential.

        Args:
            nFeatures (int):            Number of features
            nLabels (int):              Number of labels
            nHiddenFeatures (int):      Number of features in the hidden layers
            nLayers (int):              Number of hidden layers

        Returns:
            model (nn.Sequential):  Neural network model
        """
        # Your code here ...

        model = None

        return model

    def get_loss(self, loss_type: str = "CrossEntropyLoss") -> Union[nn.CrossEntropyLoss, nn.BCELoss]:
        """
        Define the loss function based on the loss_type. The available loss types are CrossEntropyLoss and BCELoss.

        Args:
            loss_type (str):    Type of loss function [CrossEntropyLoss, BCELoss]

        Returns:
            loss (Union[nn.CrossEntropyLoss, nn.BCELoss]):  Loss function
        """
        # Your code here ...

        loss = None

        return loss

    def get_optimizer(self, model: nn.Sequential, lr: float = 0.001) -> torch.optim.Adam:
        """
        Define the Adam optimizer.

        Args:
            model (nn.Sequential):  Neural network model
            lr (float):             Learning rate

        Returns:
            optimizer (torch.optim.Adam):  Optimizer
        """

        # Your code here ...

        optimizer = None

        return optimizer

    def compute_accuracy(self, output: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute the accuracy of the model.
        Accuracy is defined as below:
        
                        # of correct 
            Accuracy =  -------------
                        # of samples
        
        Don't forget to convert the tensor to a single float value.

        For ex1, the output shape is [nBatch, nLabels] and the label shape is [nBatch]. Use torch.max to get the predicted labels.
        For ex2, the output shape is [nBatch] and the label shape is [nBatch]. Use torch.round to get the predicted labels.

        Args:
            output (torch.Tensor):  Output of the model [nBatch, nLabels] for ex1 or [nBatch] for ex2
            labels (torch.Tensor):  Ground truth labels [nBatch]

        Returns:
            accuracy (float):       Accuracy of the model

        """
        # Your code here ...

        accuracy = None

        return accuracy

    def trainer(
        self,
        model: nn.Sequential,
        criterion: Union[nn.CrossEntropyLoss, nn.BCELoss],
        optimizer: torch.optim.Adam,
        dataloader_train: data_utils.DataLoader,
        dataloader_val: data_utils.DataLoader,
        num_epochs: int = 30,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Trains a model and returns the training and validation loss and accuracy.
        Complete the code where (*) is marked.

        Args:
            model:              The model to train.
            criterion:          The loss function.
            optimizer:          The optimizer.
            dataloader_train:   The training data.
            dataloader_val:     The validation data.
            num_epochs:         Number of epochs to train for.

        Returns:
            loss_train:         Training loss for each epoch.
            acc_train:          Training accuracy for each epoch.
            loss_val:           Validation loss for each epoch.
            acc_val:            Validation accuracy for each epoch.
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        compute_accuracy = kwargs.get("compute_accuracy", self.compute_accuracy)

        loss_train = np.zeros(num_epochs)
        acc_train = np.zeros(num_epochs)
        loss_val = np.zeros(num_epochs)
        acc_val = np.zeros(num_epochs)

        with tqdm(total=num_epochs, unit="Epoch", leave=True) as pbar:
            for epoch in range(num_epochs):
                # Training
                running_loss = 0.0
                running_acc = 0

                # Set the model to train mode (*)

                for local_batch, local_labels in dataloader_train:
                    # Input local_batch to the model and get the output (*)

                    # Compuate the loss (*)

                    # feedforward - backpropagation (*)
                    ## 1. initialize gradients to zero (zero_grad())
                    ## 2. Backpropagation (backward())
                    ## 3. Update parameters (step())

                    # Accumulate the loss and accuracy (*)
                    running_loss += None
                    running_acc += None

                # Average loss and accuracy (*)
                loss_train[epoch] = None
                acc_train[epoch] = None

                # Validation
                running_loss = 0.0
                running_acc = 0

                # Set the model to evaluation mode (*)

                with torch.no_grad():
                    for local_batch, local_labels in dataloader_val:
                        # Input local_batch to the model and get the output (*)

                        # Compute the loss (*)

                        # Accumulate the loss and accuracy (*)
                        running_loss += None
                        running_acc += None

                # Average loss and accuracy (*)
                loss_val[epoch] = None
                acc_val[epoch] = None

                # Update progress bar
                pbar.set_description(desc=f"Epoch {epoch: 3d}")
                pbar.set_postfix(
                    {
                        "train loss": f"{loss_train[epoch]: .3f}",
                        "train acc": f"{acc_train[epoch]: .3f}",
                        "val loss": f"{loss_val[epoch]: .3f}",
                        "val acc": f"{acc_val[epoch]: .3f}",
                    }
                )
                pbar.update()

        return loss_train, acc_train, loss_val, acc_val

    def ex1_tester(self, model: nn.Sequential, dataloader: data_utils.DataLoader, **kwargs) -> float:
        """
        Predict the labels using the trained model.
        Complete the code where (*) is marked.

        Args:
            model (nn.Sequential):      Neural network model
            dataloader (data_utils.DataLoader): DataLoader for the input and label data

        Returns:
            accuracy (float):       Accuracy of the model
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        compute_accuracy = kwargs.get("compute_accuracy", self.compute_accuracy)

        running_acc = 0

        # Set the model to evaluation mode (*)

        with torch.no_grad():
            for local_batch, local_labels in dataloader:
                # Input local_batch to the model and get the output (*)

                # Compute the accuracy (*)
                running_acc += None

        # Average accuracy
        accuracy = None
        return accuracy

    def ex2_gen_dataloader(
        self,
        input_data: np.ndarray,
        label_data: np.ndarray,
        batch_size: int = 10,
        shuffle: bool = True,
    ) -> data_utils.DataLoader:
        """
        Get dataloader for the input and label data.
        The input_data tensor should be converted to a FloatTensor.
        The label_data tensor should be converted to a LongTensor.
        Use data_utils.TensorDataset and data_utils.DataLoader.
        Transfer the data to the device.
        Make sure the dataloader is flexible to the shuffle option.

        Args:
            input_data (np.ndarray):    Input data. [nSamples, nFeatures]
            label_data (np.ndarray):    Label data. [nSamples]
            batch_size (int):           Batch size
            shuffle (bool):             Shuffle the data

        Returns:
            dataloader (DataLoader):    DataLoader for the input and label data
        """

        # Your code here ...

        dataloader = None

        return dataloader


class ex2_CNN5layers_FC(nn.Module):
    """
    CNN model with 5 convolutional layers and a fully connected
    """

    def __init__(self):
        super().__init__()

        # Initialize the layers
        # Your code here ...

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor [Batch, nChannels (1), nRows, nCols]

        Returns:
            x (torch.Tensor): Output tensor [Batch]
        """
        # Your code here ...

        x = None
        return x


class ex2_CNN1layer_global_avg(nn.Module):
    """
    CNN model with 1 convolutional layer and global average pooling
    """

    def __init__(self):
        super().__init__()

        # Initialize the layers
        # Your code here ...

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor [Batch, nChannels (1), nRows, nCols]

        Returns:
            x (torch.Tensor): Output tensor [Batch]
        """
        # Your code here ...

        x = None


class ex2_CNN5layers_global_avg(nn.Module):
    """
    CNN model with 5 convolutional layers and global average pooling
    """

    def __init__(self):
        super().__init__()

        # Initialize the layers
        # Your code here ...

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor [Batch, nChannels (1), nRows, nCols]

        Returns:
            x (torch.Tensor): Output tensor [Batch]
        """
        # Your code here ...

        x = None


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab09 import *

    op = Lab09_op()
