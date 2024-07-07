import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from torch.utils.tensorboard import SummaryWriter


# LSTM
# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def lstm_train_and_validate(model, dataloaders, loss_fn, optimizer, num_epochs, device, writer=None):
    model.to(device)
    early_stopping = EarlyStopping(patience=10, verbose=True, path="best_model_checkpoint.pth")  # Configure early stopping

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    # Initialize lists to track losses
    train_losses = []
    val_losses = []

    # Log model architecture
    if writer is not None:
        # Get a sample batch from the dataloader to represent input data
        inputs, _ = next(iter(dataloaders["train"]))
        inputs = inputs.to(device)
        writer.add_graph(model, inputs)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            running_loss = 0.0

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for features, labels in dataloaders[phase]:
                features = features.to(device)  # Move features to device
                labels = labels.to(device)  # Move labels to device
                optimizer.zero_grad()

                # forward and track history only if in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(features)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * features.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
                if writer is not None:
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
            else:
                val_losses.append(epoch_loss)
                if writer is not None:
                    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
                # Early stopping check
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        if early_stopping.early_stop:
            print("Stopping training")
            break

    if writer is not None:
        writer.close()

    model.load_state_dict(torch.load('best_model_checkpoint.pth'))  # load best model weights

    return model, train_losses, val_losses

