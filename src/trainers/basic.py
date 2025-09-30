import time
import torch
import matplotlib.pyplot as plt
import os
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Module
from torch.utils.data import DataLoader

class Trainer():
    def __init__(
        self,
        model: Module,
        save_path: str,
        train_dataloader: DataLoader,
        validate_dataloader: DataLoader,
        device: torch.device = torch.device('cpu'),
        max_epochs: int = 10,
        patience: int = 5,
        print_every_n_batches: int = 10
    ) -> None:
        """
        The trainer class

        Parameters
        -----------
        model: Module
            The model used for training
        save_path: str
            The save path for the best model 
        train_dataloader: DataLoader
            The dataloader for the train data
        validate_dataloader: DataLoader
            The dataloader for the validation data
        device: torch.device = torch.device('cpu')
            The hardware device used for training.
        max_epochs: int = 10
            The upper limit for training epochs. Defaults to 10
        patience: int = 5
            The patience used for determining if the model is overfitting.
        print_every_n_batches: int = 10
            Print average loss over n batches.
        """
        self.model = model
        self.save_path = save_path
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.print_every_n_batches = print_every_n_batches
        self.history = {
            "train_loss": [],
            "validate_loss": [],
            "best_epoch": None
        }


    def train(self):
        """
        Start training of model
        """
        self.model = self.model.to(self.device)
        optimizer = Adam(self.model.parameters())
        loss_fn = CrossEntropyLoss().to(self.device)

        print("Starting Training:")
        start_time = time.time()
        best_validate_loss = float("inf")
        overwritten_prev_best_model = False
        if not os.path.exists(self.save_path):
            print("No previous checkpoint found")
            previous_model_best_val_loss = float("inf")
        else:
            # assign last model score as best score 
            checkpoint = torch.load(self.save_path, map_location=self.device)
            try:
                previous_model_best_val_loss = checkpoint['best_val_loss']
            except:
                # if previous model doesnt have the new checkpoint key
                previous_model_best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            ### train
            self.model.train()
            cum_train_loss = 0.0
            for i, batch in enumerate(self.train_dataloader):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                cum_train_loss += loss.item()
                if (i+1) % self.print_every_n_batches == 0:
                    self.print_current_statistics(i, epoch, cum_train_loss)
            mean_train_loss = cum_train_loss / len(self.train_dataloader)

            ### validate
            self.model.eval()
            cum_validate_loss = 0.0
            # speed up tensor operations by hinting that we don't care about gradients here
            with torch.no_grad():
                for batch in self.validate_dataloader:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    cum_validate_loss += loss_fn(outputs, labels).item()
            mean_validate_loss = cum_validate_loss / len(self.validate_dataloader)

            self.history["train_loss"].append(mean_train_loss)
            self.history["validate_loss"].append(mean_validate_loss)

            # if model improved
            if mean_validate_loss < best_validate_loss:
                best_validate_loss = mean_validate_loss
                epochs_without_improvement = 0
                self.history["best_epoch"] = epoch+1
                if best_validate_loss < previous_model_best_val_loss:
                    # save model only if its better than the last saved model
                    overwritten_prev_best_model = True
                    self.save_best(epoch, best_validate_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping after epoch {epoch+1}, validate loss did not improve for {epochs_without_improvement} epochs")
                    break

        print('Finished Training in %.3fs.\n' % (time.time() - start_time))
        if overwritten_prev_best_model:
            print("Found a new best model with validate loss: %.3f" % best_validate_loss)
            print("Hint: The previous best model has been overwritten.")
        else:
            print("No new best model found, the previous best model is still the best one with validate loss: %.3f"
                  % previous_model_best_val_loss)
    

    def plot_history(self) -> None:
        """
        Plots the history collected during training, which includes train loss, validate loss and a marker for the best model. 
        """
        x_epochs = list(range(1, len(self.history["train_loss"]) + 1))
        plt.title("Loss")
        plt.plot(x_epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(x_epochs, self.history["validate_loss"], label="Validate Loss")
        plt.axvline(self.history["best_epoch"], color="g", linestyle=":", label="Best Model")
        plt.xlabel("Epoch")
        plt.xticks(x_epochs)
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)

    def save_best(self, epoch: int, best_validation_loss: float):
        """
        Saves the current model 

        Parameters
        ----------
        epoch: int
            The current epoch, starting from 0

        Hint 
        ----------
        This function has no own evaluation computation, it needs be called after determining that a model should be saved. 
        """
        torch.save({
            "state_dict": self.model.state_dict(),
            "epoch": epoch+1,
            "best_val_loss": best_validation_loss
        }, self.save_path)

    def print_current_statistics(self, batches: int, epoch: int, cumulated_loss: float):
        """
        Print statistics over the given batches
        
        Parameters
        ----------
        batches: int
            The number of already processed batches
        epoch: int 
            The current epoch
        cumulated_loss: float
            The cumulative loss over the given amount of batches
        """
        average_loss = cumulated_loss / batches
        print('[Epoch %2d] Average loss over last %3d Batches: %.3f' % ((epoch+1), (batches+1), average_loss))