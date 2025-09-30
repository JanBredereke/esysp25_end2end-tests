from math import ceil, sqrt
from os.path import basename
from dataclasses import dataclass
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef

@dataclass
class ImageRecord:
    """
    Used to store a list of images and their prediction.
    """
    image: Union[np.ndarray, Image.Image]
    path: str
    predicted: str

class Tester():
    def __init__(
            self,
            model: Module,
            load_path: str,
            test_dataloader: DataLoader,
            print_test_information: bool = False,
            device: torch.device = torch.device('cpu')
        ) -> None:
        """
        The tester calls

        Parameters
        -----------
        model: Module
            The model to be tested
        load_path: str
            The path to the model .pth file
        test_dataloader: Dataloader
            The dataloader for the test data
        print_test_information: bool = False
            Whether to print test information.
        device: torch.device = torch.device('cpu')
            The hardware device.
        """
        self.model = model
        self.load_path = load_path
        self.test_dataloader = test_dataloader
        self.device = device
        self.print_test_information = print_test_information

    def test(self) -> None:
        """
        Start the test of the model
        """
        # initialize variables
        dataiter = iter(self.test_dataloader)
        classes = self.test_dataloader.dataset.classes
        total = 0
        wrong = 0
        wrong_predictions: list[ImageRecord] = []

        all_true_labels = []
        all_predicted_labels = []

        # load model
        loaded = torch.load(self.load_path)
        self.model.load_state_dict(loaded["state_dict"])
        self.model = self.model.to(device=self.device)
        print(f"Loaded best model (from epoch {loaded['epoch']})")

        for batch in dataiter:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            batch_size = len(labels)
            total += batch_size

            # let model predict images
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)

            outputs = outputs.to(device=self.device)
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

            # check predictions
            for i in range(batch_size):
                if predicted[i] != labels[i]:
                    if self.print_test_information:
                        predicted_label = predicted[i].item()

                        true_label = classes[labels[i]]
                        dataset_index = total - batch_size + i
                        # self.test_dataloader.dataset.imgs[i] returns tupel (path of the picture, class id)
                        file_name = basename(self.test_dataloader.dataset.imgs[dataset_index][0])
                        predicted_label_text = classes[predicted_label]

                        # store image with labels for plotting
                        wrong_predictions.append(ImageRecord(
                            image=images[i],
                            path=f"{true_label}/{file_name}",
                            predicted=predicted_label_text
                        ))

                    wrong += 1

        print(f'{total} images. {wrong} were predicted wrong. Accuracy: {100 * (1 - wrong / total)}%')

        if self.print_test_information:
            self.plot_test_pictures(wrong_predictions)

        # plot confusion matrix
        mcc = matthews_corrcoef(all_true_labels, all_predicted_labels)
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(xticks_rotation=45, cmap='Blues')
        plt.title(f"Confusion Matrix\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")
        plt.tight_layout()
        plt.show()


    def plot_test_pictures(self, wrong_predictions: list[ImageRecord]):
        """
        Plots the wrong predicted test pictures

        Parameters
        -----------
        wrong_predictions: list[ImageRecord]
            The wrong predicted pictures
        """
        to_pil = transforms.ToPILImage()
        nrows = ncols = ceil(sqrt(len(wrong_predictions)))
        fig = plt.figure(figsize=(ncols*2.5, nrows*2.5))
        plt.tight_layout()
        subplot_index = 1

        for wrong_pred in wrong_predictions:
            image = to_pil(wrong_pred.image)
            sub = fig.add_subplot(nrows, ncols, subplot_index)
            subplot_index += 1
            sub.set_title(f"{wrong_pred.path}\nWrong Prediction: {wrong_pred.predicted}", fontsize=7)
            plt.axis('off')
            plt.imshow(image)

        plt.show()
