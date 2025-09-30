from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from .dataset import DatasetPair
import os


class CustomDataSet(DatasetPair):
    """
    Reads an Image Folder and loads it into pytorch DataLoader/Dataset.
    Images in the "test" directory will be used for final testing, while
    Images in the "train" directory will be split into train and validate sets
    using the given validation_ratio.
    The file structure for the personen-datensatz shall look like this:
    /path/to/my_daset/test/person/0.jpg
    /path/to/my_daset/test/noperson/0.jpg
    /path/to/my_daset/train/person/0.jpg
    /path/to/my_daset/train/noperson/0.jpg
    Considering the function parameters, e.g.:
    {root_dir}/{dataset}/test/person/0.jpg
    For the gesten-datensatz, it shall look like this:
    /path to gesten-datensatz repository/gesten-datensatz/gesture/source/train/start/0.jpg
    /path to gesten-datensatz repository/gesten-datensatz/gesture/source/train/start/1.jpg
    /path to gesten-datensatz repository/gesten-datensatz/gesture/source/train/stop/0.jpg
    /path to gesten-datensatz repository/gesten-datensatz/gesture/source/test/start/0.jpg
    /path to gesten-datensatz repository/gesten-datensatz/gesture/source/test/stop/0.jpg
    """
    def __init__(
        self,
        root_dir: str = os.path.join("..", "data"),
        dataset: str = "custom_dataset",
        validate_ratio: float = 0.2,
        image_size: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 0,
    ):
        # This may be varied
        transform_train = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                #transforms.RandomCrop(image_size, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

        train_set = ImageFolder(
            os.path.join(root_dir, dataset, "train"), transform_train
        )

        classes = train_set.class_to_idx

        test_set = ImageFolder(
            os.path.join(root_dir, dataset, "test"), transform_test
        )

        # split train_set into train_set and validate_set
        labels = [sample[1] for sample in train_set]
        train_set, validate_set, _, _ = train_test_split(
            train_set,
            labels,
            test_size=validate_ratio,
            stratify=labels, # ensures class balance in train and validate sets
            random_state=seed
        )

        super().__init__(train_set=train_set, test_set=test_set, validate_set=validate_set, classes=classes, batch_size=batch_size, num_workers=num_workers, seed=seed)
