import numpy as np
import albumentations
from datasets import load_dataset
import matplotlib.pyplot as plt


MAPPER = {
            0: "apple",
            1: "banana",
            2: "cake",
            3: "candy",
            4: "carrot",
            5: "cookie",
            6: "doughnut",
            7: "grape",
            8: "hot dog",
            9: "ice cream",
            10: "juice",
            11: "muffin",
            12: "orange",
            13: "pineapple",
            14: "popcorn",
            15: "pretzel",
            16: "salad",
            17: "strawberry",
            18: "waffle",
            19: "watermelon"
        }


class SnackDataset:
    """
    A class to load the snacks dataset and apply transformations.
    Attributes:
        dataset_name (str): The name of the dataset to be loaded.
        dataset (DatasetDict): The loaded dataset object.
        transforms (callable): The transformations to be applied to the images.
    """
    def __init__(self):
        """Resize the images to 256x256 pixels."""
        self.dataset_name = "Matthijs/snacks"
        self.dataset = load_dataset(self.dataset_name)
        self.dataset.set_transform(self.transforms)

    def get_train_set(self):
        for img in self.dataset['train']:
            yield img['pixel_values'], img['label']

    def get_test_set(self):
        for img in self.dataset['test']:
            yield img['pixel_values'], img['label']

    def get_validation_set(self):
        for img in self.dataset['validation']:
            yield img['pixel_values'], img['label']
        
    @staticmethod
    def transforms(examples):
        """Transforms the images to 256x256 pixels."""
        transform = albumentations.Compose([albumentations\
            .RandomCrop(width=256, height=256)])
        examples["pixel_values"] = [
            transform(image=np.array(image))["image"] for image in examples["image"]
        ]
        return examples
        
    def plot_snack(self, img, label=None):
        plt.imshow(img)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title(f"Label: {self.label_mapping(label).capitalize()}")
        plt.show()

    @staticmethod   
    def label_mapping(label: int):
        """Maps the label to the corresponding snack name."""
        return MAPPER.get(label)
    
    def reverse_label_mapping(self, label: str):
        """Maps the snack name to the corresponding label."""
        return {v: k for k, v in MAPPER.items()}.get(label.lower())
    
    def get_images_by_label(self, label: int, type='train'):
        """
        Returns all images of a certain label.
        type (str): The type of dataset to be used. Can be 'train', 'test' or 'validation'.
        """
        images = self.dataset[type].filter(lambda example: example['label'] == label)
        for img in images:
            yield img['pixel_values'], img['label']
            
    def get_random_image_by_label(self, label, size: int=1, type='train'):
        images = list(self.get_images_by_label(label, type=type))
        random_indices = np.random.choice(np.arange(len(images)), size=size, replace=False)
        for img in random_indices:
            yield images[img][0], images[img][1]
