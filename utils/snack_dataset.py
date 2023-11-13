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
    """Load the dataset and apply the transform"""
    def __init__(self, transform=None):
        self.dataset_name = "Matthijs/snacks"
        self.dataset = load_dataset(self.dataset_name)
        self.transforms = self.transforms if transform is None else transform
        self.dataset.set_transform(self.transforms)

    def get_train_set(self):
        train_dataset = self.dataset['train']
        for img in train_dataset:
            yield img['pixel_values'], img['label']

    def get_test_set(self):
        test_dataset = self.dataset['test']
        for img in test_dataset:
            yield img['pixel_values'], img['label']

    def get_validation_set(self):
        validation_dataset = self.dataset['validation']
        for img in validation_dataset:
            yield img['pixel_values'], img['label']
            
    def get_train_length(self):
        return len(self.dataset['train'])
    
    def get_val_length(self):
        return len(self.dataset['validation'])
    
    def get_test_length(self):
        return len(self.dataset['test'])
        
    @staticmethod
    def transforms(examples):
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
        return 

    @staticmethod   
    def label_mapping(label: int):
        return MAPPER.get(label)
    
    def reverse_label_mapping(self, label: str):
        return {v: k for k, v in MAPPER.items()}.get(label.lower())
    
    def get_images_by_label(self, label: int, type='train'):
        images = self.dataset[type].filter(lambda example: example['label'] == label)
        for img in images:
            yield img['pixel_values'], img['label']
            
    def get_random_image_by_label(self, label, size: int=1, type='train'):
        images = list(self.get_images_by_label(label, type=type))
        random_indices = np.random.choice(np.arange(len(images)), size=size, replace=False)
        for img in random_indices:
            yield images[img][0], images[img][1]
