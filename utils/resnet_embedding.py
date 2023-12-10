# Script to extract feature vectors from ResNet50 pretrained models

import numpy as np 
from tqdm import tqdm
import torch 
import torchvision.models as models
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
    
    
def get_embedding(img, model, layer, gpu='cpu'):
    img = normalize(to_tensor(img)).unsqueeze(0).to(gpu)  # Send image to device
    embedding = torch.zeros(2048).to(gpu)  # Allocate tensor on the device

    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(-1))

    h = layer.register_forward_hook(copy_data)
    
    model(img)
    h.remove()
    return embedding.cpu()
        

def get_embeddings(imgs, gpu='cpu', data='train'):
    # Initialize model
    model = models.resnet50(weights='DEFAULT').to(gpu)
    layer = model._modules.get('avgpool')
    
    print(f"Getting embeddings for {data} set")
    embeddings = []
    for img in tqdm(imgs):
        embedding = get_embedding(img, model, layer, gpu=gpu)
        # flatten and convert to numpy
        embeddings.append(embedding.detach().numpy().flatten())
    return np.array(embeddings)


if __name__ == '__main__':
    from snack_dataset import SnackDataset

    DEVICE = 'mps' # Mac GPU, set to cuda if using cuda

    # Load the dataset
    snacks = SnackDataset()

    test_imgs =list(snacks.get_test_set())
    test_imgs = [img for img, _ in test_imgs]

    # Get test set embeddings
    test_embeddings = get_embeddings(test_imgs, gpu=DEVICE)
    
    print(test_embeddings.shape)