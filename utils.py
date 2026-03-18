from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    lengths = [len(dataset)//3]*3
    datasets_split = random_split(dataset, lengths)

    loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in datasets_split]
    return loaders
