import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

BATCH_SIZE = 64

def download_the_datasets():
    # Download training data from open datasets
    training_data = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    # Download test data from open datasets
    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    

def get_the_datasets():
    training_data = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=False,
        transform=ToTensor()
    )
    
    # Download test data from open datasets
    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    
    return training_data, test_data
    
    
def print_shape(data_set):
    
    dataloader = DataLoader(data_set, batch_size=BATCH_SIZE)

    for X, y in dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y [N]: {y.shape}")
        break

def iterate_and_visualize_dataset(data_set):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }   

    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data_set), size=(1,)).item()
        img, label = data_set[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    
# I already know this isnt using cuda so no need to check it
device = 'cpu'
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    
training_data, test_data = get_the_datasets()


model = MyNeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader: DataLoader, model: MyNeuralNetwork, loss_fn,  optimzer: torch.optim.SGD):
    # In a single loop, model makes predictions on the training dataset (fed in batches) and backpropagates the
    # prediction error to adjust the model parameters.

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediciton error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        
        # Will figure out its type later
        loss.backward()

        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader: DataLoader, model: MyNeuralNetwork, loss_fn):
    # Checks the models performance against the test to ensure it is learning.

    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()
    test_loss /= n_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Ave loss: {test_loss:>8f} \n")

epochs = 5



training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Run the model

def train_the_model():
    for t in range(epochs):
        print(f"Epoch {t + 1}\n ----------------")
        train(training_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done training!")

    # Here I save the model after running it
    torch.save(model.state_dict(), "basicmodel.pt")

    

def test_the_model():
    # How well our model can be used to make predictions
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    # Model being used for predictions
    loaded_model = MyNeuralNetwork()

    # Next I load that saved model from disk
    loaded_model.load_state_dict(torch.load("basicmodel.pt"))

    print("loaded the saved model")
    
    loaded_model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f"Predicted: {predicted}, Actual: {actual}")


test_the_model()