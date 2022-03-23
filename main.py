from ast import Lambda
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

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
    # NOTE: refactor to allow multiple datasets and display them respectively with their titles
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
    
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_lables = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_lables)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_lables.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_lables.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    


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





# Run the model

def train_the_model():
    for t in range(EPOCHS):
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
        
    

def old_tensor_tests():
    data = [[1,2], [3,4]]
    np_array= np.array(data)
    x_data = torch.tensor(data)
    x_np = torch.from_numpy(np_array)
    
    # Here is my tensor same structure as the one I created just filled with 1's
    x_ones = torch.ones_like(x_data)
    
    # Same as above but with random numbers
    x_rand = torch.rand_like(x_data, dtype=torch.float32) # overrides the datatype of x_data which would be int
    
    print(f"data: {data} \n np_array: {np_array}\nx_data: {x_data}\nx_np: {x_np}")
    
    print(f"Ones Tensor: \n {x_ones}\n")
    print(f"Random Tensor: \n {x_rand} \n")
    
    # shape is a tuple of tensor dimensions
    shape = (2,3,)
    
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.rand(shape)
    zeros_tensor = torch.rand(shape)
    
    print(f"rand_tensor: {rand_tensor} \n ones_tensor: {ones_tensor} \n zeros_tensor: {zeros_tensor}")
    
    my_tensor = torch.rand(3,4)
    
    print(f"Shape of my_tensor: {my_tensor.shape}")
    print(f"Datatype of my_tensor: {my_tensor.dtype}")
    print(f"Device tensor is on: {my_tensor.device}")


def current_tensor_tests():
    shape = (4, 4,)
    
    tensor = torch.ones(shape)
    tensor2 = torch.ones(shape)
    tensor3 = torch.ones(shape)
    # standard numpy like operations
    print(tensor)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:,1] = 0
    tensor2[:,2] = 0
    tensor3[:,3] = 0
    print(f"Tensor: {tensor}")
    
    t1 = torch.cat([tensor, tensor2, tensor3], dim=-1)
    
    # The computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)
    
    print(f"y1: {y1}")
    print(f"y2: {y2}")
    print(f"y3: {y3}")
    
    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    
    print(f"z1: {z1}")
    print(f"z2: {z2}")
    print(f"z3: {z3}")
    
    # Single element tensor, if there is a one element tensor, it cna be converted to a python nemerical values using item()
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
    
    # Inplace operation can save memeory but can be an issue when computing derivatives because of an immediate loss of history
    # Generally discouraged
    
    print(f"Tensor: {tensor}")
    tensor.add_(5)
    print(f"Tensor after add: {tensor}")

    # Bridge with numpy
    t = torch.ones(5)
    print(f"Tensor t: {t}")
    n = t.numpy()
    print(f"Numpy n converted from t: {n}")
    
    # Change in the tensor will be reflected in the numpy array
    t.add_(1)
    print(f"Tensor t: {t}")
    print(f"Numpy n converted from t: {n}")
    
    # numpy array to tensor
    n = np.ones(5)
    t = torch.from_numpy(n)
    
    np.add(n, 1, out=n)
    print(f"Numpy n: {n}")
    print(f"Tensor t: {t}")


def iterate_throught_dataloader():
    train_features, train_lables = next(iter(training_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_lables.size()}")
    img = train_features[0].squeeze()
    label = train_lables[0]
    plt.imshow(img, cmap='gray')
    plt.show()
    print(f"Label: {label}")
    
    
    
# Already have the datasets locally so no need to download
# ds = datasets.FashionMNIST(
#     root="./data",
#     train=True,
#     download=False,
#     transform=ToTensor(),
#     target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float64).scatter_(0, torch.tensor(y), value=1))
# )

def building_and_testing_neural_net():
    # Only need to device if you want to use GPU, default is CPU 
    model = MyNeuralNetwork().to(device)

    # training_data, test_data = get_the_datasets()
    # training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    
    # Take a sample minibatch of 3 images of size 28 x 28 and see what happens to it as we pass it through the model
    input_image = torch.rand(3,28,28)
    print(f"Input image size: {input_image.size()}")
    
    # Initialize the nn.Flatten layer to convert each 2D 28 x 28 image into a 1D 784 vector
    # The minibatch dimension at dim=0 is maintained
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(f"Flattened image size: {flat_image.size()}")
    
    # The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(f"Hidden layer 1 size: {hidden1.size()}")
    
    # The Non linear activation are what create the complex mappings between the models inputs and outputs.
    # They are applied after linear transforations to introduce nonlinearity, helping the networks learn a
    # wide variety of phenomena.
    
    print(f"Before ReLU: {hidden1}\n \n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")
    
    
    # nn.Sequential is an ordered container of modules, data is passed through all the modules in the same order as defined.
    # You can use sequential containers to put together a quick network like seq_modules.
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20,10)
    )
    
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)
    
    # nn.Softmax is the last linear layer of the neural network, it returnts logits, raw values in [-inf, inf]
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)

    print(f"Model structure: {model}\n \n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")
        
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 3 x 3 square convolution kernel
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        pass
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
def matplotlib_stuff():
    trend = np.linspace(10, 100, 100)
    x = np.linspace(-math.pi, math.pi, 100)
    sin_wave = np.sinh(x) * 10
    noise = np.random.normal(0, 5, 100)
    noisey_sine = sin_wave + noise
    plt.plot(noisey_sine)
    plt.show()

if __name__ == '__main__':
        
    # Automatic differentiation with autograd
    x = torch.ones(5) # input tensor
    y = torch.zeros(3) # expected output
    
    w = torch.randn(5, 3, requires_grad=True) # weight tensor
    b = torch.randn(3, requires_grad=True) # bias tensor

    z = torch.matmul(x, w) + b # linear transformation
    
    # Calculate the jacobian product myself
    if False:
        z_det = z.detach()
        print(z_det.requires_grad)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # loss function
        
        print(f"Gradient function for z = {z.grad_fn}")
        print(f"Gradient funciton for loss = {loss.grad_fn}")

        loss.backward() # backpropagate the loss
        print(w.grad)
        print(b.grad)
        inp = torch.eye(5, requires_grad=True)
        out = (inp + 1).pow(2)
        out.backward(torch.ones_like(inp), retain_graph=True)
        print(f"First call\n{inp.grad}")
        out.backward(torch.ones_like(inp), retain_graph=True)
        print(f"\nSecond call\n{inp.grad}")
        inp.grad.zero_()
        out.backward(torch.ones_like(inp), retain_graph=True)
        print(f"\n Call after zeroing gradients\n{inp.grad}")
        
    r1 = torch.rand(2,2)
    
    print(f"Determinant of r1: {torch.det(r1)}")
    print(f"Sin of r1: {torch.sin(r1)}")
    print(f"Average standard deviation of r1: {torch.std(r1)}")
    
    matplotlib_stuff()
    