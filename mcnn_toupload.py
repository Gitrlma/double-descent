# 5-Layer CNN for CIFAR
# Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))  # Flattens the convolutional output

def make_cnn(c=64, num_classes=10, in_dim = 3):
    '''
    Returns a 5-layer CNN with width parameter c.
    where c is the output channels
    '''
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(in_dim, c, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.BatchNorm2d(c),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(c, c*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*2),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(c*2, c*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*4),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(c*4, c*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*8),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c*8, num_classes, bias=True)
    )


def train_model(model_name, dimensions, classes, layer_width, epochs):
    colour = 'coloured' if dimensions == 3 else 'grayscale'
    writer = SummaryWriter(log_dir=f'mcnn/width_{layer_width}/{model_name}_{colour}')

    torch.manual_seed(1234)

    if dimensions == 1: # 1 Dimension = grayscale
        transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor()]
        )
    else: # 3 Dimensions = Coloured
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    batch_size = 256

    training_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    if model_name == "40k":  # If model is 40k the dataset is reduced from 50k samples to 40k
        training_set = torch.utils.data.Subset(training_set, range(0, 40000))  # Random subset of 40k samples

    validation_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                   download=True, transform=transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                                  shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                             shuffle=False)

    model = make_cnn(c = layer_width, num_classes = classes, in_dim = dimensions)  # adoperate paper CNN
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(),
                           lr=learning_rate)  # TODO optim.SGD(model, lr=learning_rate, momentum=0.9)

    for epoch in tqdm(range(epochs)):

        # Model training
        model.train()
        train_loss = 0
        train_epoch_steps = 0
        #for batch_idx, (images, labels) in enumerate(tqdm(training_loader)):
        for images, labels in training_loader:
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()

            train_loss += loss.item() # Forming the total epoch loss as sum of all iteration losses
            train_epoch_steps += 1

            optimizer.step()

        train_epoch_loss = train_loss / train_epoch_steps # Fetch the average train epoch loss
        #writer.add_scalar("train_epoch_loss", train_epoch_loss, epoch) # Add loss to tensorboard
        #print(f'Epoch {epoch} training loss: {train_epoch_loss}')


        # Model evaluation
        total_loss = 0.0
        validation_epoch_steps = 0
        model.eval()
        #for batch_idx, (images, labels) in enumerate(tqdm(validation_loader)):
        for images, labels in validation_loader:
            with torch.no_grad():
                pred = model(images)
                loss = criterion(pred, labels)
                total_loss += loss.item()
                validation_epoch_steps += 1

        test_epoch_loss = total_loss / validation_epoch_steps  # Average test epoch loss
        writer.add_scalar("test_epoch_loss", test_epoch_loss, epoch)  # Add loss to tensorboard
        #print(f'Epoch {epoch} test loss: {test_epoch_loss}')


if __name__ == "__main__":
    torch.manual_seed(1234)
    model_names = ['50k', '40k']
    dimensions = [1, 3]
    classes = 100
    layer_width = 32
    epochs = 100
    train_model(model_names[1], dimensions[1], classes, layer_width, epochs)

'''
    for i in range(len(model_names)):
        for j in range(len(dimensions)):
            train_model(model_names[i], dimensions[j], classes, layer_width, epochs)
            print(f'Model: {model_names[i]} with {dimensions[j]} dimensions is completed')'''