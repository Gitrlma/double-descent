# 5-Layer CNN for CIFAR
# Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy

class Flatten(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x.view(x.size(0), x.size(1))  # Flattens the convolutional output

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


def train_model(model_name, dimensions, classes, layer_width, epochs, size = 128):
    colour = 'coloured' if dimensions == 3 else 'grayscale'
    writer = SummaryWriter(log_dir=f'celebA/mcnn/epoch_wise/width_{layer_width}/{model_name}_{colour}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1234)

    if dimensions == 1:  # 1 Dimension = grayscale
        transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Resize((size, size))]
        )
    else:  # 3 Dimensions = Coloured
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((size, size))]
        )

    batch_size = 256

    training_set = torchvision.datasets.CelebA(root='./data', split = 'train', download=True, transform=transform)
    if model_name == "100k":  # If model is 100k the dataset is reduced from 160k samples to 100k
        training_set = torch.utils.data.Subset(training_set, range(0, 100000))  # Random subset of 100k samples

    validation_set = torchvision.datasets.CelebA(root='./data', split = 'valid', download=True, transform=transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                                  shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                             shuffle=False)

    model = make_cnn(c = layer_width, num_classes = classes, in_dim = dimensions)  # adoperate paper CNN
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
    metric = BinaryAccuracy()
    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(),
                           lr=learning_rate)  # TODO optim.SGD(model, lr=learning_rate, momentum=0.9)

    for epoch in tqdm(range(epochs)):

        # Model training
        model.train()
        train_loss = 0
        train_epoch_steps = 0
        train_acc = 0
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred.squeeze(), labels[:, 2].float()) # label fetched is attractive
            loss.backward()
            acc = metric(pred.squeeze(), labels[:, 2].float())

            train_loss += loss.item() # Forming the total epoch loss as sum of all iteration losses
            train_acc += acc
            train_epoch_steps += 1

            optimizer.step()

        train_epoch_loss = train_loss / train_epoch_steps # Fetch the average train epoch loss
        writer.add_scalar("train_epoch_loss", train_epoch_loss, epoch) # Add loss to tensorboard
        train_epoch_acc = train_acc / train_epoch_steps  # Fetch the average train epoch accuracy
        writer.add_scalar("train_epoch_accuracy", train_epoch_acc, epoch)  # Add accuracy to tensorboard
        #print(f'Epoch {epoch} training loss: {train_epoch_loss}')


        # Model evaluation
        total_loss = 0.0
        validation_epoch_steps = 0
        test_acc = 0
        model.eval()
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                pred = model(images)
                loss = criterion(pred.squeeze(), labels[:, 2].float())
                acc = metric(pred.squeeze(), labels[:, 2].float())
                total_loss += loss.item()
                test_acc += acc
                validation_epoch_steps += 1

        test_epoch_loss = total_loss / validation_epoch_steps  # Average test epoch loss
        writer.add_scalar("test_epoch_loss", test_epoch_loss, epoch)  # Add loss to tensorboard
        test_epoch_acc = test_acc / validation_epoch_steps  # Fetch the average test epoch accuracy
        writer.add_scalar("test_epoch_accuracy", test_epoch_acc, epoch)  # Add loss to tensorboard
        print(f'Epoch {epoch}  Training loss: {train_epoch_loss}  Test loss: {test_epoch_loss}')


if __name__ == "__main__":
    torch.manual_seed(1234)
    model_names = ['160k', '100k']
    dimensions = [1, 3]
    classes = 1  # CelebA classes to predict
    layer_widths = [2, 4, 6, 8, 16, 32, 64]
    epochs = 100
    size = 32
    # train_model(model_names[1], dimensions[0], classes, layer_width, epochs, size)
    for layer_width in layer_widths:
        for i in range(len(model_names)):
            for j in range(len(dimensions)):
                train_model(model_names[i], dimensions[j], classes, layer_width, epochs, size)
                print(f'Model: {model_names[i]} with {dimensions[j]} dimensions is completed')