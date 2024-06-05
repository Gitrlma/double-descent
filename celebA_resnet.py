import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(nn.functional.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, init_channels = 64, in_dim = 3):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(in_dim, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def train_model(model_name, dimensions, classes, layer_width, epochs, size = 128):
    colour = 'coloured' if dimensions == 3 else 'grayscale'
    writer = SummaryWriter(log_dir=f'celebA/resnet/epoch_wise/width_{layer_width}/{model_name}_{colour}')
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

    training_set = torchvision.datasets.CelebA(root='./data', split = 'train', download = True, transform = transform)
    if model_name == "100k":  # If model is 100k the dataset is reduced from 160k samples to 100k
        training_set = torch.utils.data.Subset(training_set, range(0, 100000))  # Random subset of 100k samples

    validation_set = torchvision.datasets.CelebA(root='./data', split = 'valid', download = True, transform = transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers = 8)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers = 8)

    model = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes = classes, init_channels = layer_width, in_dim = dimensions) #Adoperate paper resnet
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(),
                           lr=learning_rate)  # TODO optim.SGD(model, lr=learning_rate, momentum=0.9)

    for epoch in tqdm(range(epochs)):

        # Model training
        model.train()
        train_loss = 0
        train_epoch_steps = 0
        for images, labels in training_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = criterion(pred.squeeze(), labels[:, 2].float())
            loss.backward()

            train_loss += loss.item() # Forming the total epoch loss as sum of all iteration losses
            train_epoch_steps += 1

            optimizer.step()

        train_epoch_loss = train_loss / train_epoch_steps # Fetch the average train epoch loss
        writer.add_scalar("train_epoch_loss", train_epoch_loss, epoch) # Add loss to tensorboard


        # Model evaluation
        total_loss = 0.0
        validation_epoch_steps = 0
        model.eval()
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                pred = model(images)
                loss = criterion(pred.squeeze(), labels[:, 2].float())
                total_loss += loss.item()
                validation_epoch_steps += 1

        test_epoch_loss = total_loss / validation_epoch_steps  # Average test epoch loss
        writer.add_scalar("test_epoch_loss", test_epoch_loss, epoch)  # Add loss to tensorboard
        print(f'Epoch {epoch}  Training loss: {train_epoch_loss}  Test loss: {test_epoch_loss}')


if __name__ == "__main__":
    torch.manual_seed(1234)
    model_names = ['160k', '100k']
    dimensions = [1, 3]
    classes = 1 # CelebA classes to predict
    layer_widths = [2, 4, 6, 8, 16, 32, 64]
    epochs = 100
    size = 128
    #train_model(model_names[1], dimensions[0], classes, layer_width, epochs, size)
    for layer_width in layer_widths:
        for i in range(len(model_names)):
            for j in range(len(dimensions)):
                train_model(model_names[i], dimensions[j], classes, layer_width, epochs, size)
                print(f'Model: {model_names[i]} with {dimensions[j]} dimensions is completed')