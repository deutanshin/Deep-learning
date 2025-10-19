import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################
        # TODO: implement forward pass
        ##############################################################
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        ##############################################################
        # END OF YOUR CODE
        ##############################################################

    def forward(self, x):
        ##############################################################
        # TODO: implement forward pass
        ##############################################################
        x = self.conv_layers(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.fc_layers(x)

        return x
        ##############################################################
        # END OF YOUR CODE
        ##############################################################

class Solver:
    def __init__(
        self,
        lr=3e-3,           
        epochs=40,         
        batch_size=32,
        weight_decay=1e-4,     
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ##############################################################
        # TODO: implement solver initialization
        ##############################################################
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = MyNet().to(self.device)

        self.loss_func = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )


        transform = T.Compose([T.ToTensor()])
        _train_set = tv.datasets.CIFAR10("./", train=True, download=True, transform=transform)
        
        _temp_loader = torch.utils.data.DataLoader(_train_set, batch_size=batch_size)
        total_steps = len(_temp_loader) * epochs

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, total_steps=total_steps
        )
        ##############################################################
        # END OF YOUR CODE
        ##############################################################

        # dataset preparation
        transform = T.Compose([T.ToTensor()])
        self.train_set = tv.datasets.CIFAR10("./", train=True, download=True, transform=transform)
        self.test_set  = tv.datasets.CIFAR10("./", train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=1)
        self.test_loader  = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=1)

        self.history = {"loss": [], "test_acc": []}

    def train(self):
        for epoch in range(1, self.epochs+1):
            self.model.train()

            for x, y in self.train_loader:
                ##########################################################
                # TODO: implement training loop
                ##########################################################
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.loss_func(logits, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() 
                ##########################################################
                # END OF YOUR CODE
                ##########################################################
                self.history["loss"].append(loss.item())
                
            test_acc = evaluate(self.model, self.test_loader, self.device, self.loss_func)
            self.history["test_acc"].append(test_acc)
            print(f"[epoch: {epoch}] loss {loss:.4f}, test acc {test_acc:.3f}")


@torch.no_grad()
def evaluate(model, test_loader, device, criterion):
    model.eval()

    total_correct, total_count = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_correct += (logits.argmax(1) == y).sum().item()
        total_count += y.size(0)

    model.train()
    return total_correct / total_count
