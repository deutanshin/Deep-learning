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
        pass
        ##############################################################
        # END OF YOUR CODE
        ##############################################################

    def forward(self, x):
        ##############################################################
        # TODO: implement forward pass
        ##############################################################
        return x
        ##############################################################
        # END OF YOUR CODE
        ##############################################################

class Solver:
    def __init__(
        self,
        # add your hyper-parameters here
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ##############################################################
        # TODO: implement solver initialization
        ##############################################################
        pass
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
                pass
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
