from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import passivitycheck as pc
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import fast_gradient_method as fgm
import projected_gradient_descent as pgd


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 8, 1)
        self.conv2 = nn.Conv2d(8, 1, 8, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(49, 49)
        self.fc2 = nn.Linear(49, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def extract_weights(self):
        weights = []
        biases = []
        for param_tensor in self.state_dict():
            tensor = self.state_dict()[param_tensor].detach().numpy().astype(np.float64)
            if 'weight' in param_tensor:
                weights.append(tensor)
            if 'bias' in param_tensor:
                biases.append(tensor)
        return weights, biases


def train(args, model, device, train_loader, optimizer, epoch, refmat=None):
    
    model.train()
    W_bar ={}
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if refmat is not None:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if 'conv' in name:
                        if args.passive_conv:
                            W_bar[name] = torch.tensor(refmat[name])
                            loss += torch.sum((param-W_bar[name])**2)
                    else:
                        W_bar[name] = torch.tensor(refmat[name])
                        loss += torch.sum((param-W_bar[name])**2)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),100*batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader,refmat=None):
    model.eval()
    test_loss = 0
    correct = 0
    W_bar ={}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,reduction='sum').item()
            if refmat is not None:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        W_bar[name] = torch.tensor(refmat[name])
                        test_loss += torch.sum((param-W_bar[name])**2)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_attack(model, device, test_data,labels,attack_type,epsilon):
    model.eval()
    test_loss = 0
    correct = 0
    test_data, labels = test_data.to(device), labels.to(device)
    output = model(test_data)
    test_loss += F.nll_loss(output, labels,reduction='sum').item()
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_data)
    print('\nTest set loss to {} attack (strength {}): {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(attack_type,epsilon,
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='Skips Training')
    parser.add_argument('--passive-conv', action='store_true', default=False,
                        help='Passive Convolution Layers')

    epsilon = [0.5, 1, 3, 5]
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    if not args.no_train:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn_regular.pt")
    
    auxdict = torch.load("mnist_cnn_regular.pt")
    auxmodel = Net().to(device)
    auxmodel.load_state_dict(auxdict)

    test_data = []
    labels = []
    for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_data.append(data)
            labels.append(target)
    test_data = torch.cat(test_data,dim=0)
    labels = torch.cat(labels,dim=0)
    
    test(auxmodel,device,test_loader)

    print("Vanilla Model Test Errors to Attacks\n")
    for i in epsilon:
        fgm_adv = fgm.fast_gradient_method(auxmodel,test_data,i,2,clip_min=-0.4242,clip_max=2.8215)
        test_attack(auxmodel,device,fgm_adv,labels,"FGM",i)
    
        pgd_adv = pgd.projected_gradient_descent(auxmodel,test_data,i,0.1,10,2,clip_min=-0.4242,clip_max=2.8215)
        test_attack(auxmodel,device,pgd_adv,labels,"PGD",i)

    if not args.no_train:
        Wbar={}
        for i in range(1,10):
            for name in auxdict:
                if 'weight' in name:
                    if 'conv' in name:
                        if args.passive_conv:
                            (d1,d2,d3,d4) = model.state_dict()[name].shape
                            Wbar[name] = model.state_dict()[name]
                            for i in range(d1):
                                for j in range (d2):
                                    Wbar[name][i,j,:,:] = model.state_dict()[name][i,j,:,:]*pc.checkindexes(model.state_dict()[name][i,j,:,:],-0.05,4.74)
                    else:
                        Wbar[name] = model.state_dict()[name]*pc.checkindexes(model.state_dict()[name],-0.05,4.74)
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch,Wbar)
                test(model, device, test_loader,Wbar)
                scheduler.step()

    for name in auxdict:
        model.state_dict()[name]=Wbar[name]
    
    if args.save_model:
        if args.passive_conv:
            torch.save(model.state_dict(), "mnist_cnn_robust.pt")
        else:
            torch.save(model.state_dict(), "mnist_fcnn_robust.pt")

    if args.passive_conv:
        auxdict2 = torch.load("mnist_cnn_robust.pt")
        auxmodel2 = Net().to(device)
        auxmodel2.load_state_dict(auxdict2)
    else:
        auxdict2 = torch.load("mnist_fcnn_robust.pt")
        auxmodel2 = Net().to(device)
        auxmodel2.load_state_dict(auxdict2)            

    print("Robust Model Test Errors to Attacks\n")

    test(auxmodel2,device,test_loader)
    
    for i in epsilon: 

        fgm_adv = fgm.fast_gradient_method(auxmodel2,test_data,i,2,clip_min=-0.4242,clip_max=2.8215)
        test_attack(auxmodel2,device,fgm_adv,labels,"FGM",i) 

        pgd_adv = pgd.projected_gradient_descent(auxmodel2,test_data,i,0.1,10,2,clip_min=-0.4242,clip_max=2.8215)
        test_attack(auxmodel2,device,pgd_adv,labels,"PGD",i)

if __name__ == '__main__':
    main()
