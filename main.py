from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import passivitycheck as pc
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import attacks as att


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


def train(args, model, device, train_loader, optimizer, epoch, gamma=0, refmat=None,L=0):
    
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
                            W_bar[name] = refmat[name].clone().detach()
                            loss += torch.mul(torch.mul(torch.sum((param-W_bar[name])**2),L),gamma)
                    else:
                        W_bar[name] =refmat[name].clone().detach()
                        loss += torch.mul(torch.mul(torch.sum((param-W_bar[name])**2),L),gamma)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),100*batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args,model, device, test_loader,gamma=0,refmat=None,L=0):
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
                        if 'conv' in name:
                            if args.passive_conv:
                                W_bar[name] = refmat[name].clone().detach()
                                test_loss += torch.mul(torch.mul(torch.sum((param-W_bar[name])**2),L),gamma)
                        else:
                            W_bar[name] = refmat[name].clone().detach()
                            test_loss += torch.mul(torch.mul(torch.sum((param-W_bar[name])**2),L),gamma)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_attack(model, device, test_loader,attack,attack_type,epsilon,gamma=0,refmat=None,L=0):
    model.eval()
    test_loss = 0
    correct = 0
    W_bar ={}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(attack)
            test_loss += F.nll_loss(output[batch_idx*len(data):(batch_idx+1)*len(data),:], target,reduction='sum').item()
            if refmat is not None:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        if 'conv' in name:
                            if args.passive_conv:
                                W_bar[name] = refmat[name].clone().detach()
                                test_loss += torch.mul(torch.mul(torch.sum((param-W_bar[name])**2),L),gamma)
                        else:
                            W_bar[name] = refmat[name].clone().detach()
                            test_loss += torch.mul(torch.mul(torch.sum((param-W_bar[name])**2),L.detach().numpy()),gamma)
            pred = output[batch_idx*len(data):(batch_idx+1)*len(data),:].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set loss to {} attack (strength {}): {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(attack_type,epsilon,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

    epsilon = [0.1, 0.2, 0.3, 0.5]
    
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
#      transforms.Normalize((0.1307,), (0.3081,))
        transforms.Normalize((0,), (1,))
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
            test(args,model, device, test_loader)
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

    gamma=torch.tensor(0.1)

    Wbar=None
    L=torch.tensor(0)
    if not args.no_train:
        Wbar={}
        for i in range(1,10):
            for name in auxdict:
                if 'weight' in name:
                    if 'conv' in name:
                        if args.passive_conv:
                            (d1,d2,d3,d4) = model.state_dict()[name].shape
                            for i in range(d1):
                                for j in range (d2):
                                    Wbar[name][i,j,:,:] = model.state_dict()[name][i,j,:,:]*pc.checkindexes(model.state_dict()[name][i,j,:,:],-0.05,4.74)
                                    L = L+torch.linalg.matrix_norm(Wbar[name][i,j,:,:]-model.state_dict()[name][i,j,:,:])
                    else:
                        Wbar[name] = model.state_dict()[name]*pc.checkindexes(model.state_dict()[name],-0.05,4.74)
                        L = L+torch.linalg.matrix_norm(Wbar[name]-model.state_dict()[name])
            print(L)
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch,gamma,Wbar,L)
                test(args,model, device, test_loader,gamma,Wbar,L)
                scheduler.step()
    
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

    print("Vanilla Model Test Errors to Attacks\n")
    test(args,auxmodel,device,test_loader)
  
    for i in epsilon:
        fgm_adv = att.fgm(auxmodel,test_data,i,2)
        test_attack(auxmodel,device,test_loader,fgm_adv,"FGM",i,gamma,Wbar,L)
    
        pgd_adv = att.pgd(auxmodel,test_data,i,0.1,10,2)
        test_attack(auxmodel,device,test_loader,pgd_adv,"PGD",i,gamma,Wbar,L)

    print("Robust Model Test Errors to Attacks\n")
    test(args,auxmodel2,device,test_loader)
    
    for i in epsilon: 

        fgm_adv = att.fgm(auxmodel2,test_data,i,2)
        test_attack(auxmodel2,device,test_loader,fgm_adv,"FGM",i,gamma,Wbar,L) 

        pgd_adv = att.pgd(auxmodel2,test_data,i,0.1,10,2)
        test_attack(auxmodel2,device,test_loader,pgd_adv,"PGD",i,gamma,Wbar,L)

if __name__ == '__main__':
    main()
