import os
import torch.optim as optim
from tqdm import tqdm
from adversarial_attack_defense_power_system.dataset_loader import PMUEventDataset
from adversarial_attack_defense_power_system.classifiers.models import *
from adversarial_attack_defense_power_system.classifiers.evaluation import evaluation


def model_train(interconnection='b', model_name='resnet50', device=torch.device("cpu")):
    print(f'==> Loading dataset, interconnection: {interconnection}')
    if interconnection == 'c':
        batch_size = 8
        lr = 1e-4
    else:
        batch_size = 32
        lr = 5e-4
    trainset = PMUEventDataset(interconnection=interconnection, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = PMUEventDataset(interconnection=interconnection, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

    print(f'==> Building model, model_name: {model_name}')
    if model_name == 'resnet18':
        net = ResNet18()
    elif model_name == 'resnet50':
        net = ResNet50()
    elif model_name == 'resnet152':
        net = ResNet152()
    elif model_name == 'densenet121':
        net = DenseNet121()
    elif model_name == 'efficientnet':
        net = EfficientNetB0()
    elif model_name == 'vgg13':
        net = VGG('VGG13')
    elif model_name == 'mobilenet_v2':
        net = MobileNetV2()
    else:
        print(f"Invalid model_name: {model_name}!")
        return -1
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Start training
    best_acc = 0
    for epoch in range(200):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            _, target_label = targets.max(1)
            correct += predicted.eq(target_label).sum().item()
        print(f"Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
        acc = 100. * correct / total
        # if acc > best_acc and epoch >= 180:
        #     # Save checkpoint.
        #     state = {
        #         'net': net.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch,
        #     }
        #     print('Saving trained model..')
        #     if not os.path.isdir(f'../../weights/classifier/ic_{interconnection}/{model_name}'):
        #         os.makedirs(f'../../weights/classifier/ic_{interconnection}/{model_name}', exist_ok=True)
        #     torch.save(state, f'../../weights/classifier/ic_{interconnection}/{model_name}/ckpt.pth')
        #     best_acc = acc
        #     evaluation(testloader, net, device)
    # Save checkpoint.
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    print('Saving trained model..')
    if not os.path.isdir(f'../../weights/classifier/ic_{interconnection}/{model_name}'):
        os.makedirs(f'../../weights/classifier/ic_{interconnection}/{model_name}', exist_ok=True)
    torch.save(state, f'../../weights/classifier/ic_{interconnection}/{model_name}/ckpt.pth')
    test_acc = evaluation(testloader, net, device)
    print(f"Finished Training, test acc: {test_acc}")
    return 0


if __name__ == '__main__':
    # Set up the device
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    interconnections = ['b', 'c']
    model_names = ['vgg13', 'mobilenet_v2', 'efficientnet', 'densenet121', 'resnet18', 'resnet50']

    for interconnection in interconnections[:1]:
        for model_name in model_names[5:]:
            model_train(interconnection=interconnection, model_name=model_name, device=device)
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
            if device == torch.device('mps'):
                torch.mps.empty_cache()
