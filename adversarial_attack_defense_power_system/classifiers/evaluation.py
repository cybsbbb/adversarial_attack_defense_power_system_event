import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset
from torcheval.metrics.functional import multiclass_f1_score
from adversarial_attack_defense_power_system.classifiers.load_classifier import *
from adversarial_attack_defense_power_system.dataset_loader import PMUEventDataset


def evaluation_numpy(x, y, net, device='cpu'):
    print("Evaluation on numpy...")
    # x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            _, target_label = targets.max(1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(target_label.cpu().numpy())
    y_true_tensor = torch.Tensor(y_true)
    y_pred_tensor = torch.Tensor(y_pred)
    f1score = multiclass_f1_score(y_pred_tensor, y_true_tensor, num_classes=4).detach().cpu().item()
    print(f"f1score: {f1score:.3f}")
    return f1score


def evaluation(testloader, net, device):
    # Evaluating on test and saving trained model
    print("Evaluation on test set...")
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            _, target_label = targets.max(1)
            correct += predicted.eq(target_label).sum().item()
    acc = 100. * correct / total
    print(f"Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
    return acc


def model_evaluate(interconnection='b', model_name='resnet50', device=torch.device("cpu")):
    # Load the dataset
    testset = PMUEventDataset(interconnection=interconnection, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Load the trained classifier
    net = load_classifier(interconnection=interconnection, model_name=model_name, device=device)

    # Evaluate on the test loader
    acc = evaluation(testloader, net, device)
    print(f"Test data accuracy, interconnection: {interconnection}, model_name: {model_name}, accuracy: {acc}")
    return acc


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

    for interconnection in interconnections[:]:
        for model_name in model_names[:]:
            model_evaluate(interconnection=interconnection, model_name=model_name, device=device)
            torch.cuda.empty_cache()
