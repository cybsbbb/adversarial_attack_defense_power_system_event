from pathlib import Path
from adversarial_attack_defense_power_system.classifiers.models import *


def load_classifier(interconnection='b', model_name='resnet50', device=torch.device("cpu")):
    print(f"Loading pretrained model, interconnection: {interconnection}, model_name: {model_name}.")

    # Build network
    if model_name == 'resnet18':
        net = ResNet18()
    elif model_name == 'resnet50':
        net = ResNet50()
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

    # Load checkpoint
    script_path = Path(__file__).resolve().parent
    checkpoint_path = f'{script_path}/../../weights/classifier/ic_{interconnection}/{model_name}/ckpt.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['net'])

    return net


if __name__ == '__main__':
    # Setup the device
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    net = load_classifier(interconnection='b',
                          model_name='resnet50',
                          device=torch.device("cpu"))
