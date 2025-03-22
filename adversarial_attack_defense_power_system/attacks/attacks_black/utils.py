import torch


def get_probs(net, x, y):
    x = torch.tensor(x, dtype=torch.float32, device=next(net.parameters()).device)
    pred = net(x)[0, y].detach().cpu().numpy()
    return pred


def get_label(net, x):
    x = torch.tensor(x, dtype=torch.float32, device=next(net.parameters()).device)
    pred = net(x).max(1, keepdim=True)[1].item()
    return pred
