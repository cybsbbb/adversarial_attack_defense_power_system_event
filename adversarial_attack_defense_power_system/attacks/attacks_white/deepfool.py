""" The DeepFool attack """
import copy
import torch
import numpy as np
import collections
from torch.autograd import Variable


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def deepfool_attack(sample_x, sample_y, net, attack_config=None):
    # Original Wrong Prediction, skip the attack.
    init_pred = net(sample_x).max(1, keepdim=True)[1]
    if init_pred.item() != sample_y.max(1, keepdim=True)[1].item():
        return sample_x, True

    # Get attack configs
    if attack_config is None:
        num_classes = 4
        overshoot = 0.005
        max_iter = 40
    else:
        num_classes = attack_config['num_classes']
        overshoot = attack_config['overshoot']
        max_iter = attack_config['max_iter']
    # Start attack
    image = sample_x[0]

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot
    if is_cuda:
        pert_image = image + torch.from_numpy(r_tot).cuda()
    else:
        pert_image = image + torch.from_numpy(r_tot)

    sample_x_adv = pert_image

    # Check for success
    success = False
    if net(sample_x_adv).max(1, keepdim=True)[1].item() != sample_y.max(1, keepdim=True)[1].item():
        success = True

    return sample_x_adv, success
