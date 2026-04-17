import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, steps=10):
    was_training = model.training
    model.eval()

    device = images.device
    images = images.detach().to(device)
    labels = labels.detach().to(device)

    images_adv = images.clone().detach().requires_grad_(True)

    for _ in range(steps):
        _, logits = model(images_adv)
        # ensure logits and labels have matching shapes (B,)
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1),
            labels.float().view(-1),
        )

        grad = torch.autograd.grad(loss, images_adv)[0]
        images_adv = images_adv + alpha * grad.sign()

        # Project onto epsilon-ball only (no [0,1] clamp for normalized inputs)
        eta = torch.clamp(images_adv - images, -epsilon, epsilon)
        images_adv = (images + eta).detach().requires_grad_(True)

    if was_training:
        model.train()

    return images_adv

