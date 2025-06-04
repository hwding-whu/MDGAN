"""
Loss functions for MDGAN training.
Includes gradient penalty, mutual exclusion loss, and contrastive loss.
"""

import torch


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP training.

    Args:
        discriminator (nn.Module): Discriminator model
        real_samples (torch.Tensor): Real data samples
        fake_samples (torch.Tensor): Generated fake samples
        device (torch.device): Device for computation

    Returns:
        torch.Tensor: Gradient penalty value
    """
    batch_size = real_samples.size(0)

    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, device=device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Calculate discriminator scores for interpolated samples
    d_interpolates = discriminator(interpolates)

    # Create gradient targets
    fake = torch.ones(batch_size, 1, device=device, requires_grad=False)

    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calculate gradient norm and penalty
    gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()

    return gradient_penalty


def cosine_similarity(x1, x2):
    """
    Calculate cosine similarity between two tensors.

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: Absolute cosine similarity values
    """
    # Calculate norms
    x1_norm = torch.norm(x1, dim=1, keepdim=True)
    x2_norm = torch.norm(x2, dim=1, keepdim=True)

    # Prevent division by zero
    x1_norm = torch.clamp(x1_norm, min=1e-8)
    x2_norm = torch.clamp(x2_norm, min=1e-8)

    # Normalize vectors
    x1_normalized = x1 / x1_norm
    x2_normalized = x2 / x2_norm

    # Calculate cosine similarity
    sim = torch.sum(x1_normalized * x2_normalized, dim=1)

    return torch.abs(sim)


def mutual_exclusion_loss(generators_outputs):
    """
    Calculate mutual exclusion loss between multiple generator outputs.
    Encourages generators to produce diverse outputs.

    Args:
        generators_outputs (list): List of generator output tensors

    Returns:
        torch.Tensor: Mutual exclusion loss value
    """
    n = len(generators_outputs)
    if n <= 1:
        return torch.tensor(0.0, device=generators_outputs[0].device)

    loss = torch.tensor(0.0, device=generators_outputs[0].device)
    count = 0

    # Calculate pairwise similarities between all generator outputs
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(generators_outputs[i], generators_outputs[j]).mean()
            loss = loss + sim
            count += 1

    if count > 0:
        return loss / count
    return loss


def contrastive_loss(gen_samples, minority_samples, majority_samples, margin=0.5):
    """
    Calculate contrastive loss to encourage generated samples to be similar to minority
    class and different from majority class.

    Args:
        gen_samples (torch.Tensor): Generated samples
        minority_samples (torch.Tensor): Minority class samples
        majority_samples (torch.Tensor): Majority class samples
        margin (float): Margin for contrastive loss

    Returns:
        torch.Tensor: Contrastive loss value
    """
    batch_size = gen_samples.size(0)

    # Randomly select minority class samples as positive samples
    pos_idx = torch.randint(0, minority_samples.size(0), (batch_size,))
    pos_samples = minority_samples[pos_idx]

    # Randomly select majority class samples as negative samples
    neg_idx = torch.randint(0, majority_samples.size(0), (batch_size,))
    neg_samples = majority_samples[neg_idx]

    # Calculate distances
    dist_pos = torch.sum((gen_samples - pos_samples) ** 2, dim=1)
    dist_neg = torch.sum((gen_samples - neg_samples) ** 2, dim=1)

    # Contrastive loss: max(0, dist_pos - dist_neg + margin)
    loss = torch.clamp(dist_pos - dist_neg + margin, min=0)

    return loss.mean()


def calculate_discriminator_loss(discriminator, real_imgs, fake_imgs, device, lambda_gp):
    """
    Calculate discriminator loss for WGAN-GP.

    Args:
        discriminator (nn.Module): Discriminator model
        real_imgs (torch.Tensor): Real images
        fake_imgs (torch.Tensor): Fake images
        device (torch.device): Device for computation
        lambda_gp (float): Gradient penalty weight

    Returns:
        torch.Tensor: Total discriminator loss
    """
    # WGAN loss
    real_pred = discriminator(real_imgs)
    fake_pred = discriminator(fake_imgs)

    d_loss = -torch.mean(real_pred) + torch.mean(fake_pred)

    # Gradient penalty
    gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)

    # Total loss with gradient penalty
    total_loss = d_loss + lambda_gp * gp

    return total_loss


def calculate_generator_loss(generator, discriminator, minority_tensor, majority_tensor,
                             all_generators, latent_dim, device, lambda_me, lambda_cl, margin):
    """
    Calculate total generator loss including adversarial, mutual exclusion, and contrastive losses.

    Args:
        generator (nn.Module): Current generator
        discriminator (nn.Module): Discriminator for adversarial loss
        minority_tensor (torch.Tensor): Minority class data
        majority_tensor (torch.Tensor): Majority class data
        all_generators (list): All generators for mutual exclusion loss
        latent_dim (int): Latent dimension
        device (torch.device): Device for computation
        lambda_me (float): Mutual exclusion loss weight
        lambda_cl (float): Contrastive loss weight
        margin (float): Margin for contrastive loss

    Returns:
        tuple: (total_loss, adversarial_loss, me_loss, cl_loss)
    """
    batch_size = minority_tensor.size(0) if minority_tensor.size(0) < 32 else 32

    # Generate samples
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_imgs = generator(z)

    # Generate samples for mutual exclusion loss
    all_fake_imgs_for_me = []
    for gen in all_generators:
        if gen != generator:
            with torch.no_grad():
                z_other = torch.randn(batch_size, latent_dim, device=device)
                fake_other = gen(z_other)
                all_fake_imgs_for_me.append(fake_other)
        else:
            all_fake_imgs_for_me.append(fake_imgs)

    # Calculate losses
    me_loss = mutual_exclusion_loss(all_fake_imgs_for_me)

    # Adversarial loss
    fake_preds = discriminator(fake_imgs)
    g_loss_adv = -torch.mean(fake_preds)

    # Contrastive loss
    cl_loss = contrastive_loss(fake_imgs, minority_tensor, majority_tensor, margin)

    # Total generator loss
    g_loss_total = g_loss_adv + lambda_me * me_loss + lambda_cl * cl_loss

    return g_loss_total, g_loss_adv, me_loss, cl_loss