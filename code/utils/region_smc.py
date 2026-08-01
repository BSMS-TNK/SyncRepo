import torch


@torch.no_grad()
def blend_region_wise_probabilities(
    sam_teacher_prob,
    unet_teacher_prob,
    unet_student_prob,
    eps=1e-6,
):
    '''Blend aligned `(B, C, H, W)` teacher probabilities per pixel.

    The returned confidence and alpha tensors keep the spatial dimensions and
    reduce only the class channel, so their shape is `(B, 1, H, W)`.
    `alpha_map` is the local U-Net weight; `1 - alpha_map` is the MedSAM weight.
    '''
    unet_local_conf = unet_teacher_prob.max(dim=1, keepdim=True).values
    sam_local_conf = sam_teacher_prob.max(dim=1, keepdim=True).values

    # Sum class-distribution differences while retaining a singleton channel.
    self_conf = 1.0 - 0.5 * (
        unet_teacher_prob - unet_student_prob
    ).abs().sum(dim=1, keepdim=True)
    mutual_conf = 1.0 - 0.5 * (
        unet_teacher_prob - sam_teacher_prob
    ).abs().sum(dim=1, keepdim=True)
    self_conf = self_conf.clamp(0.0, 1.0)
    mutual_conf = mutual_conf.clamp(0.0, 1.0)

    unet_score = unet_local_conf * self_conf
    sam_score = sam_local_conf
    relative_alpha = unet_score / (unet_score + sam_score + eps)

    # Agreement makes model choice less important; disagreement activates the
    # relative reliability score so either network can dominate locally.
    alpha_map = (
        mutual_conf * 0.5
        + (1.0 - mutual_conf) * relative_alpha
    ).clamp(0.0, 1.0)

    blended_prob = (
        alpha_map * unet_teacher_prob
        + (1.0 - alpha_map) * sam_teacher_prob
    )
    return blended_prob, self_conf, mutual_conf, alpha_map
