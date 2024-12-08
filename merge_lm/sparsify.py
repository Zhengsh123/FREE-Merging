import torch


def magnitude(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor
    if len(tensor.shape) == 1:
        # rank=1
        return tensor
    
    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1
    return tensor * mask


def bernoulli(
    tensor: torch.Tensor, 
    density: float, # 1 - mask_rate (probability of drawing "1")
    rescale: bool = True
) -> torch.Tensor:
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if len(tensor.shape) == 1:
        # rank=1
        return tensor

    # mask = 1 - torch.bernoulli(
    #     torch.full_like(input=tensor, fill_value=1 - density)
    # )
    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density).float()
    )

    res = tensor * mask
    if rescale:
        res *= 1 / density
    return res

def svd(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if len(tensor.shape) == 1:
        # rank=1
        return tensor
    
    device=tensor.device
    tensor=tensor.to('cpu')
    driver = None
    if tensor.is_cuda:
        driver = 'gesvda'
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
    new_rank = int(density * len(S))
    U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
    res = U @ torch.diag(S) @ Vh
    res=res.to(device)
    return res

def top(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if len(tensor.shape) == 0:
            return tensor
    else:
        top_k_int = int(tensor.shape[-1] * density)
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
        mask = torch.zeros(tensor.shape).to(tensor.device)
        mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)
        return mask * tensor*8