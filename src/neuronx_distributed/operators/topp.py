import torch
from nkilib.core.cumsum.cumsum import cumsum as nki_cumsum
from bitonic_nki_kernels import bitonic_nki_topp

def get_topp_implementation(top_p_kernel_enabled=False, top_p_kernel_implementation='', on_cpu=False, dim=1, IGNORED_LOGITS_VALUE=-3000):
    if top_p_kernel_enabled and top_p_kernel_implementation == 'bitonic':
        # The bitonic kernel expects the input to be (B, K), so dim is always 1.
        def topp_bitonic(tensor, top_p):
            return bitonic_nki_topp(tensor, top_p)

        return topp_bitonic
    elif not top_p_kernel_enabled:
        def topp(tensor, top_p):
            return baseline_topp(tensor, top_p, on_cpu=on_cpu, dim=dim, IGNORED_LOGITS_VALUE=IGNORED_LOGITS_VALUE)

        return topp
    else:
        raise ValueError(f"Unsupported topp kernel implementation, check the on_device_sampling_config.")

def baseline_topp(tensor, top_p, on_cpu=False, dim=1, IGNORED_LOGITS_VALUE=-3000):
    probs_soft_max = torch.nn.functional.softmax(input=tensor, dim=dim)
    probs_cumsum = _cumsum(
        tensor_in=probs_soft_max, dim=dim, on_cpu=on_cpu
    )
    top_p = torch.max(torch.min(probs_cumsum), top_p)
    top_p_mask = torch.greater(probs_cumsum, top_p).index_fill_(
        dim, torch.tensor([0], device=top_p.device), False
    )  # need to keep at least one token
    out = tensor.masked_fill(top_p_mask, IGNORED_LOGITS_VALUE)
    return out

def _cumsum(tensor_in, dim, on_cpu=False):
    if on_cpu:
        return torch.cumsum(tensor_in, dim=dim)
    init_shape_len = len(tensor_in.shape)
    cumsum_dim = dim % init_shape_len
    last_dim = init_shape_len - 1
    is_transposed = False
    if cumsum_dim != last_dim:
        tensor_in = torch.transpose(tensor_in, cumsum_dim, last_dim)
        is_transposed = True
    init_shape = tensor_in.shape
    cumsum_len = init_shape[last_dim]
    # Prioritize nki kernel for float dtype, then matmul cumsum if not input is not float
    if torch.is_floating_point(tensor_in):
        tensor_in = tensor_in.view(-1, cumsum_len)
        output = nki_cumsum(tensor_in, axis=1)
        output = output.view(init_shape)
        if is_transposed:
            output = torch.transpose(output, cumsum_dim, last_dim)
        return output
    else:
        triu = torch.triu(
            torch.ones(
                cumsum_len,
                cumsum_len,
                dtype=tensor_in.dtype,
                device=tensor_in.device,
            )
        )
        output = tensor_in @ triu
        if is_transposed:
            output = torch.transpose(output, cumsum_dim, last_dim)
        return output