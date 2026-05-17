import torch

try:
    from .RTE_ATFPS_func import correction, funcDinv, func_inflow_Dirichlet, restrict_basis, squeeze_basis
except ImportError:
    from RTE_ATFPS_func import correction, funcDinv, func_inflow_Dirichlet, restrict_basis, squeeze_basis


def atfps_operator(value, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device):
    return func_inflow_Dirichlet(value, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)


def selected_loss(value, target, VecSize, loss_fn):
    value_selected, _ = squeeze_basis(value, VecSize)
    target_selected, _ = squeeze_basis(target, VecSize)
    return loss_fn(value_selected, target_selected)


def corrected_candidate(candidate, residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device):
    _, residual_unselected = restrict_basis(residual, VecSize, M, mesh_size, mesh_size, device)
    _, candidate_corrected, _ = correction(
        candidate,
        residual_unselected,
        I2A,
        fsmLRBTC,
        MLRBT,
        VecSize,
        M,
        mesh_size,
        mesh_size,
        device,
    )
    return candidate_corrected


def apply_preconditioner_form(reference, update, form):
    if form == "direct":
        return update
    if form == "residual":
        return reference + update
    raise ValueError(f"preconditioner_form {form} not supported")


def mgnet_preconditioner(
    residual,
    model,
    a_list,
    inv_a_list,
    I2A,
    fsmLRBTC,
    MLRBT,
    VecSize,
    M,
    mesh_size,
    device,
    mode="raw",
    form="direct",
):
    if mode == "raw":
        reference = residual
    elif mode == "born":
        reference = funcDinv(residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    else:
        raise ValueError(f"preconditioner mode {mode} not supported")

    update = model(reference, a_list, inv_a_list)
    candidate = apply_preconditioner_form(reference, update, form)
    return corrected_candidate(candidate, residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)


def training_loss(
    model,
    batch,
    M,
    mesh_size,
    device,
    loss_fn,
    loss_type="raw",
    preconditioner_form="direct",
    physical_loss_weight=0.1,
):
    Coef, fsmLRBTC, I2A, MLRBT, VecSize, rhs = [tensor.to(device) for tensor in batch]
    a_list, inv_a_list = model.setup(Coef)

    if loss_type == "raw":
        pred = model(rhs, a_list, inv_a_list)
        Apred = atfps_operator(pred, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)
        return selected_loss(Apred, rhs, VecSize, loss_fn)

    if loss_type not in {"born", "born_mixed"}:
        raise ValueError(f"loss_type {loss_type} not supported")

    q = funcDinv(rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    update = model(q, a_list, inv_a_list)
    pred = apply_preconditioner_form(q, update, preconditioner_form)
    pred = corrected_candidate(pred, rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)

    Apred = atfps_operator(pred, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)
    q_pred = funcDinv(Apred, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    born_loss = selected_loss(q_pred, q, VecSize, loss_fn)

    if loss_type == "born":
        return born_loss

    physical_loss = selected_loss(Apred, rhs, VecSize, loss_fn)
    return born_loss + physical_loss_weight * physical_loss


def d_metric_relative_residual(residual, rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device):
    pre_residual = funcDinv(residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    pre_rhs = funcDinv(rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    denom = torch.clamp(torch.linalg.norm(pre_rhs), min=1e-30)
    return (torch.linalg.norm(pre_residual) / denom).item()
