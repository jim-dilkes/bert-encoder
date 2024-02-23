import torch


def cross_entropy(all_token_likelihoods, masked_batch, masked_batch_bool):
    masked_all_token_likelihoods = all_token_likelihoods[masked_batch_bool]
    masked_actual_token_indices = masked_batch[masked_batch_bool]
    loss = torch.nn.functional.cross_entropy(
        masked_all_token_likelihoods,
        masked_actual_token_indices,
        reduction="mean",
    )
    return loss


def negative_log_likelihood(all_token_likelihoods, masked_batch, masked_batch_bool):
    actual_token_likelihoods = all_token_likelihoods.gather(
        -1, masked_batch.unsqueeze(-1)
    ).squeeze()
    masked_actual_token_likelihoods = actual_token_likelihoods[masked_batch_bool]
    log_likelihood = torch.neg(torch.log(masked_actual_token_likelihoods))
    loss = log_likelihood.mean()
    return loss
