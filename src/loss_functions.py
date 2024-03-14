import torch


def cross_entropy(all_token_logits, ground_truth_tokens, masked_batch_bool):
    # Get the probability distributions of the masked token prediction
    masked_token_distributions = all_token_logits[masked_batch_bool]
    # Get the actual token indices of the masked tokens
    masked_actual_token_indices = ground_truth_tokens[masked_batch_bool]
    # X-entropy loss using actual token indices and predicted token distributions
    loss = torch.nn.functional.cross_entropy(
        masked_token_distributions,
        masked_actual_token_indices,
        reduction="mean",
    )
    return loss


def negative_log_likelihood(all_token_logits, ground_truth_tokens, masked_batch_bool):
    all_token_likelihoods = torch.nn.functional.softmax(all_token_logits, dim=-1)
    actual_token_likelihoods = all_token_likelihoods.gather(
        -1, ground_truth_tokens.unsqueeze(-1)
    ).squeeze()
    masked_actual_token_likelihoods = actual_token_likelihoods[masked_batch_bool]
    log_likelihood = torch.neg(torch.log(masked_actual_token_likelihoods))
    loss = log_likelihood.mean()
    return loss
