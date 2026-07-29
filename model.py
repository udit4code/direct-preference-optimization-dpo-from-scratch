"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
def log_softmax(logits, axis=-1):
    # Step 1 : Subtract max for numerical stability
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    # Step 2 : Compute log(sum(exp(.)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    return shifted - log_sum_exp

# Step 2 - softmax
def softmax(logits, axis=-1):
    # Step 1 : Shift logits for numerical stability
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    # Step 2 : Exponentiate
    exp_logits = np.exp(shifted)
    # Step 3 : Normalize
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    """
    Args:
        log_probs: (B, T, V)
        token_ids: (B, T)

    Returns:
        (B, T) containing the log-probability of the observed token
        at each position.
    """
    # Step 1 : Add a singleton dimension, so for token_ids, we go from (B, T) to (B, T, 1)
    indices = token_ids[..., None]   
    # After 1st Step, shape of indices = (B, T, 1)   
    # Step 2 : Gather along the vocabulary axis         
    gathered = np.take_along_axis(log_probs, indices, axis=-1)
    # Each (b, t) now contains the single log-probability corresponding to token_ids[b, t].
    # That's why, till now, gathered shape is (B, T, 1)
    # Step 3: Remove the singleton dimension. So, now, Final shape: (B, T)
    return gathered.squeeze(-1)

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    """
    Args:
        token_logprobs: (B, T)
        mask: (B, T) binary or bool

    Returns:
        (B,) sequence log-probabilities
    """
    return np.sum(token_logprobs * mask, axis=-1)

# Step 5 - init_policy_params (not yet solved)
# TODO: implement

# Step 6 - policy_token_logits (not yet solved)
# TODO: implement

# Step 7 - policy_sequence_logprob (not yet solved)
# TODO: implement

# Step 8 - sequence_logprob_grad (not yet solved)
# TODO: implement

# Step 9 - bradley_terry_loss (not yet solved)
# TODO: implement

# Step 10 - reward_accuracy (not yet solved)
# TODO: implement

# Step 11 - build_preference_pairs (not yet solved)
# TODO: implement

# Step 12 - sample_preference_batch (not yet solved)
# TODO: implement

# Step 13 - freeze_reference_logprobs (not yet solved)
# TODO: implement

# Step 14 - policy_reference_logratio (not yet solved)
# TODO: implement

# Step 15 - dpo_pair_margin (not yet solved)
# TODO: implement

# Step 16 - dpo_loss (not yet solved)
# TODO: implement

# Step 17 - dpo_loss_grad (not yet solved)
# TODO: implement

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

