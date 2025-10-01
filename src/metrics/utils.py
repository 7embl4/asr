import torch

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    '''
    Calculates CER metric for a single example of predicted and target texts
    
    Params:
    - pred_text: list of predicted tokens
    - target_text: list of real tokens
    Returns:
    - CER metric (from 0.0 to 1.0)
    '''
    N, M = len(target_text), len(predicted_text)
    cer_mat = torch.zeros(N + 1, M + 1, dtype=torch.int64)
    cer_mat[0, :] = torch.arange(M + 1).to(dtype=torch.int64)
    cer_mat[:, 0] = torch.arange(N + 1).to(dtype=torch.int64)
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = int(target_text[i - 1] != predicted_text[j - 1])
            cer_mat[i, j] = min(
                cer_mat[i, j - 1] + cost,
                cer_mat[i - 1, j] + cost,
                cer_mat[i - 1, j - 1] + cost
            )

    cer = cer_mat[-1, -1] / N
    return cer

def calc_wer(target_text, predicted_text) -> float:
    '''
    Calculates WER metric for a single example of predicted and target texts
    
    Params:
    - target_text: string
    - predicted_text: string
    Returns:
    - WER metric (from 0.0 to 1.0)
    '''
    target_tokens = target_text.split()
    pred_tokens = predicted_text.split()
    N, M = len(target_tokens), len(pred_tokens)
    wer_mat = torch.zeros(N + 1, M + 1, dtype=torch.int64)
    wer_mat[0, :] = torch.arange(M + 1).to(dtype=torch.int64)
    wer_mat[:, 0] = torch.arange(N + 1).to(dtype=torch.int64)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = int(target_tokens[i - 1] != pred_tokens[j - 1])
            wer_mat[i, j] = min(
                wer_mat[i, j - 1] + cost,
                wer_mat[i - 1, j] + cost,
                wer_mat[i - 1, j - 1] + cost
            )

    wer = wer_mat[-1, -1] / N
    return wer
