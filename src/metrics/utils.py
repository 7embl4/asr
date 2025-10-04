import torch
import editdistance

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
    assert len(target_text) != 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)

def calc_wer(target_text, predicted_text) -> float:
    '''
    Calculates WER metric for a single example of predicted and target texts
    
    Params:
    - target_text: string
    - predicted_text: string
    Returns:
    - WER metric (from 0.0 to 1.0)
    '''
    assert len(target_text) != 0
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())
