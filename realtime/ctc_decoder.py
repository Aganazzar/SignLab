# ctc_decoder.py
import numpy as np

def greedy_ctc_decode(preds, blank=0):
    """
    preds: (seq_len,) numpy array of predictions
    blank: blank token index
    returns: list of predicted indices (without blanks/duplicates)
    """
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    
    prev = -1
    decoded = []
    for p in preds:
        if p != prev and p != blank:
            decoded.append(int(p))
        prev = p
    return decoded
