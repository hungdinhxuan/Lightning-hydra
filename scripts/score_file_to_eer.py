#!/usr/bin/env python
# scripts/score_file_to_eer.py

import sys
import os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, precision_score, det_curve


def parse_protocol_line(line: str):
    """
    Parse protocol line handling paths with spaces and quotes.
    Format: <path> <subset> <label>
    
    Returns: (file_id, subset, label) or None if invalid
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    # Check if path is quoted
    if line.startswith('"') or line.startswith("'"):
        quote_char = line[0]
        close_quote_idx = line.find(quote_char, 1)
        if close_quote_idx == -1:
            return None
        
        file_id = line[1:close_quote_idx]
        remainder = line[close_quote_idx + 1:].strip()
        parts = remainder.split()
        
        if len(parts) != 2:
            return None
        
        subset, label = parts
        return file_id, subset, label
    
    # Parse from right: last 2 tokens are subset and label
    parts = line.rsplit(maxsplit=2)
    if len(parts) != 3:
        return None
    
    file_id, subset, label = parts
    return file_id, subset, label


def parse_score_line(line: str):
    """
    Parse score line handling paths with spaces and quotes.
    Format: <path> <score1> <score2> or <path> <score>
    
    Returns: (file_id, score1, score2) or None if invalid
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    # Check if path is quoted
    if line.startswith('"') or line.startswith("'"):
        quote_char = line[0]
        close_quote_idx = line.find(quote_char, 1)
        if close_quote_idx == -1:
            return None
        
        file_id = line[1:close_quote_idx]
        remainder = line[close_quote_idx + 1:].strip()
        parts = remainder.split()
        
        try:
            if len(parts) >= 2:
                score1 = float(parts[0])
                score2 = float(parts[1])
                return file_id, score1, score2
            elif len(parts) == 1:
                score = float(parts[0])
                return file_id, score, score
        except ValueError:
            return None
        
        return None
    
    # Parse from right: last 2 (or 1) numbers are scores
    parts = line.split()
    try:
        if len(parts) >= 3:
            score1 = float(parts[-2])
            score2 = float(parts[-1])
            file_id = ' '.join(parts[:-2])
            return file_id, score1, score2
        elif len(parts) >= 2:
            score = float(parts[-1])
            file_id = ' '.join(parts[:-1])
            return file_id, score, score
    except ValueError:
        return None
    
    return None


def eval_to_score_file(score_file, cm_key_file):
    # Read protocol file with proper parsing
    cm_data_list = []
    with open(cm_key_file, 'r') as f:
        for line in f:
            parsed = parse_protocol_line(line)
            if parsed:
                filename, subset, label = parsed
                cm_data_list.append({'filename': filename, 'subset': subset, 'label': label})
    
    cm_data = pandas.DataFrame(cm_data_list)
    
    # Read score file with proper parsing to handle paths with spaces and quotes
    score_data = []
    with open(score_file, 'r') as f:
        for line in f:
            parsed = parse_score_line(line)
            if parsed:
                filename, spoof_score, score = parsed
                score_data.append({'filename': filename, 'spoof': spoof_score, 'score': score})
    
    submission_scores = pandas.DataFrame(score_data)
    
    cm_scores = submission_scores.merge(
        cm_data, left_on='filename', right_on='filename', how='inner')
    
    cm_scores['pred'] = cm_scores.apply(
        lambda x: 'bonafide' if x['spoof'] < x['score'] else 'spoof', axis=1)
    
    accuracy = accuracy_score(cm_scores['label'], cm_scores['pred']) * 100

    bona_cm = cm_scores[cm_scores['label'] == 'bonafide']['score'].values
    spoof_cm = cm_scores[cm_scores['label'] == 'spoof']['score'].values
    
    eer_cm, th = em.compute_eer(bona_cm, spoof_cm)

    return min(cm_scores['score']), max(cm_scores['score']), th, eer_cm * 100, accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("CHECK: invalid input arguments. Please read the instruction below:")
        print("Usage: {} <score_file> <protocol_file>".format(sys.argv[0]))
        exit(1)

    submit_file = sys.argv[1]
    cm_key_file = sys.argv[2]
    
    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)

    if not os.path.isfile(cm_key_file):
        print("%s doesn't exist" % (cm_key_file))
        exit(1)

    min_score, max_score, threshold, eer_cm, accuracy = eval_to_score_file(submit_file, cm_key_file)
    
    # Print in a format that can be easily parsed by the bash script
    print(f"{min_score:.6f} {max_score:.6f} {threshold:.6f} {eer_cm:.6f} {accuracy:.6f}")