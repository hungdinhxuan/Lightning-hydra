#!/usr/bin/env python
# scripts/score_file_to_eer.py

import sys
import os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, precision_score, det_curve

def eval_to_score_file(score_file, cm_key_file):
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    cm_data.columns = ['filename', 'subset', 'label']
    submission_scores = pandas.read_csv(
        score_file, sep=' ', header=None, skipinitialspace=True)
    submission_scores.columns = ['filename', 'spoof', 'score']
    
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