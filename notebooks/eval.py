import os
import sys
import os.path
import numpy as np
import pandas as pd
import eval_metrics_DF as em

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, precision_score, det_curve


def eval_to_score_file(score_file, cm_key_file):
    # CM key file is the metadata file that contains the ground truth labels for the eval set
    # score file is the output of the system that contains the scores for the eval set
    # phase is the phase of the eval set (dev or eval)

    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    cm_data.columns = ['filename', 'subset', 'label']
    submission_scores = pandas.read_csv(
        score_file, sep=' ', header=None, skipinitialspace=True)
    submission_scores.columns = ['filename', 'spoof', 'score']
    # check here for progress vs eval set
    cm_scores = submission_scores.merge(
        cm_data, left_on='filename', right_on='filename', how='inner')
    # cm_scores.head()
    #  0       1_x   1_y      2      3
    #  a.wav  1.234   eval   Music   spoof
    bona_cm = cm_scores[cm_scores['label'] == 'bonafide']['score'].values
    spoof_cm = cm_scores[cm_scores['label'] == 'spoof']['score'].values

    # min_score of cm_scores['score']
    print("min_score: ", min(cm_scores['score']))
    # max_score of cm_scores['score']
    print("max_score: ", max(cm_scores['score']))

    eer_cm, th = em.compute_eer(bona_cm, spoof_cm)
    out_data = "eer: {}\tthreshold: {}\n".format(100*eer_cm, th)
    print(out_data)
    return eer_cm


full_df = pd.read_csv(
    "/nvme1/hungdx/Lightning-hydra/data/0_large-corpus/AIHub/protocol_for_cm.txt", sep=" ", header=None)

full_df.columns = ["utt", "subset", "label"]

# Filter with utt startwith Elevenlabs only
full_df = full_df[full_df["utt"].str.startswith("Elevenlabs")]

# filter with subset = "eval"
full_df = full_df[full_df["subset"] == "eval"]
score_file = "/nvme1/hungdx/Lightning-hydra/logs/eval/cnsl/aihub_new/AIHUB_new_lora_xlsr_conformertcm_mdt_large_corpus_s202412_v2.txt"
pred_df = pd.read_csv(score_file, sep=" ", header=None)
print(score_file)
pred_df.columns = ["utt", "spoof", "score"]
# pred_df = pred_df.drop_duplicates(subset=['utt'])


# if spoof < score, then bonafide, else spoof
pred_df['pred'] = pred_df.apply(
    lambda x: 'bonafide' if x['spoof'] < x['score'] else 'spoof', axis=1)

# merge eval_df and pred_df on utt
res_df = pd.merge(full_df, pred_df, on='utt')

print("Accuracy: {:.2f}".format(
    accuracy_score(res_df["label"], res_df["pred"])*100))
print("\n")

# Filter fail
