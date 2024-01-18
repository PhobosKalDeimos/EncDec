import parser_args
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import dataset
from models.encdec_ad.model import EncDecAD
from tqdm import tqdm
from dataset import return_dataloader
from sklearn import metrics
import pickle

args = parser_args.parse_arguments()   

def compute_metrics(anomaly_scores_norm, df_test, y_true, th=None):
    tot_anomalies = y_true.sum()
    sens = list()           # recalls o tpr
    spec = list()
    fpr = list()
    f1 = list()
    f0_1= list()
    prec = list()
    cm_list = list()
    anomlay_indexes_dict = dict()
    acc_with_err = list()
    step = 0.01
    ths = np.arange(0, 1, step)
    if th is None:
        for threshold in tqdm(ths):
            anomalies_pred = anomaly_scores_norm > threshold
            tp = 0                                                          # true positive per quella threshold
            anomaly_indexes = list()
            for index, anomaly_pred in enumerate(anomalies_pred):
                if y_true[index] and anomaly_pred:
                    anomaly_indexes.append(index)
                    tp += 1

            cm_anomaly = np.zeros((2,2))
            n_sample = len(df_test)
            n_not_collision = n_sample - tot_anomalies
            n_detected = anomalies_pred.sum()

            fp = n_detected - tp
            fn = tot_anomalies - tp
            tn = n_not_collision - fp

            cm_anomaly[0, 0] = tn
            cm_anomaly[0, 1] = fp
            cm_anomaly[1, 0] = fn
            cm_anomaly[1, 1] = tp

            cm_list.append(cm_anomaly)
            recall = tp / (tp + fn)
            sens.append(recall)
            fpr.append(1 - tn /(tn + fp))
            precision = tp / (tp + fp)
            prec.append(precision)
            spec.append(tn /(tn + fp))
            f1.append(2 * tp / (2 * tp + fp + fn))
            f0_1.append((1 + 0.1**2) * tp / ((1 + 0.1**2) * tp +  0.1**2*fp + fn))
            cm_anomaly_norm = cm_anomaly.astype('float') / cm_anomaly.sum(axis=1)[:, np.newaxis]
            acc_with_err.append( (np.mean(np.diag(cm_anomaly_norm)), np.std(np.diag(cm_anomaly_norm))) )
            anomlay_indexes_dict[threshold] = anomaly_indexes
        
        f1_max = max(f1)
        f0_1_max = max(f0_1)
        max_index_f1 = f1.index(f1_max)
        max_index_f0_1 = f0_1.index(f0_1_max)
        th_f1_max = max_index_f1 * step
        th_f0_1_max = max_index_f0_1 * step
        logging.info(f"f1: {f1_max} at th: {th_f1_max}")
        logging.info(f"f0.1: {f0_1_max} at th: {th_f0_1_max}")
        logging.info(f"AUC-PR: {metrics.average_precision_score(y_true, anomaly_scores_norm)}")
        logging.info(f"AUC-ROC: {metrics.roc_auc_score(y_true, anomaly_scores_norm)}")
        return sens, fpr, th_f1_max
    else:
        df_anomaly = df_test.loc[np.array(anomaly_scores_norm > th)]
        tp = 0                                                          # true positive per quella threshold
        anomaly_indexes = list()
        for index, anomaly_pred in enumerate(anomalies_pred):
            if y_true[index] and anomaly_pred:
                anomaly_indexes.append(index)
                tp += 1

        cm_anomaly = np.zeros((2,2))
        n_sample = len(df_test)
        n_not_collision = n_sample - tot_anomalies
        n_detected = len(df_anomaly)

        fp = n_detected - tp
        fn = tot_anomalies - tp
        tn = n_not_collision - fp

        cm_anomaly[0, 0] = tn
        cm_anomaly[0, 1] = fp
        cm_anomaly[1, 0] = fn
        cm_anomaly[1, 1] = tp

        f1 = 2 * tp / (2 * tp + fp + fn)
        f0_1 = (1 + 0.1**2) * tp / ((1 + 0.1**2) * tp +  0.1**2*fp + fn)
        logging.info(f"f1: {f1} at th: {th} for the test set")
        logging.info(f"f0.1: {f0_1} at th: {th} for the test set")
    
def plot_hist(anomaly_scores_norm, df_collision, df, plot_filename):
    logging.info(f"Counting the total number of anomalies...")
    index_anomaly = []
    idx = 0
    for _, row in df.iterrows():
        for _, collision_row in df_collision.iterrows():
            if (row['time'] >= collision_row['start']) and (row['time'] <= collision_row['end']):
                index_anomaly.append(idx)
        idx += 1
    y_true = np.zeros_like(anomaly_scores_norm)
    y_true[index_anomaly] = 1
    logging.info(f"Anomalies detected: {int(y_true.sum())}")
    anomaly_values = anomaly_scores_norm[index_anomaly]
    normal_values = np.delete(anomaly_scores_norm, index_anomaly)

    plt.hist(normal_values, bins=30, color="tab:blue", ec="dodgerblue", alpha=0.5, label='Normal')
    plt.hist(anomaly_values, bins=30, color='tab:red', ec="darkred", alpha=0.7, label='Anomalies')

    plt.xlabel('Values')
    plt.ylabel('Occurencies')
    plt.legend(loc='upper right')
    plt.title('Distribution')
    plot_filename = f"{plot_filename}.png"
    plt.savefig(plot_filename)
    plt.show()
    return y_true

def compute_metrics_pak(scores, targets, pa=True, interval=10, k=0):
    """
    :param scores: list or np.array or tensor, anomaly score
    :param targets: list or np.array or tensor, target labels
    :param pa: True/False
    :param interval: threshold search interval
    :param k: PA%K threshold
    :return: results dictionary
    """
    assert len(scores) == len(targets)  # check if the length of scores and labels are equal

    results = {}

    try:
        scores = np.asarray(scores)     # convert scores and targets in numpy arrays
        targets = np.asarray(targets)
    except TypeError:
        scores = np.asarray(scores.cpu())
        targets = np.asarray(targets.cpu())

    precision, recall, threshold = metrics.precision_recall_curve(targets, scores)  # compute precision, recall and F1 score using the precision-recall curve
    f1_score = 2 * precision * recall / (precision + recall + 1e-12)  # compute f1 score

    # Compute metrics without Point-Adjustment
    results['best_f1_wo_pa'] = np.max(f1_score)                     
    results['best_precision_wo_pa'] = precision[np.argmax(f1_score)]
    results['best_recall_wo_pa'] = recall[np.argmax(f1_score)]
    results['prauc_wo_pa'] = metrics.average_precision_score(targets, scores)
    results['auc_wo_pa'] = metrics.roc_auc_score(targets, scores)

    # if PA is true compute metrics with point adjustment
    if pa:
        # find F1 score with optimal threshold of best_f1_wo_pa = best f1 without point_adjustment
        pa_scores = pak(scores, targets, threshold[np.argmax(f1_score)], k)
        results['raw_f1_w_pa'] = metrics.f1_score(targets, pa_scores)
        results['raw_precision_w_pa'] = metrics.precision_score(targets, pa_scores)
        results['raw_recall_w_pa'] = metrics.recall_score(targets, pa_scores)

        # find best F1 score with varying thresholds
        if len(scores) // interval < 1:   # check if the legnth of scores divided by interval is less than 1
            ths = threshold   # use original threshold
        else:
            ths = [threshold[interval*i] for i in range(len(threshold)//interval)]  # create a list of ths containing thresholds at regular intervals determined by the interval parameter

        # iterate through the thresholds and compute F1 scores with point adjustment for each threshold
        pa_f1_scores = [metrics.f1_score(targets, pak(scores, targets, th, k)) for th in tqdm(ths)]
        pa_f1_scores = np.asarray(pa_f1_scores)
        results['best_f1_w_pa'] = np.max(pa_f1_scores)
        results['best_f1_th_w_pa'] = ths[np.argmax(pa_f1_scores)]
        
        pa_scores = pak(scores, targets, ths[np.argmax(pa_f1_scores)], k)
        results['best_precision_w_pa'] = metrics.precision_score(targets, pa_scores)
        results['best_recall_w_pa'] = metrics.recall_score(targets, pa_scores)
        results['pa_f1_scores'] = pa_f1_scores

    return results

def pak(scores, targets, thres, k=20):
    """

    :param scores: anomaly scores
    :param targets: target labels
    :param thres: anomaly threshold
    :param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
    :return: point_adjusted predictions
    """
    scores = np.array(scores)     # convert anomaly scores and threholsd to in numpy array
    thres = np.array(thres)

    predicts = scores > thres     # each element is true if the score is greater than the threshold
    actuals = targets > 0.01      # each elment is true if the corrisponding target label is greather than 0.01

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]   # dentify the starting indices of consecutive sequences of 1s (one_start_idx) and 0s (zero_start_idx) in the actuals array.
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    # If the length of one_start_idx is equal to the length of zero_start_idx + 1, adjust zero_start_idx by appending the length of predicts.
    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    # Iterate through each sequence of 1s and 0s, and if the sum of predicted anomalies
    # in that sequence exceeds the PA%K ratio, set all elements in that sequence to 1.
    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts

def compute_anomaly_scores(model, dataloader, d):
    errors = []
    for x, y in tqdm(dataloader):
        y_preds = []
        if args.device == 'cuda':
            x, y = x.cuda(), y.cuda()
        for p in range(args.prediction_length):
            x_w = x[:, p]
            y_p = model.forward(x_w)
            y_p = y_p[:, y_p.shape[1]-d*(p+1):y_p.shape[1]-d*p]
            y_preds.append(y_p)
        y_preds = torch.cat(y_preds, dim=1)
        e = torch.abs(y - y_preds)
        errors.append(e)
    errors = torch.cat(errors)
    anomaly_scores = model.anomaly_scorer.forward(errors.mean(dim=1))
    anomaly_scores_norm = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    return anomaly_scores_norm



def compute_anomaly_scores_embedded(model, ts, d):
    model.execute(ts=ts, data_output= f"./output/scores/f{args.frequency}_")
    anomaly_scores = model.get_points_anomaly_score()
    anomaly_scores_norm = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    return anomaly_scores_norm




def evaluation(model, pipeline):
    df_collision, X_collisions, df_test = dataset.read_folder_collisions(args.dataset_folder, args.frequency)
    X_collisions = dataset.preprocess_data(X_collisions, pipeline, train=False)
    if args.test_split == True:
        Dataloader_collisions, DataLoader_val, df_col, df_val = dataset.split_data(X_collisions, args.test_split, df_test)
        logging.info(f"Computing threshold on a test set subset")   
        model.eval()

        anomaly_scores_norm = compute_anomaly_scores_embedded(model, df_val, X_collisions.shape[1])
        df_val = df_val[-anomaly_scores_norm.shape[0]:] 
        y_true = plot_hist(anomaly_scores_norm, df_collision, df_val, 'plot_hist_val')
        _, _, th = compute_metrics(anomaly_scores_norm, df_val, y_true)
        
        anomaly_scores_norm = compute_anomaly_scores_embedded(model, df_col, X_collisions.shape[1])
        df_col = df_col[-anomaly_scores_norm.shape[0]:] 
        y_true = plot_hist(anomaly_scores_norm, df_collision, df_col, 'plot_hist_test')
        logging.info(f"Computing metrics on test set") 
        compute_metrics(anomaly_scores_norm, df_col, y_true, th)
    else:
        Dataloader_collisions = return_dataloader(X_collisions) 
        anomaly_scores_norm = compute_anomaly_scores_embedded(model, X_collisions, X_collisions.shape[1])
        df_test = df_test[-anomaly_scores_norm.shape[0]:] 
        y_true = plot_hist(anomaly_scores_norm, df_collision, df_test, 'plot_hist_test')
        anomaly_score = {
            'anomaly_scores_norm' : anomaly_scores_norm,
            'true_labels' : y_true
        }
        with open('anomaly_score.pickle', 'wb') as handle:
            pickle.dump(anomaly_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Computing metrics on test set") 
        # metrics = compute_metrics_pak(anomaly_scores_norm, y_true, pa=True, interval=10, k=0)
        # logging.info(f"compute pak metrics = {metrics}") 
        fpr, tpr, _ = compute_metrics(anomaly_scores_norm, df_test, y_true)
        plt.title("Roc Curve")
        plt.plot(fpr, tpr, color="r")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig("Roc Curve.png")
        plt.show()
    
if args.resume == True:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(f"Resuming model....")   
    logging.info(f"Arguments: {vars(args)}")   
    checkpoint = torch.load(args.model_path)
    X_train = dataset.read_folder_normal(args.dataset_folder, args.frequency)
    X_train, pipeline = dataset.preprocess_data(X_train)

    model = EncDecAD(X_train.shape[1])
    model.load_state_dict(checkpoint["model"])
    anomaly_scorer = checkpoint["anomaly_scorer"]
    model.anomaly_scorer = anomaly_scorer 

    if args.device == 'cuda':
        model = model.to('cuda')

    evaluation(model, pipeline)



