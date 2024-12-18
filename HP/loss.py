import numpy as np
import torch

def loglike_loss(model, sequences):
    neg_log_likelihood = 0
    nums = 0
    for sequence in sequences:
        nums += len(sequence)
        T_i, _ = sequence[-1]
        # event_ll
        for m, (t_m, c_m) in enumerate(sequence):
            intensity = model.compute_intensity(t_m, c_m, sequence)
            log_intensity = np.log(intensity)
            neg_log_likelihood -= log_intensity

        # non_event_ll
        for c in range(model.num_types):
            integral_term = 0
            for m, (t_m, c_m) in enumerate(sequence):
                integral_term += model.A[c][c_m] * (1 - np.exp(-(T_i - t_m)))
            neg_log_likelihood += (model.mu[c] * T_i + integral_term)

    return -neg_log_likelihood, -neg_log_likelihood/nums


def cal_accuracy(events,predicted_events):
    """ Event prediction loss, cross entropy or label smoothing. """
    correct_num = 0
    nums = 0
    for event,pred_event in zip(events,predicted_events):
        event = event[1:]
        pred_event = pred_event[:-1]
        nums += len(event)
        truth = np.array([type for _, type in event])
        pred_type = np.array([type for _, type in pred_event])
        correct_num += np.sum(pred_type == truth)

    return correct_num,correct_num / nums

def time_loss(events,predicted_events):
    """ Time prediction loss. """
    num_events = 0
    se = 0
    for event, pred_event in zip(events, predicted_events):
        event = event[1:]
        pred_event = pred_event[:-1]
        num_events += len(event)
        true = np.array([time for time, _ in event])
        prediction = np.array([time for time, _ in pred_event])
        # event time gap prediction
        diff = prediction - true
        se += np.sum(diff * diff)

    rmse_norm = np.linalg.norm(np.array([se - num_events]), ord=2) / np.linalg.norm(np.array([se]), ord=2)
    rmse = np.sqrt(se / num_events)
    return rmse_norm, rmse