import torch
import numpy as np


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


# y_true, y_pred -> [N_CLASS, H, W]
def dice_coef_multilabel(y_true, y_pred, num_labels=3):
    #print(torch.min(y_true), torch.min(y_pred))
    #print(torch.max(y_true), torch.max(y_pred))
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    dice_val = 0
    for i in range(num_labels):
        # dice score for single class
        dice_val += dice_coef(y_true[i,:,:,], y_pred[i,:,:,])
        #print(f'Dice Val {dice_val} for label {i}')
    # tot dice score is the mean of each dice score for each class
    return dice_val/num_labels


def eval_batch(data, output, labels, n_class=3):
    batch_size = len(data)
    dice_score, dice_batch = 0, 0
    for i in range(batch_size):
        dice_score = dice_coef_multilabel(labels[i], output[i], n_class)
        # view one sample for batch
        dice_batch += dice_score
    return dice_batch / batch_size