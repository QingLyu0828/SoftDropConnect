import numpy as np


def acc(outputs, targets, samples):
    each = np.zeros(samples)
    for i in range(samples):
        mat = outputs[:,:,i].cpu().numpy()
        index = mat.argmax(axis=1)
        accu = np.sum(index == targets.data.cpu().numpy())
        each[i] = accu
    overall = np.mean(each)
    return overall, each

