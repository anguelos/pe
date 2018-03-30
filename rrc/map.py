import numpy as np
import scipy.spatial.distance

from matplotlib import pyplot as plt


def get_precision_at_recall_at_matrices(distance_matrix,labels,self_distance=True,nicest=True,remove_singleton_queries=True):
    """

    :param distance_matrix: A float matrix expected to have the distances between samples. Order in both rows is
        expected to be the same. The matrix should be square the diagonal should be zero.
    :param labels: A numpy array of numbers or strings providing the class id for each sample in the matrix.
    :param self_distance: Does each query contain the distance to its self? In retrieval the answer should be True while
        in classification it should probably be False.
    :param nicest: Teak the distance matrix by e so that the sorting has a minimal bias in favor or against the method.
    :param remove_singleton_queries: Should queries with samples who are the sole representative of their class
        participate in the output matrices?
    :return:  precision_at,recall_at, correct_retrievals,labels
    """
    assert labels.size**2==distance_matrix.size
    assert isinstance(labels,np.ndarray)
    e=1e-10
    distance_matrix=distance_matrix.astype('double')
    correct_mat=labels[None,:]==labels[:,None]
    if nicest:
        retrival_mat=np.argsort(distance_matrix-e*correct_mat,axis=1)
    else:
        retrival_mat = np.argsort(distance_matrix + e * correct_mat, axis=1)
    #retrival_mat = np.argsort(distance_matrix,axis=1)
    correct_retrievals=correct_mat[np.arange(labels.size)[:,None],retrival_mat]
    if self_distance:
        correct_retrievals=correct_retrievals[:,1:]

    precision_dividers=np.ones_like(correct_retrievals,dtype='float').cumsum(axis=1)

    recall_dividers=np.ones_like(correct_retrievals,dtype='float')*correct_retrievals.sum(axis=1)[:, None]+e
    #recall_dividers[recall_dividers.cumsum(axis=1) > correct_retrievals.sum(axis=1)[:, None]] = 0;
    #recall_dividers=recall_dividers.cumsum(axis=1)

    precision_at = correct_retrievals.cumsum(axis=1) / precision_dividers
    recall_at=correct_retrievals.cumsum(axis=1)/recall_dividers
    recall_at[recall_at>1.0]=1.0
    if remove_singleton_queries:
        labs, nums = np.unique(labels, return_counts=True);
        keep = dict(zip(labs.tolist(), [n > 1 for n in nums.tolist()]));
        keep_idx = np.array([keep[val] for val in labels.reshape(-1)])
        precision_at=precision_at[keep_idx,:]
        recall_at = recall_at[keep_idx, :]
        correct_retrievals = correct_retrievals[keep_idx, :]
        labels=labels[keep_idx]
    return precision_at,recall_at, correct_retrievals,labels


def get_map(labels,distance_matrix=None,embeddings=None,metric='euclidean'):
    e=1e-10
    if distance_matrix is None:
        if embeddings is None:
            raise ValueError()
        assert labels.size==embeddings.shape[0]
        distance_matrix=scipy.spatial.distance.cdist(embeddings,embeddings,metric=metric)
    print embeddings.shape
    print distance_matrix.shape

    precision_at,_, correct_retrievals,_= get_precision_at_recall_at_matrices(distance_matrix,labels,self_distance=True)

    #getting precision at correct predictions. averaging horizontally (per query) and than vertically mAP.
    AP=((precision_at*correct_retrievals).cumsum(axis=1)/(correct_retrievals+e).cumsum(axis=1))[:,-1]
    return AP.mean(),AP,precision_at

