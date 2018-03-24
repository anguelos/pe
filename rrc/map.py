import numpy as np
import scipy.spatial.distance

def get_precision_at_matrix(distance_matrix,labels,self_distance=True):
    assert labels.size**2==distance_matrix.size
    assert isinstance(labels,np.array)
    correct_mat=labels[None,:]==labels[:,None]
    retrival_mat=np.argsort(distance_mat,axis=1)
    correct_retrievals=correct_mat[np.range(labels.size)[:,None],retrival_mat]
    if self_distance:
        correct_retrievals=correct_retrievals[:,1:]

    dividers=np.ones_like(correct_retrievals,dtype='float')
    dividers[dividers.cumsum(axis=1) > correct_retrievals.sum(axis=1)[:, None]] = 0;
    dividers=dividers.cumsum(axis=1)

    return correct_retrievals.cumsum(axis=1)/dividers, correct_retrievals


def get_map(labels,distance_matrix=None,embeddings=None,metric='euclidean'):
    if distance_matrix is None:
        if embeddings is None:
            raise ValueError()
        assert
        distance_matrix=scipy.spatial.distance.pdist(embeddings,metric=metric)

    precision_at, correct_retrievals= get_precision_at_matrix(distance_matrix,labels,self_distance=True)

    #getting precision at correct predictions. averaging horizontally (per query) and than vertically mAP.
    AP=((precision_at*correct_retrievals).cumsum(axis=1)/correct_retrievals.cumsum(axis=1))[:,-1]
    return AP.mean(),AP
