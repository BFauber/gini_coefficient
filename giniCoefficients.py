#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''giniCoefficients.py'''


'''
Ben Fauber
Austin, Texas USA
Feb. 2025
'''


import numpy as np
from sklearn.preprocessing import MinMaxScaler


def l2_normalize_vec(v):
    '''
    Returns normalized vector using L2-norm
    '''
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm



def gini_coefficient_numpy(array: np.ndarray) -> float:
    """
    Calculates Gini Coefficient using the method of:
    
    Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; Wooley, R.
    `Bootstrapping the Gini Coefficient of Inequality.` Ecology, 1987, 68, 1548-1551.
    
    AND 
    
    Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; Wooley, R. `Errratum to 
    Bootstrapping the Gini Coefficient of Inequality.` Ecology, 1988, 69, 1307.
    
    Input:
    ------
    array : np.ndarray of n x 1 dimensions
    
    Output:
    -------
    scalar : float
    """
    array = array.flatten()
    if np.any(array < 0):
        return "Error: Gini coefficient requires non-negative values."
    sorted_array = np.sort(array)
    n = sorted_array.size
    if n == 0:
        return 0
    index = np.arange(1, n+1)
    gini = (2 * np.sum(index * sorted_array) / np.sum(sorted_array) - (n+1)) / n
    return gini


def by_row_similarity(row: int, embeddings: np.ndarray) -> np.ndarray:
    """
    Calculates one-versus-all similarity of a row element 
    versus the np.ndarray (n x d) embeddings where the row
    element is in the d-dimensional embeddings array.
    """
    row_element = embeddings[row, :]
    similarities = np.dot(embeddings, row_element)
    similarities = 1 - similarities
    similarities[similarities < 0.0] = 0.0
    return similarities


def calc_gini_by_row(embeddings: np.ndarray, normalize: bool=True) -> np.ndarray:
    """
    Calculates all-versus-all similarities of np.ndarrray (n x d) of embeddings.

    NOTE: ASSUMES all embeddings are L2-NORMALIZED.
    
    `Normalize` is a MinMax scaling of outputs such that [0,1] for all Gini values.
    
    Input:
    ------
    embeddings : np.ndarray of n x d dimensions *NOTE: ASSUMES ALL EMBEDDINGS ARE
        L2-NORMALIZED
    normalize : bool {default=True} enables or disables the `MinMax` scaling of
        the resulting Gini coefficients to [0,1]
    
    Output:
    -------
    array : np.ndarray of n x 1 dimensions of the Gini coefficients of 
        the input array
    """
    gini_list = []
    i = 0
    n = embeddings.shape[0]
    while i < n:
        similarities = by_row_similarity(i, embeddings)
        gini = gini_coefficient_numpy(similarities)
        gini_list.append(gini)
        i += 1
    gini_array = np.array(gini_list).reshape(-1, 1)
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(gini_array)
        gini_array = scaler.transform(gini_array).flatten()
    return gini_array


