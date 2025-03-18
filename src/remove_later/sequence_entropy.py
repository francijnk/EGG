# https://github.com/pafoster/pyitlib/blob/befc5357ed871f8a63ae8563ad0cb95106aec42c/pyitlib/discrete_random_variable.py
import torch
import numpy as np
import pyitlib.discrete_random_variable as itlib
from collections import defaultdict
from typing import Optional


def entropy(x: torch.Tensor, alphabet: Optional[torch.Tensor] = None):
    """
    Estimates entropy of the RV X represented by the tensor x.

    If the tensor has more than one dimension, each tensors indexed by the 1st
    dimension is treated as one of the elements to apply the operation upon.
    """
    if x.dim() == 0:
        return 0.
    elif x.dim() > 1:
        _, x = torch.unique(x, return_inverse=True, dim=0)

    if alphabet is not None:
        alphabet = alphabet.numpy()

    H = itlib.entropy(x.numpy(), estimator='JAMES-STEIN', Alphabet_X=alphabet)

    return H.item()


def entropy_joint_(
        messages: Optional[torch.Tensor],
        attributes: Optional[torch.Tensor],
        alphabet: Optional[torch.Tensor] = None,
        estimator: Optional[str] = 'JAMES-STEIN'):
    """
    Computes joint entropy.
    Code sourced from pyitlib.discrete_random_variable, with minor adjustments to return 
    the estimated joint probability distribution and handle conversion from tensors to arrays.
    """
    assert messages is not None or attributes is not None

    if attributes is None:
        xy = messages.t().numpy()
        x_index = np.ones_like(xy)
        alphabet = itlib._sanitise_array_input(np.unique(xy, axis=0), -1)  # swap
    elif messages is None:
        xy = attributes.t().numpy()
        x_index = np.zeros_like(xy)
        alphabet = itlib._autocreate_alphabet(xy, -1)
    else:
        x, y = messages.t().numpy(), attributes.t().numpy()
        xy = np.vstack((x, y))
        x_index = np.ones_like(xy)
        x_index[len(x):] = 0
        alphabet_x = itlib._autocreate_alphabet(x, -1)[0].tolist()
        alphabet_y = itlib._autocreate_alphabet(y, -1)[0].tolist()
        print(type(alphabet_x), alphabet_x[0])
        max_len = max(len(sublist) for sublist in alphabet_x + alphabet_y)
        alphabet, _ = itlib._sanitise_array_input([
            (lambda x: (x + max_len * [-1])[:max_len])(sublist)
            for sublist in alphabet_x + alphabet_y], -1)

    print(xy)
    (xy, alphabet), _ = itlib._map_observations_to_integers((xy, alphabet), (-1, -1))

    # Re-shape X, so that we may handle multi-dimensional arrays equivalently
    # and iterate across 0th axis
    xy = xy.reshape(-1, xy.shape[-1])
    x_index = x_index.reshape(-1, xy.shape[-1])
    alphabet = alphabet.reshape(-1, alphabet.shape[-1])
    print("xy/alph", xy.shape, alphabet.shape)

    itlib._verify_alphabet_sufficiently_large(xy, alphabet, -1)

    # sort columns and record, which ones are sourced from which input
    for i in range(x.shape[0]):
        index = xy[i].argsort(kind='mergesort')
        xy = xy[:, index]
        x_index = x_index[:, index]

    # Compute symbol run-lengths
    # Compute symbol change indicators
    B = np.any(xy[:, 1:] != xy[:, :-1], axis=0)
    # Obtain symbol change positions
    I = np.append(np.where(B), xy.shape[1] - 1)
    # Compute run lengths
    L = np.diff(np.append(-1, I))

    print("B", B, B.shape)
    print("I", I, I.shape)
    print("L", L, L.shape)

    _alphabet = xy[:, I]
    n_additional_empty_bins = itlib._determine_number_additional_empty_bins(
        L, _alphabet, alphabet, -1)
    L, _ = itlib._remove_counts_at_fill_value(L, _alphabet, -1)
    if not np.any(L):
        return np.float64(np.NaN)

    # P_0 is the probability mass assigned to each additional empty bin
    P, P_0 = itlib._estimate_probabilities(L, 'JAMES-STEIN', n_additional_empty_bins)
    H_0 = n_additional_empty_bins * P_0 * -np.log2(P_0 + np.spacing(0))
    H = itlib.entropy_pmf(P, 2, require_valid_pmf=False) + H_0
    print("P", P, P.shape)
    print("P0", P_0, P_0.shape)
    print("n_additional_empty_bins", n_additional_empty_bins)

    #xy_total_probs = {k: sum(P)}
    #xy_n_empty_bins = {}

    return H

    # xy
    # alphabet_xy































def mutual_info_(x: torch.Tensor, y: torch.Tensor, z: Optional[torch.Tensor] = None):
    """
    I(X1, ..., Xn; Y)
    """


    assert y.dim() == 1

    def entropy_joint_conditional(_a, _b):
        # For a compound RV A = A1, ..., Ak and a possibly compound RV B,
        # compute H(A1, ..., Ak | B)

    if x.dim() == 1 and z is None:
        information_mutual()
    elif x.dim() == 1 and z is not None
        information_mutual_conditional()
    elif x.dim() > 1 and z is None:
        H_x = entropy_joint()
        H_conditional = 0
        for i, y_val in enumerate(torch.unique(y)):
            index = y[y == y_val]
            select rows 
            

    mi_array = np.empty((x.size(1), y.size(1)), dtype=np.float)

    if z is None:
        pass
    elif z.dim() == 1:
        pass
    else:
        
            

    

















