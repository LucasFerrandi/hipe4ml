"""
Module containing the analysis utils.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


def bdt_efficiency_array(y_truth, y_score, n_points=50,
                         keep_lower=False, calculate_contamination=False,
                         calculate_bkg_efficiency=False):
    """
    Calculate the model efficiency as a function of the score
    threshold. The candidates for each class should be labeled
    with 0, ..., N. In case of binary classification, 0 should
    correspond to the background while 1 to the signal

    Parameters
    ------------------------------------------------
    y_truth: array
        Training or test set labels. The candidates for each
        class should be labeled with 0, ..., N.
        In case of binary classification, 0 should
        correspond to the background while 1 to the signal

    y_score: array
        Estimated probabilities or decision function.

    n_points: int
        Number of points to be sampled

    keep_lower: bool
        If True compute the efficiency using the candidates with
        score lower than the score threshold; otherwise using the
        candidates with score higher than the score threshold.

    calculate_contamination : bool
        If True, also calculate the contamination from each other class.
        In that case, significance is always computed as well

    calculate_bkg_efficiency : bool
        If True, also calculate prompt efficiency, non-prompt efficiency,
        and background rejection versus the background score threshold.
        Background, prompt and nonprompt should correspond respectively to 0, 1 and 2.
        The three arrays are appended to the returned tuple in that order.

    Returns
    ------------------------------------------------
    efficiencies : numpy array
        Efficiency as a function of the threshold value
        Numpy array of numpy arrays in case of multi-classification

    threshold : numpy array
        Threshold values.

    contaminations : numpy array
        Only returned if calculate_contamination=True.
        For multi-class classification:
        contaminations[signal_class, source_class, threshold]
        For binary classification:
        contaminations[1, 0, threshold]
        contains the background contamination of the signal sample.

    significance : numpy array
        Only returned if calculate_contamination=True.
        For binary classification, significance is computed as
        signal / sqrt(total contamination). For multi-class
        classification, it is computed class-by-class.
    """
    y_truth = np.asarray(y_truth)
    y_score = np.asarray(y_score)
    if y_truth.ndim == 0:
        y_truth = y_truth.reshape(1)
    if y_score.ndim == 0:
        y_score = y_score.reshape(1)

    if y_truth.shape[0] != y_score.shape[0]:
        raise ValueError("y_truth and y_score must have the same number of samples")

    operator = np.greater
    if keep_lower:
        operator = np.less
    n_classes = len(np.unique(y_truth))
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1")
    else:
        min_score = np.min(y_score)
        max_score = np.max(y_score)
        threshold = np.linspace(min_score, max_score, n_points)

    bkg_efficiencies = None
    if calculate_bkg_efficiency:
        if y_truth.ndim != 1 or y_score.ndim != 2 or y_score.shape[1] < 3:
            raise ValueError(
                "BKG-score efficiencies require y_truth labels and "
                "y_score columns [bkg, prompt, nonprompt]"
            )
        prompt_mask = y_truth == 1
        nonprompt_mask = y_truth == 2
        background_mask = y_truth == 0
        n_prompt = np.sum(prompt_mask)
        n_nonprompt = np.sum(nonprompt_mask)
        n_background = np.sum(background_mask)
        if n_prompt == 0 or n_nonprompt == 0 or n_background == 0:
            raise ValueError(
                "y_truth must contain at least 1 background, prompt, and nonprompt event"
            )
        bkg_score = y_score[:, 0]
        prompt_efficiency = np.zeros(n_points)
        nonprompt_efficiency = np.zeros(n_points)
        background_rejection = np.zeros(n_points)
        # The background score is used as an upper cut independently of
        # keep_lower, which controls the main efficiency curves above.
        for i, thr in enumerate(threshold):
            mask = np.less(bkg_score, thr)
            prompt_efficiency[i] = np.sum(prompt_mask & mask) / n_prompt
            nonprompt_efficiency[i] = np.sum(nonprompt_mask & mask) / n_nonprompt
            background_rejection[i] = np.sum(background_mask & ~mask) / n_background
        bkg_efficiencies = (
            prompt_efficiency, nonprompt_efficiency, background_rejection
        )

    significance = None
    if n_classes <= 2:
        # Class 1 is the signal
        n_sig = np.sum(y_truth)
        efficiencies = np.zeros(n_points)
        if calculate_contamination:
            contaminations = np.zeros((2, 2, n_points))
            # significance = np.zeros(n_points)
            significance = np.zeros((n_classes, n_points))
        for i, thr in enumerate(threshold):
            mask = operator(y_score, thr)
            n_selected = np.sum(mask)
            n_sig_selected = np.sum(y_truth[mask])
            if n_sig > 0:
                efficiencies[i] = n_sig_selected / n_sig

            if calculate_contamination and n_selected > 0:
                # Class 0 contaminating class 1
                contaminations[1, 0, i] = (
                    np.sum((y_truth == 0) & mask) / n_selected
                )

                n_contamination = n_selected - n_sig_selected

                # Significance
                # Remember: if significance is extracted from MC, it is dependent of the model's f_B, then it is not trustworthy. Such as contamination
                if n_contamination > 0:
                    ratio = n_sig_selected / n_contamination
                    inner = 2.0 * ((n_sig_selected + n_contamination) * np.log1p(ratio) - n_sig_selected)
                    significance[0][i] = np.sqrt(np.maximum(inner, 0.0)) # "Asimov" Significance (eq. 97 from Eur. Phys. J. C (2011) 71: 1554)

                    # significance[i] = n_sig_selected / np.sqrt(n_contamination)
                elif n_sig_selected > 0:
                    significance[i] = 9999 # simulate infinite significance when there is no background
                else:
                    significance[i] = 0.0

    else:
        y_truth_multi = label_binarize(
            y_truth, classes=range(n_classes)
        )
        efficiencies = []
        if calculate_contamination:
            contaminations = np.zeros(
                (n_classes, n_classes, n_points)
            )
            significance = np.zeros((n_classes, n_points))
        for clas in range(n_classes):
            n_sig = np.sum(y_truth_multi[:, clas])
            efficiency = np.zeros(n_points)
            for i, thr in enumerate(threshold):
                mask = operator(y_score[:, clas], thr)
                n_selected = np.sum(mask)
                n_sig_selected = np.sum(
                    y_truth_multi[:, clas][mask]
                )
                if n_sig > 0:
                    efficiency[i] = n_sig_selected / n_sig
                if calculate_contamination and n_selected > 0:
                    # Calculate contamination from every other class
                    for source_class in range(n_classes):
                        if source_class == clas:
                            continue
                        n_contamination = np.sum(
                            y_truth_multi[:, source_class][mask]
                        )
                        contaminations[
                            clas, source_class, i
                        ] = n_contamination / n_selected
                    n_contamination = n_selected - n_sig_selected
                    if n_contamination > 0:
                        ratio = n_sig_selected / n_contamination
                        inner = 2.0 * ((n_sig_selected + n_contamination) * np.log1p(ratio) - n_sig_selected)
                        significance[clas, i] = np.sqrt(np.maximum(inner, 0.0)) # "Asimov" Significance (eq. 97 from Eur. Phys. J. C (2011) 71: 1554)
                        # Remember: if significance is extracted from MC, it is dependent of the model's f_B, then it is not trustworthy. Such as contamination
                    elif n_sig_selected > 0:
                        significance[clas, i] = np.inf
                    else:
                        significance[clas, i] = 0.0
            efficiencies.append(efficiency)
        efficiencies = np.array(efficiencies)

    result = (efficiencies, threshold)
    if calculate_contamination:
        result += (contaminations, significance)
    if calculate_bkg_efficiency:
        result += bkg_efficiencies
    return result


def bdt_efficiency_vs_var(
    y_truth,
    y_score,
    var,
    nonpCut,
    bkgCut,
    n_points=20,
    var_max=None
):
    """
    Calculate prompt and non-prompt efficiencies as a function of var (e.g: pT).

    Parameters
    ----------
    y_truth : array-like
        True class labels: 0 = background, 1 = prompt, 2 = non-prompt. # todo: generalize the order

    y_score : array-like
        BDT scores with columns:
        [bkg_score, prompt_score, nonprompt_score].

    var : array-like
        var distribution of each candidate.

    nonpCut : float
        Minimum non-prompt BDT score.

    bkgCut : float
        Maximum background BDT score.

    n_points : int
        Number of var bins.

    Returns
    -------
    pt_centers : ndarray
        Centers of the var bins.

    prompt_eff : ndarray
        Prompt efficiency in each var bin.

    prompt_err : ndarray
        Statistical uncertainty on prompt efficiency.

    nonprompt_eff : ndarray
        Non-prompt efficiency in each var bin.

    nonprompt_err : ndarray
        Statistical uncertainty on non-prompt efficiency.
    """

    y_truth = np.asarray(y_truth)
    y_score = np.asarray(y_score)
    var = np.asarray(var)

    if y_score.ndim != 2 or y_score.shape[1] < 3:
        raise ValueError(
            "y_score must have shape (n_samples, 3): "
            "[bkg_score, prompt_score, nonprompt_score]"
        )

    if not (len(y_truth) == len(y_score) == len(var)):
        raise ValueError(
            "y_truth, y_score and var must have the same number of samples"
        )

    # BDT scores
    bkg_score = y_score[:, 0]
    nonp_score = y_score[:, 2]

    # BDT selection
    selection = (
        (nonp_score > nonpCut)
        & (bkg_score < bkgCut)
    )

    if not var_max:
        var_max = np.max(var)

    # var bins
    var_edges = np.linspace(
        np.min(var),
        # np.max(var),
        var_max,
        n_points + 1
    )

    pt_centers = 0.5 * (var_edges[:-1] + var_edges[1:])

    prompt_eff = np.zeros(n_points)
    prompt_err = np.zeros(n_points)

    nonprompt_eff = np.zeros(n_points)
    nonprompt_err = np.zeros(n_points)

    prompt = y_truth == 1
    nonprompt = y_truth == 2

    for i in range(n_points):
        if i == n_points - 1:
            pt_mask = (
                (var >= var_edges[i])
                & (var <= var_edges[i + 1])
            )
        else:
            pt_mask = (
                (var >= var_edges[i])
                & (var < var_edges[i + 1])
            )

        # Prompt
        n_prompt = np.sum(prompt & pt_mask) # N of true prompt in this var range
        n_prompt_selected = np.sum(
            prompt & pt_mask & selection
        )
        if n_prompt > 0:
            prompt_eff[i] = n_prompt_selected / n_prompt
            prompt_err[i] = np.sqrt(
                prompt_eff[i] * (1 - prompt_eff[i]) / n_prompt
            ) # Binomial error

        # Non-prompt
        n_nonprompt = np.sum(nonprompt & pt_mask)
        n_nonprompt_selected = np.sum(
            nonprompt & pt_mask & selection
        )

        if n_nonprompt > 0:
            nonprompt_eff[i] = n_nonprompt_selected / n_nonprompt
            nonprompt_err[i] = np.sqrt(
                nonprompt_eff[i] * (1 - nonprompt_eff[i]) / n_nonprompt
            )

    return (
        pt_centers,
        prompt_eff,
        prompt_err,
        nonprompt_eff,
        nonprompt_err,
    )



def score_from_efficiency_array(y_truth, y_score, efficiency_selected, keep_lower=False):
    """
    Return the score array corresponding to an external fixed efficiency
    array.

    Parameters
    -----------------------------------------------
    y_truth: array
        Training or test set labels. The candidates for each
        class should be labeled with 0, ..., N.
        In case of binary classification, 0 should
        correspond to the background while 1 to the signal

    y_score: array
        Estimated probabilities or decision function.

    keep_lower: bool
        If True compute the efficiency using the candidates with
        score lower than the score threshold; otherwise using the
        candidates with score higher than the score threshold.

    efficiency_selected: list or array
        Efficiency array along which calculate
        the corresponding score array

    Returns
    -----------------------------------------------
    out: numpy array
        Score array corresponding to efficiency_selected
    """
    score_list = []
    eff, score = bdt_efficiency_array(
        y_truth, y_score, n_points=1000, keep_lower=keep_lower)
    for eff_val in efficiency_selected:
        interp = InterpolatedUnivariateSpline(score, eff-eff_val)
        score_list.append(interp.roots()[0])
    score_array = np.array(score_list)
    return score_array


def train_test_generator(data_list, labels_list, sliced_df=False, **kwds):
    """
    Return a list containing respectively training set dataframe,
    training label array, test set dataframe, test label array
    computed from a list of TreeHandler objects. If sliced_df == True,
    the method preforms the train-test split for each slice

    Parameters
    -----------------------------------------------
    data_list: list
        List of TreeHandler models. For example: if you perform binary
        classification the list should contain the TreeHandlers corresponding
        to the signal and the background candidates

    labels_list: list
        List containing the labels associated to each DataHandler. For example:
        if you perform binary classification, the list should be [1,0]

    sliced_df: bool
        If True, the function searches for the slices stored in the DataHandler
        and perform the train_test_split for each slice

    **kwds
        Extra arguments are passed on to sklearn.model_selection.train_test_split:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Returns
    -----------------------------------------------
    out: list
        List containing respectively training set dataframe,
        training label array, test set dataframe, test label array. If sliced_df==True
        returns a list containing the above cited list for each slice
    """
    if sliced_df is False:
        labels_train_test = []
        df_list = []
        for data, labels in zip(data_list, labels_list):
            data_frame = data.get_data_frame()
            labels_train_test += len(data_frame)*[labels]
            df_list.append(data_frame)
        del data_list, data_frame
        df_tot_train_test = pd.concat(df_list, sort=True)
        del df_list
        train_test = train_test_split(df_tot_train_test, np.array(labels_train_test), **kwds)
        # swap for ModelHandler compatibility
        train_test[1], train_test[2] = train_test[2], train_test[1]
        return train_test

    train_test_slices = []
    n_slices = len(data_list[0].get_projection_binning())
    for slice_ind in range(n_slices):
        labels_train_test = []
        df_list = []
        for data, labels in zip(data_list, labels_list):
            data_frame = data.get_slice(slice_ind)
            labels_train_test += len(data_frame)*[labels]
            df_list.append(data_frame)
        del data_frame
        df_tot_train_test = pd.concat(df_list, sort=True)
        del df_list
        train_test = train_test_split(df_tot_train_test, np.array(labels_train_test), **kwds)
        train_test[1], train_test[2] = train_test[2], train_test[1]
        train_test_slices.append(train_test)
    return train_test_slices
