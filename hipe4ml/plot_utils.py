"""
Module containing the plot utils. Each function returns a matplotlib object
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.special import softmax
from sklearn.metrics import (auc, average_precision_score, mean_squared_error,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

import hipe4ml.tree_handler


def _plot_output(df_train, df_test, lims, bins, label, color, kwds):
    """
    Utility function for plot_output_train_test
    """
    plt.hist(df_train, color=color, alpha=0.5, range=lims, bins=bins,
             histtype='stepfilled', label=f'{label} pdf Training Set', **kwds)

    if 'density' in kwds and kwds['density']:
        hist, bins = np.histogram(df_test, bins=bins, range=lims, density=True)
        scale = len(df_test) / sum(hist)
        err = np.sqrt(hist * scale) / scale
    else:
        hist, bins = np.histogram(df_test, bins=bins, range=lims)
        scale = len(df_train) / len(df_test)
        err = np.sqrt(hist) * scale
        hist = hist * scale

    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o',
                 c=color, label=f'{label} pdf Test Set')


def plot_output_train_test(model, data, bins=80, output_margin=True, labels=None, logscale=False, **kwds):
    """
    Plot the model output distributions for each class and output
    both for training and test set.

    Parameters
    ----------------------------------------
    model: hipe4ml model handler

    data: list
        Contains respectively: training
        set dataframe, training label array,
        test set dataframe, test label array

    bins: int or sequence of scalars or str
        If bins is an int, it defines the number of equal-width
        bins in the given range (10, by default). If bins is a
        sequence, it defines a monotonically increasing array of
        bin edges, including the rightmost edge, allowing for
        non-uniform bin widths.
        If bins is a string, it defines the method used to
        calculate the optimal bin width, as defined by
        np.histogram_bin_edges:
        https://docs.scipy.org/doc/numpy/reference/generated
        /numpy.histogram_bin_edges.html#numpy.histogram_bin_edges

    output_margin: bool
        Whether to output the raw untransformed margin value.

    labels: list
        Contains the labels to be displayed in the legend
        If None the labels are class1, class2, ..., classN

    logscale: bool
        Whether to plot the y axis in log scale

    **kwds
        Extra arguments are passed on to plt.hist()

    Returns
    ----------------------------------------
    out: matplotlib.figure.Figure or list of them
        Model output distributions for each class
    """
    class_labels = np.unique(data[1])
    n_classes = len(class_labels)

    prediction = []
    for xxx, yyy in ((data[0], data[1]), (data[2], data[3])):
        for class_lab in class_labels:
            prediction.append(model.predict(
                xxx[yyy == class_lab], output_margin))

    low = min(np.min(d) for d in prediction)
    high = max(np.max(d) for d in prediction)
    low_high = (low, high)

    # only one figure in case of binary classification
    if n_classes <= 2:
        res = plt.figure()
        labels = ['Signal', 'Background'] if labels is None else labels
        colors = ['b', 'r']
        for i_class, (label, color) in enumerate(zip(labels, colors)):
            _plot_output(
                prediction[i_class], prediction[i_class+2], low_high, bins, label, color, kwds)
        if logscale:
            plt.yscale('log')
        plt.xlabel('BDT output', fontsize=13, ha='right', position=(1, 20))
        plt.ylabel('Counts (arb. units)', fontsize=13,
                   horizontalalignment='left')
        plt.legend(frameon=False, fontsize=12, loc='best')

    # n figures in case of multi-classification with n classes
    else:
        res = []
        labels = [
            f'class{class_lab}' for class_lab in class_labels] if labels is None else labels
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i_class) for i_class in range(len(labels))]
        for output, out_label in zip(class_labels, labels):
            res.append(plt.figure())
            for i_class, (label, color) in enumerate(zip(labels, colors)):
                _plot_output(prediction[i_class][:, output], prediction[i_class+n_classes][:, output], low_high, bins,
                             label, color, kwds)
            if logscale:
                plt.yscale('log')
            plt.xlabel(f'BDT output for {out_label}',
                       fontsize=13, ha='right', position=(1, 20))
            plt.ylabel('Counts (arb. units)', fontsize=13,
                       horizontalalignment='left')
            plt.legend(frameon=False, fontsize=12, loc='best')

    return res

# flake8: noqa: C901
def plot_distr(data_list, column=None, bins=50, labels=None, colors=None, **kwds):  # pylint: disable=too-many-branches
    """
    Draw histograms comparing the distributions of each class.

    Parameters
    -----------------------------------------
    data_list: TreeHandler, pandas.Dataframe or list of them
        Contains a TreeHandler or a dataframe for each class

    column: str or list of them
        Contains the name of the features you want to plot
        Example: ['dEdx', 'pT', 'ct']. If None all the features
        are selected

    bins: int or sequence of scalars or str
        If bins is an int, it defines the number of equal-width
        bins in the given range (10, by default). If bins is a
        sequence, it defines a monotonically increasing array of
        bin edges, including the rightmost edge, allowing for
        non-uniform bin widths.

    labels: str or list of them
        Contains the labels to be displayed in the legend
        If None the labels are class1, class2, ..., classN

    colors: str or list of them
        List of the colors to be used to fill the histograms

    **kwds:
        extra arguments are passed on to pandas.DataFrame.hist():
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html

    Returns
    -----------------------------------------
    out: numpy array of matplotlib.axes.AxesSubplot
        Distributions of the features for each class
    """
    list_of_df = []
    if not isinstance(data_list, list):
        data_list = [data_list]
        labels = [labels]
        colors = [colors]
    if isinstance(data_list[0], hipe4ml.tree_handler.TreeHandler):
        for handl in data_list:
            list_of_df.append(handl.get_data_frame())
    else:
        list_of_df = data_list

    if column is not None:
        if not isinstance(column, (list, np.ndarray, pd.Index)):
            column = [column]

    else:
        column = list(list_of_df[0].columns)

    if labels is None:
        labels = [f'class{i_class}' for i_class, _ in enumerate(list_of_df)]

    if colors is None:
        colors = [None for i_class, _ in enumerate(list_of_df)]

    for i_class, (dfm, lab, col) in enumerate(zip(list_of_df, labels, colors)):
        if i_class == 0:
            axes = dfm.hist(column=column, bins=bins, label=lab, color=col, **kwds)
            axes = axes.flatten()
            axes = axes[:len(column)]
        else:
            dfm.hist(ax=axes, column=column, bins=bins, label=lab, color=col, **kwds)
    for axs in axes:
        axs.set_ylabel('Counts')
    axes[-1].legend(loc='best')
    if len(axes) == 1:
        axes = axes[0]
    return axes


def plot_corr(data_list, columns, labels=None, **kwds):
    """
    Calculate pairwise correlation between features for
    each class (e.g. signal and background in case of binary
    classification)

    Parameters
    -----------------------------------------------
    data_list: list
        Contains dataframes for each class

    columns: list
        Contains the name of the features you want to plot
        Example: ['dEdx', 'pT', 'ct']

    labels: list
        Contains the labels to be displayed in the legend
        If None the labels are class1, class2, ..., classN

    **kwds: extra arguments are passed on to DataFrame.corr()

    Returns
    ------------------------------------------------
    out: matplotlib.figure.Figure or list of them
        Correlations between the features for each class
    """
    list_of_df = []
    if isinstance(data_list[0], hipe4ml.tree_handler.TreeHandler):
        for handl in data_list:
            list_of_df.append(handl.get_data_frame())
    else:
        list_of_df = data_list
    corr_mat = []
    for dfm in list_of_df:
        dfm = dfm[columns]
        corr_mat.append(dfm.corr(**kwds))

    if labels is None:
        labels = []
        if len(corr_mat) != 2:
            for i_mat, _ in enumerate(corr_mat):
                labels.append(f'class{i_mat}')
        else:
            labels.append('Signal')
            labels.append('Background')

    res = []
    for mat, lab in zip(corr_mat, labels):
        if len(corr_mat) < 2:
            res = plt.figure(figsize=(8, 7))
            grid = ImageGrid(res, 111, axes_pad=0.15, nrows_ncols=(1, 1), share_all=True,
                             cbar_location='right', cbar_mode='single', cbar_size='7%', cbar_pad=0.15)
        else:
            res.append(plt.figure(figsize=(8, 7)))
            grid = ImageGrid(res[-1], 111, axes_pad=0.15, nrows_ncols=(1, 1), share_all=True,
                             cbar_location='right', cbar_mode='single', cbar_size='7%', cbar_pad=0.15)

        opts = {'cmap': plt.get_cmap(
            'coolwarm'), 'vmin': -1, 'vmax': +1, 'snap': True}

        axs = grid[0]
        heatmap = axs.pcolor(mat, **opts)
        axs.set_title(lab, fontsize=14, fontweight='bold')

        lab = mat.columns.values

        # shift location of ticks to center of the bins
        axs.set_xticks(np.arange(len(lab)), minor=False)
        axs.set_yticks(np.arange(len(lab)), minor=False)
        axs.set_xticklabels(lab, minor=False, ha='left',
                            rotation=90, fontsize=10)
        axs.set_yticklabels(lab, minor=False, va='bottom', fontsize=10)
        axs.tick_params(axis='both', which='both', direction="in")

        for tick in axs.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

        plt.colorbar(heatmap, axs.cax)

    return res

def plot_bdt_eff(
    threshold,
    eff_sig,
    cont_sig=None,
    signif_sig=None,
    leg_labels=None,
    contaminColors=None,
    chosen_bkg_rejec=None,
    bkg_efficiencies=None
):
    """
    Plot the BDT efficiency, purity, contamination and significance.

    When bkg_efficiencies is supplied, an additional plot is produced
    showing prompt and nonprompt efficiency as a function of the
    bkg-score threshold. Supply the three arrays returned by
    bdt_efficiency_array(..., calculate_bkg_efficiency=True).

    Parameters
    -----------------------------------
    threshold: array
        Score threshold array.

    eff_sig: array
        Model efficiency array.

    signif_sig: array, optional
        Significance array with shape (n_classes, n_thresholds).

    leg_labels: array, optional
        Labels for the efficiency curves.

    cont_sig: array, optional
        Contamination array with shape
        (n_classes, n_classes, n_thresholds).

    contaminColors: list of str, optional
        Colors for contamination curves.

    bkg_efficiencies: tuple of arrays, optional
        Prompt efficiency, nonprompt efficiency, and background rejection
        calculated by bdt_efficiency_array with calculate_bkg_efficiency=True.
    """
    threshold = np.asarray(threshold)
    eff_sig = np.asarray(eff_sig)

    if threshold.ndim != 1 or eff_sig.ndim not in (1, 2):
        raise ValueError(
            'threshold must be one-dimensional and '
            'eff_sig must be one- or two-dimensional'
        )

    if eff_sig.shape[-1] != threshold.size:
        raise ValueError(
            'threshold and eff_sig must have the same length'
        )

    curves = eff_sig[np.newaxis, :] if eff_sig.ndim == 1 else eff_sig

    res = []

    # Efficiency / purity / contamination / significance
    if cont_sig is not None:
        if curves.shape[0] == 1:
            raise ValueError(
                'Calculation of purity, contamination and significance '
                'only implemented for the multi-class case'
            )

        cont_sig = np.asarray(cont_sig)

        if cont_sig.ndim != 3 or cont_sig.shape[0] != cont_sig.shape[1]:
            raise ValueError(
                'cont_sig must have shape '
                '(n_classes, n_classes, n_thresholds)'
            )

        if cont_sig.shape[2] != threshold.size:
            raise ValueError(
                'threshold and cont_sig must have the same length'
            )

        n_classes = cont_sig.shape[0]

        if leg_labels is None or len(leg_labels) != n_classes:
            raise ValueError(
                f'leg_labels must contain one label for each class. '
                f'Number of labels provided: '
                f'{0 if leg_labels is None else len(leg_labels)}, '
                f'number of classes: {n_classes}'
            )

        if curves.shape[0] not in (1, n_classes):
            raise ValueError(
                'eff_sig must contain one curve or one curve for each class'
            )

        for target, target_label in enumerate(leg_labels):
            fig, (axis, axis_signif) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(8, 8),
                gridspec_kw={'height_ratios': [2, 1]}
            )

            # Main plot
            efficiency_index = 0 if curves.shape[0] == 1 else target

            axis.plot(
                threshold,
                curves[efficiency_index],
                '.-',
                color='maroon',
                label=f'{target_label} efficiency'
            )

            purity = 1 - np.sum(
                np.delete(cont_sig[target], target, axis=0),
                axis=0
            )

            axis.plot(
                threshold,
                purity,
                '.-',
                color='midnightblue',
                label=f'{target_label} Purity'
            )

            colorId = 0
            for source, source_label in enumerate(leg_labels):
                if source != target:
                    axis.plot(
                        threshold,
                        cont_sig[target, source],
                        '.--',
                        label=f'{source_label} contamination',
                        color=(
                            contaminColors[colorId]
                            if contaminColors is not None
                            else None
                        )
                    )
                    colorId += 1

            axis.set_ylabel('Efficiency / Purity')
            axis.set_title(
                f'Efficiency and Purity vs Score Threshold for {target_label}'
            )
            axis.grid(True)
            axis.legend()

            # Significance
            axis_signif.plot(
                threshold,
                signif_sig[target],
                '.-',
                color='maroon',
                label=f'"Asimov" Significance for {target_label} as signal'
            )

            axis_signif.set_xlabel(f'{target_label}-Score Threshold')
            axis_signif.set_ylabel('Significance')
            axis_signif.grid(True)
            axis_signif.legend()

            plt.tight_layout()
            res.append(fig)

    # Prompt / nonprompt efficiency vs BKG score threshold
    if bkg_efficiencies is not None:
        if len(bkg_efficiencies) != 3:
            raise ValueError(
                'bkg_efficiencies must contain prompt efficiency, '
                'nonprompt efficiency, and background rejection'
            )
        eff_prompt, eff_nonprompt, bkg_rejec = (
            np.asarray(values) for values in bkg_efficiencies
        )
        if any(values.shape != threshold.shape for values in
               (eff_prompt, eff_nonprompt, bkg_rejec)):
            raise ValueError(
                'BKG efficiency arrays must match the threshold shape'
            )

        fig, axis = plt.subplots(figsize=(8, 5))

        axis.plot(
            threshold,
            eff_prompt,
            '.-',
            color='maroon',
            label='Prompt efficiency'
        )

        axis.plot(
            threshold,
            eff_nonprompt,
            '.-',
            color='midnightblue',
            label='Non-prompt efficiency'
        )

        axis.plot(
            threshold,
            bkg_rejec,
            '.-',
            color='darkorange',
            label='Background rejection'
        )

        if chosen_bkg_rejec:
            axis.axvline(
                x=chosen_bkg_rejec,
                color='black',
                linestyle=':',
                linewidth=1.5,
                label=f'Threshold = {chosen_bkg_rejec:.2f}'
            )


        axis.set_xlabel('bkg-Score Threshold')
        axis.set_ylabel('Fraction')
        axis.set_title(
            'Background-rejection performance'
        )
        axis.grid(True)
        axis.legend()

        plt.tight_layout()
        res.append(fig)

    return res



def plot_bdt_efficiency_vs_var(var_centers, prompt_eff, prompt_err, nonprompt_eff, nonprompt_err, nonpCut, bkgCut, var_name = None, fig=None, ax=None):

    if fig is None and ax is None: # For only one nonpCut
        fig, ax = plt.subplots(figsize=(4, 5))

    ax.errorbar(
        var_centers,
        prompt_eff,
        yerr=prompt_err,
        marker="o",
        linestyle="-",
        color="maroon",
        label="Prompt",
    )

    ax.errorbar(
        var_centers,
        nonprompt_eff,
        yerr=nonprompt_err,
        marker="o",
        linestyle="-",
        color="darkblue",
        label="Non-prompt",
    )

    if var_name:
        ax.set_xlabel(var_name, fontsize=18)
    ax.set_ylabel("Efficiency", fontsize=18)
    ax.set_ylim(0, 1.05)

    ax.set_title(
        rf"non-prompt score $>$ {nonpCut}",
        fontsize=20
    )

    ax.legend(fontsize=18)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    # plt.show()
    return fig, ax
    











    # Multi-class classification case
    if leg_labels is None or len(leg_labels) != curves.shape[0]:
        raise ValueError(
            f'leg_labels must contain one label for each efficiency dimension. Number of labels provided: {0 if leg_labels is None else len(leg_labels)}, number of efficiency dimensions: {curves.shape[0]}')

    res = []
    for i_row, label in enumerate(leg_labels):
        values = curves[i_row]
        ylabel = 'Efficiency'
        res.append(make_plot(values, label, f'{ylabel} vs Score Threshold for {label}'))
    return res


def _plot_roc_ovr(y_truth, y_score, n_classes, labels, average):
    """
    Utility function for plot_roc in the multi-class case. Calculate and plot the
    ROC curves with the one-vs-rest approach
    """
    cmap = plt.get_cmap('tab10')
    # convert multi-class labels to multi-labels to obtain roc curves
    y_truth_multi = label_binarize(y_truth, classes=range(n_classes))
    for i_class, lab in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_truth_multi[:, i_class], y_score[:, i_class])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, c=cmap(i_class),
                 label=f'{lab} Vs Rest (AUC = {roc_auc:.4f})')
    # compute global ROC AUC
    global_roc_auc = roc_auc_score(
        y_truth, y_score, average=average, multi_class='ovr')
    plt.plot([], [], ' ', label=f'Average OvR ROC AUC: {global_roc_auc:.4f}')


def _plot_roc_ovo(y_truth, y_score, n_classes, labels, average):
    """
    Utility function for plot_roc in the multi-class case. Calculate and plot the
    ROC curves with the one-vs-one approach
    """
    cmap = plt.get_cmap('tab10')
    for i_comb, (aaa, bbb) in enumerate(combinations(range(n_classes), 2)):
        a_mask = y_truth == aaa
        b_mask = y_truth == bbb
        ab_mask = np.logical_or(a_mask, b_mask)
        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]
        fpr_a, tpr_a, _ = roc_curve(a_true, y_score[ab_mask, aaa])
        roc_auc_a = auc(fpr_a, tpr_a)
        fpr_b, tpr_b, _ = roc_curve(b_true, y_score[ab_mask, bbb])
        roc_auc_b = auc(fpr_b, tpr_b)
        plt.plot(fpr_a, tpr_a, lw=1, c=cmap(i_comb),
                 label=f'{labels[aaa]} Vs {labels[bbb]} (AUC = {roc_auc_a:.4f})')
        plt.plot(fpr_b, tpr_b, lw=1, c=cmap(i_comb), alpha=0.6,
                 label=f'{labels[bbb]} Vs {labels[aaa]} (AUC = {roc_auc_b:.4f})')
    # compute global ROC AUC
    global_roc_auc = roc_auc_score(
        y_truth, y_score, average=average, multi_class='ovo')
    plt.plot([], [], ' ', label=f'Average OvO ROC AUC: {global_roc_auc:.4f}')


def plot_roc(y_truth, y_score, pos_label=None, labels=None, average='macro', multi_class_opt='raise'):
    """
    Calculate and plot the roc curve

    Parameters
    -------------------------------------
    y_truth: array
        True labels for the belonging class. If labels are not
        {0, 1} in binary classification, then pos_label should
        be explicitly given. In multi-classification labels must
        be {0, 1, ..., N}

    y_score: array
        Target scores, can either be probability estimates or
        non-thresholded measure of decisions (as returned
        by “decision_function” on some classifiers).

    pos_label: int or str
        The label of the positive class. Only available in binary
        classification. When pos_label=None, if y_true is in {0, 1},
        pos_label is set to 1, otherwise an error will be raised.

    labels: list
        Contains the labels to be displayed in the legend, used only in case of
        multi-classification. They must be in the same order as the y_score columns.
        If None the labels are class1, class2, ..., classN

    average: string
        Option for the average of ROC AUC scores used only in case of multi-classification.
        You can choose between 'macro' and 'weighted'. For more information see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

    multi_class_opt: string
        Option to compute ROC curves used only in case of multi-classification.
        The one-vs-one 'ovo' and one-vs-rest 'ovr' approaches are available

    Returns
    -------------------------------------
    out: matplotlib.figure.Figure
        Plot containing the roc curves
    """
    # get number of classes
    n_classes = len(np.unique(y_truth))

    res = plt.figure()
    if n_classes <= 2:
        # binary case
        fpr, tpr, _ = roc_curve(y_truth, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f'ROC (AUC = {roc_auc:.4f})')

    else:
        # multi-class case
        if (labels is None) or (labels and len(labels) != n_classes):
            labels = [f'class{i_class}' for i_class in range(n_classes)]
        if multi_class_opt not in ['ovo', 'ovr']:
            print('ERROR: if n_class > 2 multi_class_opt must be ovo or ovr')
            return res

        # check to have numpy arrays
        if not isinstance(y_truth, np.ndarray):
            y_truth = np.array(y_truth)
        if not isinstance(y_score, np.ndarray):
            y_score = np.array(y_score)

        # if y_score contains raw outputs transform them to probabilities
        if not np.allclose(1, y_score.sum(axis=1)):
            y_score = softmax(y_score, axis=1)

        # one-vs-rest case
        if multi_class_opt == 'ovr':
            _plot_roc_ovr(y_truth, y_score, n_classes, labels, average)
        # one-vs-one case
        if multi_class_opt == 'ovo':
            _plot_roc_ovo(y_truth, y_score, n_classes, labels, average)

    plt.plot([0, 1], [0, 1], '-.', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid()

    return res


def plot_roc_train_test(y_truth_test, y_score_test, y_truth_train, y_score_train, pos_label=None, labels=None,
                        average='macro', multi_class_opt='raise'):
    """
    Calculate and plot the roc curve for test and train sets

    Parameters
    -------------------------------------
    y_truth_test: array
        True labels for the belonging class of the test set. If labels
        are not {0, 1} in binary classification, then pos_label should
        be explicitly given. In multi-classification labels must be
        {0, 1, ..., N}

    y_score_test: array
        Target scores for the test set, can either be probability
        estimates or non-thresholded measure of decisions (as returned
        by “decision_function” on some classifiers).

    y_truth_train: array
        True labels for the belonging class of the train set. If labels
        are not {0, 1} in binary classification, then pos_label should
        be explicitly given. In multi-classification labels must be
        {0, 1, ..., N}

    y_score_train: array
        Target scores for the train set, can either be probability
        estimates or non-thresholded measure of decisions (as returned
        by “decision_function” on some classifiers).

    pos_label: int or str
        The label of the positive class. Only available in binary
        classification. When pos_label=None, if y_true is in {0, 1},
        pos_label is set to 1, otherwise an error will be raised.

    labels: list
        Contains the labels to be displayed in the legend, used only in case of
        multi-classification. They must be in the same order as the y_score columns.
        If None the labels are class1, class2, ..., classN

    average: string
        Option for the average of ROC AUC scores used only in case of multi-classification.
        You can choose between 'macro' and 'weighted'. For more information see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

    multi_class_opt: string
        Option to compute ROC curves used only in case of multi-classification.
        The one-vs-one 'ovo' and one-vs-rest 'ovr' approaches are available

    Returns
    -------------------------------------
    out: matplotlib.figure.Figure
        Plot containing the roc curves
    """
    # call plot_roc for both train and test sets
    fig_test = plot_roc(y_truth_test, y_score_test,
                        pos_label, labels, average, multi_class_opt)
    fig_train = plot_roc(y_truth_train, y_score_train,
                         pos_label, labels, average, multi_class_opt)
    axes_test = fig_test.get_axes()[0]
    axes_train = fig_train.get_axes()[0]

    # plot results together
    res = plt.figure()
    for roc_test, roc_train in zip(axes_test.lines, axes_train.lines):
        test_label = roc_test.get_label()
        train_label = roc_train.get_label()
        if 'Luck' in [test_label, train_label]:
            continue

        plt.plot(roc_test.get_xdata(), roc_test.get_ydata(), lw=roc_test.get_lw(), c=roc_test.get_c(),
                 alpha=roc_test.get_alpha(), marker=roc_test.get_marker(), linestyle=roc_test.get_linestyle(),
                 label=f'Test -> {test_label}')

        linestyle = roc_train.get_linestyle()
        if linestyle in '-':
            linestyle = '--'
        plt.plot(roc_train.get_xdata(), roc_train.get_ydata(), lw=roc_train.get_lw(), c=roc_train.get_c(),
                 alpha=roc_train.get_alpha(), marker=roc_train.get_marker(), linestyle=linestyle,
                 label=f'Train -> {train_label}')

    plt.plot([0, 1], [0, 1], '-.', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid()

    return res


# def plot_feature_imp(df_in, y_truth, model, labels=None, n_sample=10000, approximate=True):
#     """
#     Calculate the feature importance using the shap algorithm for
#     each feature. The calculation is performed on a subsample of the
#     input training/test set

#     Parameters
#     -------------------------------------------
#     df_in: Pandas dataframe
#         Training or test set dataframe

#     y_truth: array
#         Training or test set label

#     model: hipe4ml model_handler
#         Trained model

#     labels: list
#         Contains the labels to be displayed in the legend
#         If None the labels are class1, class2, ..., classN

#     n_sample: int
#         Number of candidates employed to fill
#         the shap plots. If larger than the number of
#         candidates in each class, minimum number of candidates
#         in a given class used instead

#     approximate: bool
#         Run fast and approximat roughly the SHAP values. For more information
#         see https://shap.readthedocs.io/en/latest/#shap.TreeExplainer.shap_values

#     Returns
#     -------------------------------------------
#     out: List of matplotlib.figure.Figure
#         Plots with shap feature importance. The first ones are the shap violin plots
#         computed for each class(in case of binary classification only one plot is returned).
#         The last plot of the list is built by taking the mean absolute value of the SHAP
#         values for each feature to get a standard bar plot (stacked bars are produced for
#         multi-class outputs)

#     """
#     class_labels, class_counts = np.unique(y_truth, return_counts=True)
#     n_classes = len(class_labels)
#     for class_count in class_counts:
#         n_sample = min(n_sample, class_count)

#     subs = []
#     for class_lab in class_labels:
#         subs.append(df_in[y_truth == class_lab].sample(n_sample))

#     df_subs = pd.concat(subs)
#     df_subs = df_subs[model.get_training_columns()]
#     explainer = shap.TreeExplainer(model.get_original_model())
#     shap_values = explainer.shap_values(df_subs, approximate=approximate)
#     res = []

#     if n_classes <= 2:
#         res.append(plt.figure(figsize=(18, 9)))
#         shap.summary_plot(shap_values, df_subs, plot_size=(
#             18, 9), class_names=labels, show=False)
#         res.append(plt.figure(figsize=(18, 9)))
#         shap.summary_plot(shap_values, df_subs, plot_type='bar', plot_size=(
#             18, 9), class_names=labels, show=False)
#     else:
#         shap_values_transposed = shap_values.transpose(2, 0, 1)
#         for i_class in range(n_classes):
#             res.append(plt.figure(figsize=(18, 9)))
#             shap.summary_plot(shap_values_transposed[i_class], df_subs, plot_size=(
#                 18, 9), class_names=labels, show=False)
#         res.append(plt.figure(figsize=(18, 9)))
#         shap.summary_plot(list(shap_values_transposed), df_subs, plot_type='bar', plot_size=(
#             18, 9), class_names=labels, show=False)

#     return res




def plot_feature_imp(
        df_in,
        y_truth,
        model,
        labels=None,
        n_sample=10000,
        approximate=True,
        features=None):
    """
    Calculate the feature importance using the SHAP algorithm for each feature.
    The calculation is performed on a subsample of the input training/test set.

    Parameters
    -------------------------------------------
    df_in: Pandas dataframe
        Training or test set dataframe.

    y_truth: array
        Training or test set label.

    model: hipe4ml model_handler
        Trained model.

    labels: list
        Contains the labels to be displayed in the legend.
        If None the labels are class1, class2, ..., classN.

    n_sample: int
        Number of candidates employed to fill the SHAP plots.
        If larger than the number of candidates in each class,
        the minimum number of candidates in a given class is used instead.

    approximate: bool
        Run fast and approximate SHAP values.
        For more information, see:
        https://shap.readthedocs.io/en/latest/#shap.TreeExplainer.shap_values

    features: list or None
        List of feature names to display in the SHAP plots.
        If None, all training features are displayed.

        Note that the model is always evaluated using all its training
        features; this option only controls which features are shown
        in the final plots.

    Returns
    -------------------------------------------
    out: List of matplotlib.figure.Figure
        Plots with SHAP feature importance.

        For binary classification:
            - SHAP summary (beeswarm) plot
            - SHAP summary bar plot

        For multi-class classification:
            - One SHAP summary plot per class
            - One combined SHAP summary bar plot
    """

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------
    class_labels, class_counts = np.unique(y_truth, return_counts=True)
    n_classes = len(class_labels)

    # Limit the number of candidates to the smallest class if necessary
    n_sample = min(n_sample, np.min(class_counts))

    # ------------------------------------------------------------------
    # Subsample the input dataframe, keeping the same number of
    # candidates from each class
    # ------------------------------------------------------------------
    subs = []

    for class_lab in class_labels:
        class_df = df_in[y_truth == class_lab]
        subs.append(class_df.sample(n_sample))

    df_subs = pd.concat(subs)

    # ------------------------------------------------------------------
    # Get the columns actually used by the trained model
    # ------------------------------------------------------------------
    training_columns = model.get_training_columns()

    df_subs = df_subs[training_columns]

    # ------------------------------------------------------------------
    # Check requested plotting features
    # ------------------------------------------------------------------
    if features is None:
        plot_features = training_columns
    else:
        missing_features = [
            feature for feature in features
            if feature not in training_columns
        ]

        if missing_features:
            raise ValueError(
                f"The following features are not training features: "
                f"{missing_features}"
            )

        # Preserve the order supplied by the user
        plot_features = list(features)

    # Indices of the selected features in the training feature list
    plot_indices = [
        training_columns.index(feature)
        for feature in plot_features
    ]

    # Dataframe containing only the variables shown in the plots
    df_plot = df_subs[plot_features]

    # ------------------------------------------------------------------
    # Calculate SHAP values using ALL features
    # ------------------------------------------------------------------
    explainer = shap.TreeExplainer(model.get_original_model())

    shap_values = explainer.shap_values(
        df_subs,
        approximate=approximate
    )

    res = []

    # ------------------------------------------------------------------
    # Binary classification
    # ------------------------------------------------------------------
    if n_classes <= 2:

        # --------------------------------------------------------------
        # SHAP beeswarm plot
        # --------------------------------------------------------------
        fig = plt.figure(figsize=(18, 9))

        # SHAP can return either a list or an array depending on
        # the SHAP/model/version combination
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0][:, plot_indices]
        else:
            shap_values_array = np.asarray(shap_values)

            if shap_values_array.ndim == 3:
                # Keep the first class, consistent with the original
                shap_values_plot = shap_values_array[:, plot_indices, 0]
            else:
                shap_values_plot = shap_values_array[:, plot_indices]

        shap.summary_plot(
            shap_values_plot,
            df_plot,
            plot_size=(18, 9),
            class_names=labels,
            show=False
        )

        title = "SHAP Feature Importance — Summary"
        if labels is not None and len(labels) > 0:
            title += f" ({labels[0]})"

        plt.title(title, fontsize=18, pad=15)
        plt.tight_layout()

        res.append(fig)

        # --------------------------------------------------------------
        # SHAP bar plot
        # --------------------------------------------------------------
        fig = plt.figure(figsize=(18, 9))

        shap.summary_plot(
            shap_values_plot,
            df_plot,
            plot_type="bar",
            plot_size=(18, 9),
            class_names=labels,
            show=False
        )

        plt.title(
            "SHAP Feature Importance — Mean |SHAP|",
            fontsize=18,
            pad=15
        )
        plt.tight_layout()

        res.append(fig)

    # ------------------------------------------------------------------
    # Multi-class classification
    # ------------------------------------------------------------------
    else:

        shap_values_array = np.asarray(shap_values)

        # Expected shape:
        # (n_samples, n_features, n_classes)
        if shap_values_array.ndim != 3:
            raise ValueError(
                f"Unexpected SHAP array shape for multiclass model: "
                f"{shap_values_array.shape}"
            )

        shap_values_transposed = shap_values_array.transpose(2, 0, 1)

        # --------------------------------------------------------------
        # One beeswarm plot for each class
        # --------------------------------------------------------------
        for i_class in range(n_classes):

            fig = plt.figure(figsize=(18, 9))

            shap.summary_plot(
                shap_values_transposed[i_class][:, plot_indices],
                df_plot,
                plot_size=(18, 9),
                class_names=labels,
                show=False
            )

            if labels is not None and i_class < len(labels):
                class_name = labels[i_class]
                title = (
                    f"SHAP Feature Importance — "
                    f"Class {i_class}: {class_name}"
                )
            else:
                title = (
                    f"SHAP Feature Importance — "
                    f"Class {i_class}"
                )

            plt.title(title, fontsize=18, pad=15)
            plt.tight_layout()

            res.append(fig)

        # --------------------------------------------------------------
        # Combined bar plot
        # --------------------------------------------------------------
        fig = plt.figure(figsize=(18, 9))

        shap_values_selected = [
            shap_values_transposed[i_class][:, plot_indices]
            for i_class in range(n_classes)
        ]

        shap.summary_plot(
            shap_values_selected,
            df_plot,
            plot_type="bar",
            plot_size=(18, 9),
            class_names=labels,
            show=False
        )

        plt.title(
            "SHAP Feature Importance — Mean |SHAP| by Class",
            fontsize=18,
            pad=15
        )
        plt.tight_layout()

        res.append(fig)

    return res




def plot_precision_recall(y_truth, y_score, labels=None, pos_label=None):
    """ Plot precision recall curve

    Parameters
    -------------------------------------
    y_truth: array
        True labels for the belonging class. If labels are not
        {0, 1, ..., N}, then pos_label should be explicitly given.

    y_score: array
        Estimated probabilities or decision function.

    pos_label : int or str
        The label of the positive class. When pos_label=None,
        if y_true is in {0, 1, ..., N}, pos_label is set to 1,
        otherwise an error will be raised.

    Returns
    -------------------------------------
    out: matplotlib.figure.Figure
        Plot containing the precision recall curves
    """
    # get number of classes
    n_classes = len(np.unique(y_truth))

    if (labels is None and n_classes > 2) or (labels and len(labels) != n_classes):
        labels = []
        for i_class in range(n_classes):
            labels.append(f'class{i_class}')

    res = plt.figure()
    if n_classes <= 2:
        precision, recall, _ = precision_recall_curve(
            y_truth, y_score, pos_label=pos_label)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
        average_precision = average_precision_score(y_truth, y_score)
        plt.title(
            f'2-class Precision-Recall curve: AP={average_precision:0.2f}')
    else:
        cmap = plt.get_cmap('tab10')
        precision, recall = ({}, {})
        # convert multi-class labels to multi-labels to obtain a curve for each class
        y_truth_multi = label_binarize(y_truth, classes=range(n_classes))
        for clas, lab in enumerate(labels):
            precision[clas], recall[clas], _ = precision_recall_curve(
                y_truth_multi[:, clas], y_score[:, clas], pos_label=pos_label)
            plt.step(recall[clas], precision[clas], color=cmap(clas), lw=1, where='post',
                     label=lab)
        # compute also micro average
        precision['micro'], recall['micro'], _ = precision_recall_curve(
            y_truth_multi.ravel(), y_score.ravel())
        plt.step(recall['micro'], precision['micro'], color='black', where='post',
                 linestyle='--', lw=1, label='average')
        average_precision = average_precision_score(
            y_truth_multi, y_score, average='micro')
        plt.title(
            f'Average precision score, micro-averaged over all classes: {average_precision:0.2f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    if n_classes > 2:
        plt.legend(loc='lower left')
        plt.grid()
    return res


def plot_learning_curves(model, data, n_points=10, labels=None):
    """ Plot learning curves

    Parameters
    -------------------------------------
    model: hipe4ml model_handler

    data: list
        Contains respectively: training
        set dataframe, training label array,
        test set dataframe, test label array

    n_points: int
        Number of points used to sample the learning curves

    labels: list
        Contains the labels to be displayed in the legend, used only in case of multi-classification. They must be in the same order as the 'data' columns.
        If None the labels are class-1, class-2, ..., class-N

    Returns
    -------------------------------------
    out: matplotlib.figure.Figure
        Plot containing the learning curves
    """

    class_labels = np.unique(data[1])
    n_classes = len(class_labels)

    if n_classes < 2:
        res = plt.figure()
        train_errors, test_errors = [], []
        min_cand = 100
        max_cand = len(data[0])
        step = int((max_cand-min_cand)/n_points)
        array_n_cand = np.arange(start=min_cand, stop=max_cand, step=step)
        for n_cand in array_n_cand:
            model.fit(data[0][:n_cand], data[1][:n_cand])
            y_train_predict = model.predict(data[0][:n_cand], output_margin=False)
            y_test_predict = model.predict(data[2], output_margin=False)
            train_errors.append(mean_squared_error(
                y_train_predict, data[1][:n_cand], multioutput='uniform_average'))
            test_errors.append(mean_squared_error(
                y_test_predict, data[3], multioutput='uniform_average'))
        plt.plot(array_n_cand, np.sqrt(train_errors), 'r', lw=1, label='Train')
        plt.plot(array_n_cand, np.sqrt(test_errors), 'b', lw=1, label='Test')
        plt.ylim([0, np.amax(np.sqrt(test_errors))*2])
        plt.xlabel('Training set size')
        plt.ylabel('RMSE')
        plt.grid()
        plt.legend(loc='best')

        return res

    # Multi-class case: one subplot for each class
    res = plt.figure(figsize=(8, 6 * n_classes))

    for labelId in class_labels:
        ax = res.add_subplot(n_classes, 1, labelId + 1)

        train_errors, test_errors = [], []

        min_cand = 100
        max_cand = len(data[0])
        step = int((max_cand - min_cand) / n_points)
        array_n_cand = np.arange(min_cand, max_cand, step=step)

        for n_cand in array_n_cand:
            model.fit(data[0][:n_cand], data[1][:n_cand])

            y_train_predict = model.predict(
                data[0][:n_cand], output_margin=False
            )
            y_test_predict = model.predict(
                data[2], output_margin=False
            )

            train_errors.append(
                mean_squared_error(
                    y_train_predict[:, labelId],
                    data[1][:n_cand],
                    multioutput='uniform_average'
                )
            )

            test_errors.append(
                mean_squared_error(
                    y_test_predict[:, labelId],
                    data[3],
                    multioutput='uniform_average'
                )
            )

        ax.plot(array_n_cand, np.sqrt(train_errors),
                'r', lw=1, label='Train')
        ax.plot(array_n_cand, np.sqrt(test_errors),
                'b', lw=1, label='Test')

        ax.set_ylim([0, np.amax(np.sqrt(test_errors)) * 2])
        ax.set_xlabel('Training set size')
        ax.set_ylabel(f'{labels[labelId] if labels else f"Class-{labelId}"} RMSE')
        ax.grid()
        ax.legend(loc='best')

    return res

