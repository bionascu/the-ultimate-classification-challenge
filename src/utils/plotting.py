import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.paths import data_processed_dir

def plot_categ_distrib(num_bins = 100):

    cat_id_distrib = pd.read_pickle(os.path.join(data_processed_dir, 'cat_id_counts.pickle'))
    cat_1_distrib = pd.read_pickle(os.path.join(data_processed_dir, 'cat_1_counts.pickle'))
    cat_2_distrib = pd.read_pickle(os.path.join(data_processed_dir, 'cat_2_counts.pickle'))
    cat_3_distrib = pd.read_pickle(os.path.join(data_processed_dir, 'cat_3_counts.pickle'))

    fig = plt.figure(figsize=(15, 10))
    n, bins = np.histogram(cat_id_distrib, bins=num_bins)
    bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
    ax1 = fig.add_subplot(221)
    ax1.scatter(bins_mean, n)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('number of products per category id')
    ax1.set_ylabel('number of category ids')

    n, bins = np.histogram(cat_1_distrib, bins=num_bins)
    bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
    ax2 = fig.add_subplot(222)
    ax2.scatter(bins_mean, n)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('number of products per category level 1')
    ax2.set_ylabel("number of category level 1's")

    fig = plt.figure()
    n, bins = np.histogram(cat_2_distrib, bins=num_bins)
    bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
    ax3 = fig.add_subplot(221)
    ax3.scatter(bins_mean, n)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_xlabel('number of products per category level 2')
    ax3.set_ylabel("number of category level 2's")

    n, bins = np.histogram(cat_3_distrib, bins=num_bins)
    bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
    ax4 = fig.add_subplot(222)
    ax4.scatter(bins_mean, n)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_xlabel('number of products per category level 3')
    ax4.set_ylabel("number of category level 3's")

