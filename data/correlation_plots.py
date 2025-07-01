import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import math

# loads data
def load_data(file, features):
    data = pd.read_csv(file)
    df = data[features + ['PKT_CLASS']]

    return df


# plots correlation matrix
def corr_matrix(data, features):
    plt.figure(figsize=(10, 6))
    corr = data[features].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu")
    plt.title("Correlation Matrix of each Feature")

    plt.savefig("corr_matrix.png")
    plt.close()


# returns feature pairs with low correlation value
def low_corr_pairs(data, features):   
    corr = data[features].corr()
    pairs = list(combinations(features, 2))
    threshold = 0.3 # max correlation value

    low_pairs = [(x, y) for x, y in pairs if abs(corr.loc[x, y]) < threshold]

    return low_pairs


# scatterplot matrix of weakly correlated features
def corr_scatter(data, pairs):
    fig = plt.figure(figsize=(20, 15))
    num_pairs = len(pairs)
    n_col = 6
    n_row = math.ceil(num_pairs/6)

    handles, labels = None, None
    
    for i, (x,y) in enumerate(pairs):
        ax = plt.subplot(n_row, n_col, i+1)
        sns.scatterplot(data=data, x=x, y=y, hue="PKT_CLASS", s=10, alpha=0.6)
        ax.set_title(f"{x} vs {y}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().set_visible(False)
        
        if handles is None or labels is None: # get legend data
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower right', bbox_to_anchor = (.95, 0.05))

    plt.tight_layout()
    plt.savefig("corr_scatterplot.png")
    plt.close()


def main():
    file = "sample-data.csv"
    features = [
        'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE',                               
        'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 
        'UTILIZATION', 'PKT_DELAY', 'PKT_IN', 'PKT_OUT'
    ]

    data = load_data(file, features)

    corr_matrix(data, features)
    low_pairs = low_corr_pairs(data, features)
    corr_scatter(data, low_pairs) # might take a while to load


if __name__ == "__main__":
    main()
    