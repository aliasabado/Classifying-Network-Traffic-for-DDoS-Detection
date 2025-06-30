import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

def load_arff(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    df['PKT_CLASS'] = df['PKT_CLASS'].str.decode('utf-8')
    df['is_attack'] = (df['PKT_CLASS'] != 'Normal').astype(int)
    return df

def plot_all_features(df, features):
    num_features = len(features)
    n_cols = 3
    n_rows = (num_features + n_cols - 1) // n_cols
    plt.figure(figsize=(16, 12))

    #plot each graph 
    for idx, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, idx + 1)
        
#ozoom in the axis
        lower = df[feature].quantile(0.01)
        upper = df[feature].quantile(0.99)
        
        sns.kdeplot(df[df['is_attack'] == 0][feature], label='Normal', shade=True)
        sns.kdeplot(df[df['is_attack'] == 1][feature], label='Attack', shade=True)
        plt.title(feature)
        plt.xlim(lower, upper)  
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    file_path = 'final-dataset.arff' 
    df = load_arff(file_path)

    features = [
        'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE',                               
        'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 
        'UTILIZATION', 'PKT_DELAY', 'PKT_IN', 'PKT_OUT'
    ]

    plot_all_features(df, features)

if __name__ == "__main__":
    main()
