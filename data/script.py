import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.io import arff
import numpy as np

# https://www.geeksforgeeks.org/how-to-automate-data-cleaning-in-python/

def load_arff_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    str_columns = ['PKT_TYPE', 'FLAGS', 'NODE_NAME_FROM', 'NODE_NAME_TO', 'PKT_CLASS']
    for col in str_columns:
        if col in df.columns:
            df[col] = df[col].str.decode('utf-8')
    
    return df

def load_csv_data(file_path):
    df = pd.read_csv(file_path)

    return df

def drop_duplicates(df, column_name=None):
    rows_before = df.shape[0]
    print(f"Dataset has {rows_before} rows before removing duplicates")
    if column_name:
        df = df.drop_duplicates(subset=[column_name])
        print(f"Removed duplicates based on column: {column_name}")
    else:
        df = df.drop_duplicates()

    rows_after = df.shape[0]
    rows_removed = rows_before - rows_after
    
    if rows_removed > 0:
        print(f"Removed {rows_removed} duplicate rows")
        print(f"Dataset now has {rows_after} rows")
    else:
        print("No duplicate rows found in the dataset")
    
    return df

def encode_categorical(df, columns_to_encode):
    label_encoders = {}
    for column in columns_to_encode:
            le = LabelEncoder()
            df[column + '_encoded'] = le.fit_transform(df[column])
            label_encoders[column] = le                                                                                      #https://www.geeksforgeeks.org/machine-learning/label-encoding-across-multiple-columns-in-scikit-learn/
            print(f"Encoded {column}: {len(le.classes_)} unique values")
    
    return df, label_encoders

def making_it_binary(df, target_column='PKT_CLASS'):
    df['is_attack'] = (df[target_column] != 'Normal').astype(int)
    
    normal_count = sum(df['is_attack'] == 0)
    attack_count = sum(df['is_attack'] == 1)
    
    print(f"Binary output created:")
    print(f"Normal traffic: {normal_count}")
    print(f"Attack traffic: {attack_count}")
    
    return df

def outlier_handling(df, column_with_outliers):
    initial_shape = df.shape[0]
    
    q1 = df[column_with_outliers].quantile(0.25)
    q3 = df[column_with_outliers].quantile(0.75)
    iqr = q3 - q1
    
    df = df[(df[column_with_outliers] > (q1 - 1.5 * iqr)) 
            & (df[column_with_outliers] < (q3 + 1.5 * iqr))]
    
    removed = initial_shape - df.shape[0]
    print(f"Removed {removed} outliers from {column_with_outliers}")
    
    return df

def check_missing_values(df):
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    if total_missing > 0:
        print(f"Found {total_missing} missing values in the dataset:")
        print(missing_data[missing_data > 0])
        columns_with_missing = missing_data[missing_data > 0].index.tolist()
        print(f"Columns with missing data: {columns_with_missing}")
        
    else:
        print("No missing data found in the dataset!")
    
    return df

def select_features(df):
# These are the features I have selected for now
    important_features = [
        'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE',                               
        'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 
        'UTILIZATION', 'PKT_DELAY', 'PKT_IN', 'PKT_OUT',
        'PKT_TYPE_encoded', 'FLAGS_encoded'
    ]
    features_we_can_use = []
    for feature in important_features:
        if feature in df.columns:
            features_we_can_use.append(feature)
    
    print(f"Selected {len(features_we_can_use)} features for detecting attacks")
    return features_we_can_use

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled")
    return X_scaled, scaler

def network_attack_preprocessing_pipeline(file_path):
    print("Starting Network Attack Data Preprocessing Pipeline...")
    print("=" * 50)
    print("1. Loading data...")  
    df = load_csv_data(file_path)  # modify to either ARFF or CSV file
    print(f"Dataset loaded: {df.shape}")

    print("\n2. Removing duplicates...")
    df_no_duplicates = drop_duplicates(df)

    print("\n3. Handling missing values...")
    df_no_nulls = check_missing_values(df_no_duplicates)
    
   
    print("\n4. Making output binary...")
    df_with_target = making_it_binary(df_no_nulls)
    
    print("\n5. Encoding categorical variables...")
    categorical_columns = ['PKT_TYPE', 'FLAGS', 'NODE_NAME_FROM', 'NODE_NAME_TO']
    df_encoded, label_encoders = encode_categorical(df_with_target, categorical_columns)
    
    print("\n6. Handling outliers (optional)...")
    # Uncomment if you want to remove outliers
    # df_no_outliers = outlier_handling(df_encoded, 'PKT_SIZE')
    df_no_outliers = df_encoded  # Skipping outlier removal for now cause outlier could be an attack
    
    
    print("\n7. Selecting features...")
    feature_columns = select_features(df_no_outliers)
    
   
    print("\n8. Preparing final dataset...")
    X = df_encoded[feature_columns]
    y = df_encoded['is_attack']
    

    print("\n9. Scaling features...")
    X_scaled, scaler = scale_features(X)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE!")
    print(f"Final dataset shape (Rows and column): {X_scaled.shape}")
    print(f"Output distribution (How many are normal and how many are attacks): {np.bincount(y)}")
    
    return {
        'X': X_scaled,
        'y': y.values
    }

if __name__ == "__main__":
    #Change path here to test on short file
    file_path = 'sample-data.csv'
 
    results = network_attack_preprocessing_pipeline(file_path)
    np.save('X_processed.npy', results['X'])
    np.save('y_processed.npy', results['y'])
            
    print("\nProcessed data saved as .npy files") 