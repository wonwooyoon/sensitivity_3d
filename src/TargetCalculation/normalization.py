import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

target_df = pd.read_csv('./src/TargetCalculation/output/inout.csv')

scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(target_df)
normalized_df = pd.DataFrame(normalized_data, columns=target_df.columns)
normalized_output_csv_path = './src/TargetCalculation/output/normalized_inout.csv'
normalized_df.to_csv(normalized_output_csv_path, index=False)
scaler_path = './src/TargetCalculation/output/minmax_scaler.pkl'
joblib.dump(scaler, scaler_path)
    
print("Scaler data min:", scaler.data_min_)
print("Scaler data max:", scaler.data_max_)