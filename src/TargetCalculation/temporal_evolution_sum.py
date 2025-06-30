import os
import pandas as pd

aq_uo2_granite = pd.DataFrame()
aq_uo2_bentonite = pd.DataFrame()
aq_uo2_source = pd.DataFrame()
aq_uo2_bentonite_adsorbed = pd.DataFrame()
aq_uo2_source_mineralized = pd.DataFrame()
aq_uo2_effluxed = pd.DataFrame()

num_rows = 101

# Prepare lists to collect DataFrames for each sample
granite_list = []
bentonite_list = []
source_list = []
bentonite_adsorbed_list = []
source_mineralized_list = []
effluxed_list = []

for i in range(1, 340):
    target_csv_path = f'./src/TargetCalculation/output/sample_{i}/target_values.csv'
    col_name = f'data_{i}'

    if os.path.exists(target_csv_path):
        target_data = pd.read_csv(target_csv_path)
        required_columns = ['Aqueous UO2++ in Granite', 'Aqueous UO2++ in Bentonite', 
                            'Aqueous UO2++ in Source', 'Adsorbed UO2++ in Bentonite', 
                            'Mineralized UO2++ in Source', 'Effluxed UO2++']
        if all(col in target_data.columns for col in required_columns):
            granite_list.append(target_data['Aqueous UO2++ in Granite'].rename(col_name))
            bentonite_list.append(target_data['Aqueous UO2++ in Bentonite'].rename(col_name))
            source_list.append(target_data['Aqueous UO2++ in Source'].rename(col_name))
            bentonite_adsorbed_list.append(target_data['Adsorbed UO2++ in Bentonite'].rename(col_name))
            source_mineralized_list.append(target_data['Mineralized UO2++ in Source'].rename(col_name))
            effluxed_list.append(target_data['Effluxed UO2++'].rename(col_name))
        else:
            granite_list.append(pd.Series([0]*num_rows, name=col_name))
            bentonite_list.append(pd.Series([0]*num_rows, name=col_name))
            source_list.append(pd.Series([0]*num_rows, name=col_name))
            bentonite_adsorbed_list.append(pd.Series([0]*num_rows, name=col_name))
            source_mineralized_list.append(pd.Series([0]*num_rows, name=col_name))
            effluxed_list.append(pd.Series([0]*num_rows, name=col_name))
    else:
        granite_list.append(pd.Series([0]*num_rows, name=col_name))
        bentonite_list.append(pd.Series([0]*num_rows, name=col_name))
        source_list.append(pd.Series([0]*num_rows, name=col_name))
        bentonite_adsorbed_list.append(pd.Series([0]*num_rows, name=col_name))
        source_mineralized_list.append(pd.Series([0]*num_rows, name=col_name))
        effluxed_list.append(pd.Series([0]*num_rows, name=col_name))

# Concatenate all columns for each DataFrame
aq_uo2_granite = pd.concat(granite_list, axis=1)
aq_uo2_bentonite = pd.concat(bentonite_list, axis=1)
aq_uo2_source = pd.concat(source_list, axis=1)
aq_uo2_bentonite_adsorbed = pd.concat(bentonite_adsorbed_list, axis=1)
aq_uo2_source_mineralized = pd.concat(source_mineralized_list, axis=1)
aq_uo2_effluxed = pd.concat(effluxed_list, axis=1)
        

# Save the DataFrames to CSV files
aq_uo2_granite.to_csv('./src/TargetCalculation/output/aq_uo2_granite.csv', index=False)
aq_uo2_bentonite.to_csv('./src/TargetCalculation/output/aq_uo2_bentonite.csv', index=False)
aq_uo2_source.to_csv('./src/TargetCalculation/output/aq_uo2_source.csv', index=False)
aq_uo2_bentonite_adsorbed.to_csv('./src/TargetCalculation/output/aq_uo2_bentonite_adsorbed.csv', index=False)
aq_uo2_source_mineralized.to_csv('./src/TargetCalculation/output/aq_uo2_source_mineralized.csv', index=False)
aq_uo2_effluxed.to_csv('./src/TargetCalculation/output/aq_uo2_effluxed.csv', index=False)
