import pandas as pd
import os
from matplotlib import font_manager
from sklearn.preprocessing import MinMaxScaler
import joblib

font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf')

class TargetValueAnalysis:

    def __init__(self):
        pass
        

    def read_path(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            return 1
        except FileNotFoundError:
            return 0
            
    def calculate_aqueous(self):
        material_1 = self.data[self.data['Material ID'] == 1]
        self.aqueous_granite = (1000 * material_1['Total UO2++ [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        
        material_2 = self.data[self.data['Material ID'] == 2]
        self.aqueous_bentonite = (1000 * material_2['Total UO2++ [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()

        material_3 = self.data[self.data['Material ID'] == 3]
        self.aqueous_source = (1000 * material_3['Total UO2++ [M]'] * material_3['Volume [m^3]'] * material_3['Porosity']).sum()
    
    def calculate_adsorbed(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        self.adsorbed = (material_2['Total Sorbed UO2++ [mol_m^3]'] * material_2['Volume [m^3]']).sum()

    def calculate_mineral(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        material_3 = self.data[self.data['Material ID'] == 3]
        
        self.mineral_bent = (material_2['UO2:2H2O(am) VF [m^3 mnrl_m^3 bulk]'] * material_2['Volume [m^3]'] * 2000).sum()
        self.mineral_sour = (material_3['UO2:2H2O(am) VF [m^3 mnrl_m^3 bulk]'] * material_3['Volume [m^3]'] * 2000).sum()

    def calculate_aq_speciation(self):
        material_1 = self.data[self.data['Material ID'] == 1]
        material_2 = self.data[self.data['Material ID'] == 2]
        
        self.m1_a1 = (material_1['Ca2UO2(CO3)3 [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        self.m1_a2 = (material_1['CaUO2(CO3)3-- [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        self.m1_a3 = (material_1['UO2(CO3)3---- [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        self.m1_a4 = (material_1['MgUO2(CO3)3-- [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        self.m1_a5 = (material_1['UO2(CO3)2-- [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()

        self.m2_a1 = (material_2['Ca2UO2(CO3)3 [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
        self.m2_a2 = (material_2['CaUO2(CO3)3-- [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
        self.m2_a3 = (material_2['UO2(CO3)3---- [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
        self.m2_a4 = (material_2['MgUO2(CO3)3-- [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
        self.m2_a5 = (material_2['UO2(CO3)2-- [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()


    def calculate_ad_speciation(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        self.m2_s1 = (material_2['>SOUO2+ [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_s2 = (material_2['>SOUO2OH [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_s3 = (material_2['>SOUO2(OH)2- [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_s4 = (material_2['>SOUO2(OH)3-- [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_s5 = (material_2['>SOUO2CO3- [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_s6 = (material_2['>SOUO2(CO3)2--- [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_w1 = (material_2['>WOUO2+ [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_w2 = (material_2['>WOUO2OH [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()
        self.m2_w3 = (material_2['>WOUO2CO3- [mol_m^3 bulk]'] * material_2['Volume [m^3]']).sum()

    def calculate_components(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        self.calcium = (material_2['Total Ca++ [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
        self.carbonate = (material_2['Total CO3-- [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
        self.mat2_vol = (material_2['Volume [m^3]'] * material_2['Porosity']).sum()
        self.calcium_conc = self.calcium / self.mat2_vol
        self.carbonate_conc = self.carbonate / self.mat2_vol

        material_1 = self.data[self.data['Material ID'] == 1]
        self.calcium_frac = (material_1['Total Ca++ [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        self.carbonate_frac = (material_1['Total CO3-- [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        self.mat1_vol = (material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        self.calcium_frac_conc = self.calcium_frac / self.mat1_vol
        self.carbonate_frac_conc = self.carbonate_frac / self.mat1_vol 
    
    def calculate_inout(self, inout_path):
                
            # Read the CSV file into a DataFrame
            inout_data = pd.read_csv(inout_path, delim_whitespace=True, header=None, skiprows=1)
            # Read the first row of the file as a string
            with open(inout_path, 'r') as file:
                inout_header = file.readline().strip()
            # Remove all double quotes from the header and split by commas
            inout_header = inout_header.replace('"', '').split(',')

            # Assign the processed header to the DataFrame
            inout_data.columns = inout_header

            # Ensure the required columns exist
            if 'OUTLET UO2++ [mol]' in inout_data.columns and 'INLET UO2++ [mol]' in inout_data.columns:
                # Calculate the result as the difference between inlet and outlet
                inout_data['Result'] = -inout_data['INLET UO2++ [mol]'] - inout_data['OUTLET UO2++ [mol]']

                # Sample every 100 rows from the result column
                self.inout = pd.concat([pd.Series([0]), inout_data['Result'].iloc[99::100]], ignore_index=True)
                self.inout = self.inout.to_frame(name='Effluxed UO2++')  # Ensure self.inout has a proper header
            else:
                raise ValueError("Required columns 'OUTLET UO2++ [mol]' and 'INLET UO2++ [mol]' are missing in the input file.")

            self.target_values = pd.concat([self.target_values, self.inout], axis=1)  # Concatenate with proper header

            
    def save_target_values(self):
        target_values = pd.DataFrame({'Aqueous UO2++ in Granite': [self.aqueous_granite], 
                                      'Aqueous UO2++ in Bentonite': [self.aqueous_bentonite], 
                                      'Aqueous UO2++ in Source': [self.aqueous_source],
                                      'Adsorbed UO2++ in Bentonite': [self.adsorbed], 
                                      'Mineralized UO2++ in Bentonite': [self.mineral_bent], 
                                      'Mineralized UO2++ in Source': [self.mineral_sour], 
                                      'm1_a1': [self.m1_a1], 
                                      'm1_a2': [self.m1_a2],
                                      'm1_a3': [self.m1_a3],
                                      'm1_a4': [self.m1_a4],
                                      'm1_a5': [self.m1_a5],
                                      'm2_a1': [self.m2_a1],
                                      'm2_a2': [self.m2_a2],
                                      'm2_a3': [self.m2_a3],
                                      'm2_a4': [self.m2_a4],
                                      'm2_a5': [self.m2_a5],
                                      'm2_s1': [self.m2_s1],
                                      'm2_s2': [self.m2_s2],
                                      'm2_s3': [self.m2_s3],
                                      'm2_s4': [self.m2_s4],
                                      'm2_s5': [self.m2_s5],
                                      'm2_s6': [self.m2_s6],
                                      'm2_w1': [self.m2_w1],
                                      'm2_w2': [self.m2_w2],
                                      'm2_w3': [self.m2_w3],
                                      'Calcium': [self.calcium],
                                      'Carbonate': [self.carbonate],
                                      'Calcium_frac': [self.calcium_frac],
                                      'Carbonate_frac': [self.carbonate_frac],
                                      'Calcium_conc': [self.calcium_conc],
                                      'Carbonate_conc': [self.carbonate_conc],
                                      'Calcium_frac_conc': [self.calcium_frac_conc],
                                      'Carbonate_frac_conc': [self.carbonate_frac_conc],
                                      })
        
        if not hasattr(self, 'target_values'):
            self.target_values = target_values
        else:
            self.target_values = pd.concat([self.target_values, target_values], ignore_index=True)

    def save_csv(self, target_path):
        
        self.target_values.to_csv(target_path, index=False)

    
if __name__ == '__main__':

    for j in range(1, 302):
        if os.path.exists(f'./src/TargetCalculation/output/sample_{j}/sample_{j}_time_10000.0.csv'):
            if not os.path.exists(f'./src/TargetCalculation/output/sample_{j}/target_values.csv'):
            
                tva = TargetValueAnalysis()
                
                target_csv_path = f'./src/TargetCalculation/output/sample_{j}/target_values.csv'
                inout_path = f'/mnt/d/WWY/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_{j}/sample_{j}-mas.dat'
                
                for i in range(0, 101):
    
                    file_path = f'./src/TargetCalculation/output/sample_{j}/sample_{j}_time_{i*100:.1f}.csv'
        
                    check = tva.read_path(file_path)
                    if check == 0:
                        continue
                    tva.calculate_aqueous()
                    tva.calculate_adsorbed()
                    tva.calculate_mineral()
                    tva.calculate_aq_speciation()
                    tva.calculate_ad_speciation()
                    tva.calculate_components()
                    tva.save_target_values()
                
                tva.calculate_inout(inout_path)
                tva.save_csv(target_csv_path)
    
    num = 0    
    target_df = pd.DataFrame()
    input_csv_path = f'./src/Sampling/output/lhs_sampled_data.csv'
    input_csv = pd.read_csv(input_csv_path, names=['x1', 'x2', 'x3', 'x4', 'x5'], header=None)

    for j in range(1, 302):
        
        target_csv_path = f'./src/TargetCalculation/output/sample_{j}/target_values.csv'
        
        if os.path.exists(target_csv_path):
            target_data = pd.read_csv(target_csv_path)
            
            required_columns = ['Aqueous UO2++ in Granite', 'Aqueous UO2++ in Bentonite', 
                                'Aqueous UO2++ in Source', 'Adsorbed UO2++ in Bentonite', 
                                'Mineralized UO2++ in Source', 'Effluxed UO2++']
            if all(col in target_data.columns for col in required_columns):
                input_row = input_csv.iloc[j - 1]  # Get the j-th row from the input CSV
                last_values = target_data[required_columns].iloc[-1]
                
                last_values = pd.concat([input_row, last_values], axis=0)
                last_values = last_values.to_frame().T  # Transpose to make it a single row DataFrame
                last_values.columns = ['x1', 'x2', 'x3', 'x4', 'x5'] + required_columns

                target_df = pd.concat([target_df, last_values], ignore_index=True)

    
    # Filter rows where the sum of columns from the 6th to the last column is within the threshold
    threshold = 241.500084
    tolerance = 1e-5
    target_df = target_df[target_df.iloc[:, 5:].sum(axis=1).sub(threshold).abs() <= tolerance]
    
    output_csv_path = './src/TargetCalculation/output/inout.csv'
    target_df.to_csv(output_csv_path, index=False)

