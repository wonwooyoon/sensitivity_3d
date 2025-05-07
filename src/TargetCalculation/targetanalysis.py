import pandas as pd
import glob
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf')
plt.rcParams['font.family'] = 'Arial'

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
        self.aqueous_granite = (material_1['Total UO2++ [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        
        material_2 = self.data[self.data['Material ID'] == 2]
        self.aqueous_bentonite = (material_2['Total UO2++ [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()

        material_3 = self.data[self.data['Material ID'] == 3]
        self.aqueous_source = (material_3['Total UO2++ [M]'] * material_3['Volume [m^3]'] * material_3['Porosity']).sum()
    
    def calculate_adsorbed(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        self.adsorbed = (material_2['Total Sorbed UO2++ [mol_m^3]'] * material_2['Volume [m^3]']).sum()

    def calculate_mineral(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        material_3 = self.data[self.data['Material ID'] == 3]
        
        self.mineral_bent = (material_2['UO2:2H2O(am) VF [m^3 mnrl_m^3 bulk]'] * material_2['Volume [m^3]'] * 38884.93558).sum()
        self.mineral_sour = (material_3['UO2:2H2O(am) VF [m^3 mnrl_m^3 bulk]'] * material_3['Volume [m^3]'] * 38884.93558).sum()

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
    
    def calculate_efflux_aux(self, efflux_path):
        with open(efflux_path, 'r') as f:
            lines = f.readlines()
            data = lines[0].split(',')
            efflux_total_uo2_index = []
            efflux_qlx_index = []
            for i, item in enumerate(data):
                if 'Total UO2++' in item:
                    efflux_total_uo2_index.append(i)
                elif 'qlx' in item:
                    efflux_qlx_index.append(i)
            result = [0]
            for line in lines[1:]:
                line_data = line.split()
                efflux_total_uo2 = [float(line_data[i]) for i in efflux_total_uo2_index]
                efflux_qlx = [float(line_data[i]) for i in efflux_qlx_index]
                efflux = [5e-3 * efflux_total_uo2[i] * efflux_qlx[i] for i in range(len(efflux_total_uo2))]
                efflux_sum = sum(efflux)            
                result.append(efflux_sum+result[-1])
            return result 

    def calculate_efflux(self, efflux_path, efflux_csv_path):
        
        self.efflux_seq = None

        for file in glob.glob(efflux_path + '*.pft'):
            efflux_single_file = self.calculate_efflux_aux(file)

            if self.efflux_seq is None:
                self.efflux_seq = efflux_single_file
            else: 
                self.efflux_seq = [a + b for a, b in zip(self.efflux_seq, efflux_single_file)]

        self.efflux = self.efflux_seq[-1]                
        self.efflux_seq_df = pd.DataFrame({'Efflux': self.efflux_seq})
        self.efflux_seq_df.to_csv(efflux_csv_path, index=False)
            
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

    for j in range(1, 301):
        if os.path.exists(f'./src/TargetCalculation/output/sample_{j}/sample_{j}_time_2400.0.csv'):
            if not os.path.exists(f'./src/TargetCalculation/output/sample_{j}/target_values.csv'):
            
                tva = TargetValueAnalysis()
                
                target_csv_path = f'./src/TargetCalculation/output/sample_{j}/target_values.csv'
                #efflux_path = f'/mnt/d/WWY/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_{j}/sample_{j}-obs-'
                #efflux_csv_path = f'./src/TargetCalculation/output/sample_{j}/efflux.csv'

                #tva.calculate_efflux(efflux_path, efflux_csv_path)
                #print(f'Efflux calculated for sample {j}')
        
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
    
                tva.save_csv(target_csv_path)
    
    num = 0    
    target_df = pd.DataFrame()
    efflux_df = pd.DataFrame()
    input_csv_path = f'./src/Sampling/output/lhs_sampled_data.csv'
    input_csv = pd.read_csv(input_csv_path, names=['x1', 'x2', 'x3', 'x4', 'x5'], header=None)

    # plot x1, x2, and x3

    for k in range(3):

        merged_df = pd.DataFrame()
        
        for j in range(301):
            
            target_csv_path = f'./src/TargetCalculation/output/sample_{j+1}/target_values.csv'

            if os.path.exists(target_csv_path):
                
                df = pd.read_csv(target_csv_path)

                if k == 0:
                    df_last_row = df.iloc[-1, :2].to_frame().T
                    df_last_row.index = [j]
                    input_row = input_csv.iloc[j, :].to_frame().T
                    input_row.index = [j]
                    df_last_row = pd.concat([input_row, df_last_row], axis=1)
                    target_df = pd.concat([target_df, df_last_row], axis=0)

                # Merge the k-th column into the merged_df
                merged_column = df.iloc[:, k].to_frame()
                merged_column.columns = [f'Sample_{j+1}_Col_{k}']
                merged_df = pd.concat([merged_df, merged_column], axis=1)

        # Save the merged dataframe to a CSV file
        merged_df.to_csv(f'./src/TargetCalculation/output/merged_target_values_{k+1}.csv', index=False)

    # plot x4
    
    # merged_df = pd.DataFrame()

    # for j in range(301):

    #     efflux_csv_path = f'/home/geofluids/research/sensitivity_3d/src/TargetCalculation/output/sample_{j+1}/efflux.csv'

    #     if os.path.exists(efflux_csv_path):
            
    #         df = pd.read_csv(efflux_csv_path)
    #         df_last_row = df.iloc[-1, :].to_frame().T
    #         df_last_row.index = [j]
    #         efflux_df = pd.concat([efflux_df, df_last_row], axis=0)
    #         num += 1

    #         merged_column = df.iloc[:, 0].to_frame()
    #         merged_column.columns = [f'Sample_{j+1}_Col_3']
    #         merged_df = pd.concat([merged_df, merged_column], axis=1)

    # target_df = pd.concat([target_df, efflux_df], axis=1)

    # print(f'Case loaded: {num}')

    # merged_df.to_csv(f'./src/TargetCalculation/output/merged_target_values_4.csv', index=False)

    # # make input-output pairs
    # target_df.to_csv('./src/TargetCalculation/output/inout.csv', index=False)

