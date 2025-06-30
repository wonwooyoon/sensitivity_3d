import numpy as np
import pandas as pd

for i in range(0, 6): # output
    
    shap_dict = {}

    for j in range(0, 5): # input

        x = pd.read_csv(f'./src/PINN/output/alpha_1.0_beta_0.001_optuna/shap_values_output{i}.csv').iloc[:, j].values
        x_color = pd.read_csv('./src/TargetCalculation/output/normalized_inout.csv').iloc[:, j].values

        num_bins = 50           # bin 개수
        min_range = 0.0         # y의 최소 반폭 (데이터 적을 때)
        max_range = 1.0         # y의 최대 반폭 (데이터 많을 때)

        counts, bin_edges = np.histogram(x, bins=num_bins)
        bin_indices = np.digitize(x, bins=bin_edges) - 1

        y_values = np.zeros_like(x)
        max_count = counts.max()
        min_count = counts.min()

        for k in range(num_bins):
            in_bin = bin_indices == k
            count = np.sum(in_bin)
            if count == 0:
                continue

            # y 범위 설정 (밀도가 높을수록 넓은 범위)
            y_half_range = min_range + (counts[k] - min_count) / (max_count - min_count) * (max_range - min_range)
            
            # count개의 값을 -y_half_range ~ y_half_range 사이에 균등하게 배치
            if count == 1:
                y_bin = np.array([0.0])
            else:
                y_bin = np.linspace(-y_half_range, y_half_range, count)

            # 정렬된 순서대로 y값을 할당하면 일정한 분포를 그림
            y_values[in_bin] = y_bin

        shap_dict[f'x{j}'] = x
        shap_dict[f'y{j}'] = y_values
        shap_dict[f'x{j}_color'] = x_color

    shap_values = pd.DataFrame(shap_dict)
    shap_values.to_csv(f'./src/PINN/output/alpha_1.0_beta_0.001_optuna/shap_values_output{i}_all.csv', index=False)
        