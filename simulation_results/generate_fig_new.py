# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:23:49 2023

@author: zyimi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def load_data(file_path):
    loaded_data = np.load(file_path, allow_pickle=True)
    data = [item.item() for item in loaded_data.values()]
    return data

M = 1
N = 6
dr = 0.15
data_file_path = f'ch{M}_{N}nodes_dr{dr}'
nn_path = 'nn_' + data_file_path + '.npz'
benchmark_path = 'benchmark_' + data_file_path + '.npz'
QCSMA_path = 'QCSMA_' + data_file_path + '.npz'

def get_df(data):
    averages = {
        "ave_delay": [],
        "throughput": [],
        "Total throughput of 5 experiments": []
    }
    
    num_experiments = len(data)
    num_links = len(data[0]["ave_delay"])
    
    for i in range(num_links):
        avg_delay_link = 0
        avg_throughput_link = 0
        
        for j in range(num_experiments):
            avg_delay_link += data[j]["ave_delay"][i]
            avg_throughput_link += data[j]["throughput"][i]
        
        avg_delay_link /= num_experiments
        averages["ave_delay"].append(round(avg_delay_link, 2))
        
        avg_throughput_link /= num_experiments
        averages["throughput"].append(avg_throughput_link)
    
    total_throughput = sum(np.sum(data[j]["throughput"]) for j in range(num_experiments))
    averages["Total throughput of 5 experiments"] = total_throughput
    
    index = [f"{i+1}" for i in range(num_links)]
    df = pd.DataFrame(averages, index=index)
    print(df)
    return df

# data = load_data('nn_ch3_20nodes_dr0.3.npz')
data = load_data(nn_path)
df1 = get_df(data)
# Example usage

QCSMA_data = load_data(QCSMA_path)
df2 = get_df(QCSMA_data)

benchmark_data = load_data(benchmark_path)
df3 = get_df(benchmark_data)



if N >= 20:
    def autolabel(rects, bar_number):
        for rect in rects:
            height = rect.get_height()
            offset = rect.get_width() * 0.8  # You can adjust the offset as needed
            x_coordinate = rect.get_x() + rect.get_width() / 2 - offset
            
            if bar_number == 3 and height > ax.get_ylim()[1]:
                ax.annotate(f'{height:.1f}', xy=(x_coordinate, ax.get_ylim()[1]),
                            xytext=(10, 0), textcoords='offset points', ha='center', va='bottom',
                            arrowprops=dict(facecolor='red', arrowstyle='->'))
            else:
                ax.annotate(f'{height:.1f}', xy=(x_coordinate, height),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

else:
    def autolabel(rects, bar_number):
        for rect in rects:
            height = rect.get_height()
            if bar_number == 3 and height > ax.get_ylim()[1]:  # if it's the third bar and height is more than ylim
                ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, ax.get_ylim()[1]),
                            xytext=(10, 0), textcoords='offset points', ha='center', va='bottom',
                            arrowprops=dict(facecolor='red', arrowstyle='->'))
            else:
                ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        
        
num_links = len(data[0]["ave_delay"])
# Plotting
bar_width = 0.2  # reduced width to fit three bars
opacity = 0.8

fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(len(df1))

# Custom colors
colors = ['steelblue', 'darkgreen', 'purple']  # added a color for the QCSMA bar

# ave_delay bars
rects1 = ax.bar(index - bar_width, df1["ave_delay"], bar_width, alpha=opacity, color=colors[0], label='RL_method')
rects2 = ax.bar(index, df2["ave_delay"], bar_width, alpha=opacity, color=colors[1], label='Benchmark')
rects3 = ax.bar(index + bar_width, df3["ave_delay"], bar_width, alpha=opacity, color=colors[2], label='QCSMA')  # added a third bar

max_val_second_bar = max(df3['ave_delay'])
ax.set_ylim(0, max_val_second_bar * 1.2)


# Axis labels
ax.set_xlabel('Link number')
ax.set_ylabel('average delay time of successful packets')
ax.set_xticks(index)
ax.set_xticklabels(df1.index)
ax.legend()

autolabel(rects1, 1)
autolabel(rects2, 2)
autolabel(rects3, 3)  # added autolabel call for the third bar
plt.savefig('./figures/{}delay.png'.format(data_file_path), format='png')
plt.tight_layout()
plt.show()















