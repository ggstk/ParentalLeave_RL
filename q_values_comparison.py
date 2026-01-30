import pickle
import numpy as np
import os
from MARL import Employee,ParentalLeave,NashQLearning
# 定义存储路径
experiment_num=80
base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models"
q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")
# q0_path = "C:/Users/SDOlab/Desktop/                       ParentalLeave/saved_models/Q0"
# q1_path = "C:/Users/SDOlab/Desktop/ParentalLeave/saved_models/Q1"
# 从文件中加载 Q0

employee0 = Employee((0,0,1,0,0)) #首先职位是4是没有升职的因素影响的
employee1 = Employee((0,0,1,0,0))#(w_yrs,pos,opt,age,time) (7, 3, 1, 8, 8)
alpha,delta=0,0 
c1,c2 = 3300,10000
env = ParentalLeave(employees=[employee0, employee1], c1=c1, c2=c2, alpha=alpha, delta=delta)

target_states=env.joint_states


with open(q0_path,'rb') as file:
    Q0_loaded = pickle.load(file)

# 从文件中加载 Q1
with open(q1_path, 'rb') as f:
    Q1_loaded = pickle.load(f)

#打印 Q0 和 Q1
print("Q0 loaded from file:")
#print(Q0_loaded)

print("\nQ1 loaded from file:")
# print(Q1_loaded)

import pandas as pd

data0 = Q0_loaded
data1 = Q1_loaded

data2 = {key: value for key, value in Q0_loaded.items() if key[0] in target_states}
data3 = {key: value for key, value in Q1_loaded.items() if key[0] in target_states}

df0=pd.DataFrame(list(data0.items()),columns=["S_A","Q0"])
df1=pd.DataFrame(list(data1.items()),columns=["S_A","Q1"])

merged_df = pd.merge(df0, df1,on="S_A",how ="outer")

# Split the S_A column into two columns: joint_state and joint_action
merged_df[['joint_state', 'joint_action']] = pd.DataFrame(merged_df['S_A'].tolist(), index=merged_df.index)

# Drop the original S_A column
merged_df = merged_df[['joint_state', 'joint_action', 'Q0', 'Q1']]

merged_df[['state0', "state1"]] = pd.DataFrame(merged_df['joint_state'].tolist(), index=merged_df.index)
merged_df = merged_df[['state0', "state1", 'joint_action', 'Q0', 'Q1']]
# 检查 state0 和 state1 是否为 (0, 0, 0, 0, 25)  ~是取反的意思
condition = ~((merged_df['state0'] == (0, 0, 0, 0, 25)) | (merged_df['state1'] == (0, 0, 0, 0, 25)))

# 使用条件过滤数据框
merged_df = merged_df[condition]
# Save to Excel file
#q_values_comparison_path=os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/q_values_comparison/",f"target_states_exp{experiment_num}.xlsx")
q_values_comparison_path=os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/q_values_comparison/",f"q_values_comparison_exp{experiment_num}.xlsx")
merged_df.to_excel(q_values_comparison_path, index=False)


