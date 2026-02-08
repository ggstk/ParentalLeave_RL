import pickle
import numpy as np
import os
import pandas as pd
# from MARL import Employee,ParentalLeave,NashQLearning
from MARL_early_stopping import Employee,ParentalLeave,NashQLearning

Q_tables_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/Q_tables"

# Load Q_tables
def Load_Q_values(experiment_num):
    q0_path = os.path.join(Q_tables_dir, f"Q0_exp{experiment_num}.pickle")
    q1_path = os.path.join(Q_tables_dir, f"Q1_exp{experiment_num}.pickle")

    with open(q0_path,'rb') as file:
        Q0 = pickle.load(file)

    with open(q1_path, 'rb') as f:
        Q1 = pickle.load(f)

    df0=pd.DataFrame(list(Q0.items()),columns=["S_A","Q0"])
    df1=pd.DataFrame(list(Q1.items()),columns=["S_A","Q1"])

    merged_df = pd.merge(df0, df1,on="S_A",how ="outer")

    # Split the S_A column into two columns: joint_state and joint_action
    merged_df[['joint_state', 'joint_action']] = pd.DataFrame(merged_df['S_A'].tolist(), index=merged_df.index)

    # Drop the original S_A column
    merged_df = merged_df[['joint_state', 'joint_action', 'Q0', 'Q1']]

    merged_df[['state0', "state1"]] = pd.DataFrame(merged_df['joint_state'].tolist(), index=merged_df.index)
    merged_df = merged_df[['state0', "state1", 'joint_action', 'Q0', 'Q1']]
    condition = ~((merged_df['state0'] == (0, 0, 0, 0, 25)) | (merged_df['state1'] == (0, 0, 0, 0, 25)))
    merged_df = merged_df[condition]

    # Save to Excel file
    q_values_path=os.path.join(Q_tables_dir,f"q_values_exp{experiment_num}.xlsx")
    merged_df.to_excel(q_values_path, index=False)


Load_Q_values(0)
#Save Q_values to excel
# for experiment_num in range(0,78):
    # Load_Q_values(experiment_num)










