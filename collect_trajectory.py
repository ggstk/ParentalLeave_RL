import os,pickle
import random
# from MARL import Employee,ParentalLeave,NashQLearning
from MARL_early_stopping import Employee,ParentalLeave,NashQLearning


experiment_num=56
Q_tables_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/Q_tables/"
q0_path = os.path.join(Q_tables_dir, f"Q0_exp{experiment_num}.pickle")
q1_path = os.path.join(Q_tables_dir, f"Q1_exp{experiment_num}.pickle")

with open(q0_path,'rb') as file:
    Q0_loaded = pickle.load(file)

with open(q1_path, 'rb') as f:
    Q1_loaded = pickle.load(f)

employee0 = Employee((2, 2, 1, 0, 0)) 
employee1 = Employee((2, 2, 1, 0, 0))

env = ParentalLeave(employees=[employee0, employee1], c1=7500, c2=7500,
                        alpha=0.1, delta=0.2)

nash_q_agent = NashQLearning(experiment_num=experiment_num,env=env, discount_factor=0.99,learning_rate=1,max_iter = 5*10**6,
                             q_table0=Q0_loaded,q_table1=Q1_loaded)

trajectory=nash_q_agent.collect_final_nash_trajectory()

print(trajectory)
