import os,pickle
from MARL import Employee,ParentalLeave,NashQLearning
import random
experiment_num=56
base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")
# 从文件中加载 Q0,Q1
with open(q0_path,'rb') as file:
    Q0_loaded = pickle.load(file)

with open(q1_path, 'rb') as f:
    Q1_loaded = pickle.load(f)

employee0 = Employee((2, 2, 1, 0, 0)) #首先职位是4是没有升职的因素影响的
employee1 = Employee((2, 2, 1, 0, 0))#(w_yrs,pos,opt,age,time)

env = ParentalLeave(employees=[employee0, employee1], c1=7500, c2=7500,
                        alpha=0.1, delta=0.2)

#nash_q_agent = NashQLearning(experiment_num=3, env=env, discount_factor=0.99, learning_rate=1, max_iter=5 * 10 ** 6)
nash_q_agent = NashQLearning(experiment_num=experiment_num,env=env, discount_factor=0.99,learning_rate=1,max_iter = 5*10**6,
                             q_table0=Q0_loaded,q_table1=Q1_loaded)

nash_q_agent.collect_final_nash_trajectory()
#nash_q_agent.reset_states_policy()