import pandas as pd
import numpy as np
import nashpy as nash
import os,pickle
# from MARL import Employee,ParentalLeave,NashQLearning
from MARL_early_stopping import Employee,ParentalLeave,NashQLearning


experiment_num=0
Q_tables_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/Q_tables/"
file_path = os.path.join(Q_tables_dir, f"q_values_exp{experiment_num}.xlsx")
nash_equilibrium_dir="/home/zhaolixue/ZHAOLIXUE/ParentalLeave/nash_equilibrium"

merged_df = pd.read_excel(file_path)

q0_path = os.path.join(Q_tables_dir, f"Q0_exp{experiment_num}.pickle")
q1_path = os.path.join(Q_tables_dir, f"Q1_exp{experiment_num}.pickle")

with open(q0_path,'rb') as file:
    Q0= pickle.load(file)
with open(q1_path, 'rb') as f:
    Q1= pickle.load(f)

employee0 = Employee((7, 3, 1, 8, 8)) 
employee1 = Employee((7, 3, 1, 8, 8))
env = ParentalLeave(employees=[employee0, employee1])
    
# Create an instance of the NashQLearning agent          
nash_q_agent = NashQLearning(experiment_num=experiment_num,env=env, discount_factor=0.95,learning_rate=0.5,q_table0=Q0,q_table1=Q1)                         

nash_equilibria = []
counter = 0
print(len(merged_df.groupby(["state0", "state1"])))

for joint_state, group in merged_df.groupby(["state0", "state1"]):
   
    joint_state = tuple(eval(x) for x in joint_state)
    q0_values = group['Q0'].tolist()
    q1_values = group['Q1'].tolist()

    if len(q0_values) < 4:
        counter += 1

    elif joint_state[0] != (0, 0, 0, 0, 25) and joint_state[1] != (0, 0, 0, 0, 25):
        # if len(q0_values) == 4:
        q_values0 = {action: nash_q_agent._get_q_value(nash_q_agent.env.current_states, action, nash_q_agent.q_table0) for action in
                        nash_q_agent.joint_actions}
        q_values1 = {action: nash_q_agent._get_q_value(nash_q_agent.env.current_states, action, nash_q_agent.q_table1) for action in
                        nash_q_agent.joint_actions}
        q_values0_matrix = np.array(list(q_values0.values())).reshape(2, 2)
        q_values1_matrix = np.array(list(q_values1.values())).reshape(2, 2)

        greedy_game = nash.Game(q_values0_matrix, q_values1_matrix)
        # equilibriums = list(greedy_game.support_enumeration())
        equilibriums = list(greedy_game.support_enumeration())

        # equilibriums = list(greedy_game.lemke_howson(initial_dropped_label=0))
        # print("nash_eq",equilibriums)
        temp = []
        for equilibrium in equilibriums:
            # print(equilibrium)
            if len(np.where(equilibrium[0] == 1)[0]) != 0:
                action0 = np.where(equilibrium[0] == 1)[0][0]
            if len(np.where(equilibrium[1] == 1)[0]) != 0:
                action1 = np.where(equilibrium[1] == 1)[0][0]
            actions = (int(action0), int(action1))
            temp.append(actions)
        temp = set(temp)
        temp = list(temp)
        nash_equilibria.append((joint_state[0], joint_state[1], temp))

    elif (joint_state[0][2] == 0 or joint_state[0][3] == 9) and (joint_state[1][2] == 0 or joint_state[1][3] == 9):
        nash_equilibria.append((joint_state[0], joint_state[1], [(0,0)]))

    elif (joint_state[0][2] == 0 or joint_state[0][3] == 9):
        if (joint_state, (0, 0)) in Q1 and (joint_state, (0, 1)) in Q1:
            action0 = 0
            action1_0 = Q1[(joint_state,(0,0))]
            action1_1 = Q1[(joint_state, (0,1))]
            if action1_0>action1_1:
                action1=0
            else:
                action1=1
            nash_equilibria.append((joint_state[0], joint_state[1], [(action0,action1)]))

    elif (joint_state[1][2] == 0 or joint_state[1][3] == 9):
        if (joint_state, (0, 0)) in Q1 and (joint_state, (1, 0)) in Q1:
            action1 = 0
            action0_1 = Q0[(joint_state, (0, 0))]
            action1_1 = Q0[(joint_state, (1, 0))]
            if action0_1 > action1_1:
                action0 = 0
            else:
                action0 = 1
            nash_equilibria.append((joint_state[0], joint_state[1],[(action0, action1)]))

print("counter", counter)

df_nash = pd.DataFrame(nash_equilibria, columns=["state0", "state1", "nash_equilibrium"])
condition = ~((df_nash['state0'].apply(lambda x: x[2]) == 0) & (df_nash['state1'].apply(lambda x: x[2]) == 0)) & \
            ~((df_nash['state0'].apply(lambda x: x[3]) == 9) & (df_nash['state1'].apply(lambda x: x[3]) == 9))


#df_nash = df_nash[condition]
nasheq_path = os.path.join(nash_equilibrium_dir,
                           f"nash_actions_exp{experiment_num}.xlsx")
df_nash.to_excel(nasheq_path, index=False)

# states which the state of two employees are same
# df_nash_filtered = df_nash[df_nash['state0'] == df_nash['state1']]
#df_nash_filtered = df_nash[df_nash['state0'][1] == df_nash['state1'][1]]
# df_nash_filtered = df_nash[df_nash['state0'].apply(lambda x: x[1]) == df_nash['state1'].apply(lambda x: x[1])]
df_nash_filtered = df_nash[
    # (df_nash['state0'].apply(lambda x: x[1]) == df_nash['state1'].apply(lambda x: x[1])) & 
    (df_nash['state0'].apply(lambda x: x[2]) == 1) &  
    (df_nash['state1'].apply(lambda x: x[2]) == 1) 
    # (df_nash['state0'].apply(lambda x: x[3]) == 8)&
    # (df_nash['state1'].apply(lambda x: x[3]) == 8)
]

nasheq_filtered_path = os.path.join(nash_equilibrium_dir,
                                    f"nash_filtered_actions_exp{experiment_num}.xlsx")

df_nash_filtered.to_excel(nasheq_filtered_path, index=False)










