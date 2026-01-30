import os,pickle,json
from MARL import Employee,ParentalLeave,NashQLearning
import matplotlib.pyplot as plt
import seaborn as sns
base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
employee0 = Employee((0, 0, 1, 0, 0)) #首先职位是4是没有升职的因素影响的
employee1 = Employee((0, 0, 1, 0, 0))#(w_yrs,pos,opt,age,time)

probs=[]
errors=[]
probs1,probs2 =[],[]
errors1,errors2 = [],[]
n=100

# experiment_range = range(0,30)
# # experiment_range = [1,33,34]
# # agent1和agent2一起
# for experiment_num in experiment_range:
#     print(experiment_num)
#     q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
#     q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")
    
#     with open(q0_path, 'rb') as file:
#         Q0= pickle.load(file)

#     with open(q1_path, 'rb') as f:
#         Q1 = pickle.load(f)

#     env = ParentalLeave(employees=[employee0, employee1])

#     nash_q_agent = NashQLearning(experiment_num=experiment_num, env=env, discount_factor=0.99, learning_rate=0.5, max_iter=5 * 10 ** 6,
#                                 q_table0=Q0, q_table1=Q1)

#     prob,margin_of_error= nash_q_agent.calculate_prob(n)#计算两个智能体都在工作25年之后离开的概率 range（0,30）
#     probs.append(prob) #probs是实验0到实验30的结果
#     errors.append(margin_of_error)
# base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
# probs_data_path = os.path.join(base_dir, f"Probs(c1=c2,n=100).json")
# probs_data_path1 = os.path.join(base_dir, f"Errors(c1=c2,n=100).json")

# with open(probs_data_path, 'w') as f:
#     json.dump(probs, f)

# with open(probs_data_path1, 'w') as f:
#     json.dump(errors, f)

# experiment_range = range(0,42)
# for experiment_num in experiment_range:
#     print(experiment_num)
#     q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
#     q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")
    
#     with open(q0_path, 'rb') as file:
#         Q0= pickle.load(file)

#     with open(q1_path, 'rb') as f:
#         Q1 = pickle.load(f)

#     env = ParentalLeave(employees=[employee0, employee1])

#     nash_q_agent = NashQLearning(experiment_num=experiment_num, env=env, discount_factor=0.99, learning_rate=1, max_iter=5 * 10 ** 6,
#                                 q_table0=Q0, q_table1=Q1)

#     prob1,prob2,error1,error2= nash_q_agent.calculate_each_prob(n)#计算两个智能体都在工作25年之后离开的概率 range（0,30）
#     probs1.append(prob1) 
#     probs2.append(prob2)
#     errors1.append(error1)
#     errors2.append(error2)


# base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
# probs_data_path1 = os.path.join(base_dir, f"Probs1(c1=c2).json")
# probs_data_path2 = os.path.join(base_dir, f"Probs2(c1=c2).json")
# probs_data_path3 = os.path.join(base_dir, f"Errors1(c1=c2).json")
# probs_data_path4 = os.path.join(base_dir, f"Errors2(c1=c2).json")
# with open(probs_data_path1, 'w') as f:
#     json.dump(probs1, f)

# with open(probs_data_path2, 'w') as f:
#     json.dump(probs2, f)


# with open(probs_data_path3, 'w') as f:
#     json.dump(errors1, f)

# with open(probs_data_path4, 'w') as f:
#     json.dump(errors2, f)




experiment_range = range(42,78)
for experiment_num in experiment_range:
    print(experiment_num)
    q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
    q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")
    
    with open(q0_path, 'rb') as file:
        Q0= pickle.load(file)

    with open(q1_path, 'rb') as f:
        Q1 = pickle.load(f)

    env = ParentalLeave(employees=[employee0, employee1])

    nash_q_agent = NashQLearning(experiment_num=experiment_num, env=env, discount_factor=0.99, learning_rate=1, max_iter=5 * 10 ** 6,
                                q_table0=Q0, q_table1=Q1)

    prob1,prob2,error1,error2= nash_q_agent.calculate_each_prob(n)#计算两个智能体都在工作25年之后离开的概率 range（0,30）
    probs1.append(prob1) #probs是实验0到实验30的结果
    probs2.append(prob2)
    errors1.append(error1)
    errors2.append(error2)

base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
probs_data_path1 = os.path.join(base_dir, f"Probs1(unequal).json")
probs_data_path2 = os.path.join(base_dir, f"Probs2(unequal).json")
probs_data_path3 = os.path.join(base_dir, f"Errors1(unequal).json")
probs_data_path4 = os.path.join(base_dir, f"Errors2(unequal).json")
with open(probs_data_path1, 'w') as f:
    json.dump(probs1, f)

with open(probs_data_path2, 'w') as f:
    json.dump(probs2, f)


with open(probs_data_path3, 'w') as f:
    json.dump(errors1, f)

with open(probs_data_path4, 'w') as f:
    json.dump(errors2, f)











