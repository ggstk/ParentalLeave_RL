import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import os,pickle,json
# from MARL import Employee,ParentalLeave,NashQLearning
from MARL_early_stopping import Employee,ParentalLeave,NashQLearning
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
Q_tables_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/Q_tables/"
results_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/results/"


employee0 = Employee((0, 0, 1, 0, 0)) 
employee1 = Employee((0, 0, 1, 0, 0))
n=100

def run_experiment(experiment_num):
    print("experiment_num:",experiment_num,"n:",n)
    q0_path = os.path.join(Q_tables_dir, f"Q0_exp{experiment_num}.pickle")
    q1_path = os.path.join(Q_tables_dir, f"Q1_exp{experiment_num}.pickle")

    with open(q0_path, 'rb') as file:
        Q0 = pickle.load(file)

    with open(q1_path, 'rb') as f:
        Q1 = pickle.load(f)

    env = ParentalLeave(employees=[employee0, employee1])

    nash_q_agent = NashQLearning(
        experiment_num=experiment_num,
        env=env,
        discount_factor=0.99,
        learning_rate=1,
        max_iter=5 * 10 ** 6,
        q_table0=Q0,
        q_table1=Q1
    )
    start_time = datetime.now()
    prob1, prob2, error1, error2 = nash_q_agent.calculate_each_prob(n)
    end_time = datetime.now()
    print(f"Experiment {experiment_num} finished! Running time: {end_time - start_time}")
    return experiment_num, prob1, prob2, error1, error2 


#Calculate the probability that both employees end in a target terminal state(m=0） under Nash Equilibrium
#(w_yrs,pos,opt,age,time)
def calculate_both_agent_probability():
    experiment_range = range(0,78)
    n=100
    probs = [None] * len(experiment_range)
    errors = [None] * len(experiment_range)

    
    with ProcessPoolExecutor() as executor:
        results = executor.map(run_experiment, experiment_range)  # 병렬 실행

    for experiment_num, prob, margin_of_error in results:
        probs[experiment_num] = prob
        errors[experiment_num] = margin_of_error

    print("probs",probs)
    print("errors",errors)


    probs_data_path = os.path.join(results_dir, f"Probs_both.json")
    probs_data_path1 = os.path.join(results_dir, f"Errors_both.json")

    with open(probs_data_path, 'w') as f:
        json.dump(probs, f)

    with open(probs_data_path1, 'w') as f:
        json.dump(errors, f)


#Calculate the probability that just one of the employees ends in a target terminal state(m=0） under Nash Equilibrium
#(w_yrs,pos,opt,age,time)
def calculate_one_agent_probability():
    experiment_range = range(0,78)
    n=100
    probs = [None] * len(experiment_range)
    errors = [None] * len(experiment_range)


    with ProcessPoolExecutor() as executor:
        results = executor.map(run_experiment, experiment_range)  # 병렬 실행

    for experiment_num, prob, margin_of_error in results:
        probs[experiment_num] = prob
        errors[experiment_num] = margin_of_error

    print("probs",probs)
    print("errors",errors)

    probs_data_path = os.path.join(results_dir, f"Probs_one.json")
    probs_data_path1 = os.path.join(results_dir, f"Errors_one.json")

    # probs_data_path = os.path.join(results_dir, f"Probs_one.json")
    # probs_data_path1 = os.path.join(results_dir, f"Errors_one.json")

    with open(probs_data_path, 'w') as f:
        json.dump(probs, f)

    with open(probs_data_path1, 'w') as f:
        json.dump(errors, f)




#calculate each agent's probability that each employee ends in a target terminal state(m=0） under Nash Equilibrium


def calculate_each_agent_probability():
    experiment_range = range(0,42) #(42,78)
    probs1 = [None] * len(experiment_range)
    probs2 = [None] * len(experiment_range)
    errors1 = [None] * len(experiment_range)
    errors2 = [None] * len(experiment_range)

    with ProcessPoolExecutor() as executor:
        results = executor.map(run_experiment, experiment_range)  

    for experiment_num, prob1, prob2, error1, error2 in results:
        probs1[experiment_num] = prob1
        probs2[experiment_num] = prob2
        errors1[experiment_num] = error1
        errors2[experiment_num] = error2

    probs_data_path1 = os.path.join(results_dir, f"Probs1(0-42).json")
    probs_data_path2 = os.path.join(results_dir, f"Probs2(0-42).json")
    probs_data_path3 = os.path.join(results_dir, f"Errors1(0-42).json")
    probs_data_path4 = os.path.join(results_dir, f"Errors2(0-42).json")
    with open(probs_data_path1, 'w') as f:
        json.dump(probs1, f)

    with open(probs_data_path2, 'w') as f:
        json.dump(probs2, f)

    with open(probs_data_path3, 'w') as f:
        json.dump(errors1, f)

    with open(probs_data_path4, 'w') as f:
        json.dump(errors2, f)




#calculate_both_agent_probability  -->probs_data_path = os.path.join(results_dir, f"Probs_both.json")
                                #  --> probs_data_path1 = os.path.join(results_dir, f"Errors_both.json")
#calculate_one_agent_probability  -->probs_data_path = os.path.join(results_dir, f"Probs_one.json")
                                # -->probs_data_path1 = os.path.join(results_dir, f"Errors_one.json")
#calculate_each_agent_probability -->probs_data_path1 = os.path.join(results_dir, f"Probs1.json")
                                    # probs_data_path2 = os.path.join(results_dir, f"Probs2.json")
                                    # probs_data_path3 = os.path.join(results_dir, f"Errors1.json")
                                    # probs_data_path4 = os.path.join(results_dir, f"Errors2.json")

calculate_each_agent_probability()  # time：4:31:48.832723