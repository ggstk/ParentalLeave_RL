import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import os,pickle,json
from MARL import Employee,ParentalLeave,NashQLearning
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
employee0 = Employee((0, 0, 1, 0, 0)) #首先职位是4是没有升职的因素影响的
employee1 = Employee((0, 0, 1, 0, 0))#(w_yrs,pos,opt,age,time)

# 병렬 실행
experiment_range = range(0,42)
# probs1,probs2 =[],[]
# errors1,errors2 = [],[]
n=100
# #결과를 저장할 리스트 (0~29까지의 인덱스를 그대로 사용)
probs = [None] * len(experiment_range)
errors = [None] * len(experiment_range)

def run_experiment(experiment_num):
    """각 실험을 실행하고 결과를 반환"""
    print("experiment_num:",experiment_num,"n:",n)
    q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
    q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")

    with open(q0_path, 'rb') as file:
        Q0 = pickle.load(file)

    with open(q1_path, 'rb') as f:
        Q1 = pickle.load(f)

    env = ParentalLeave(employees=[employee0, employee1])

    nash_q_agent = NashQLearning(
        experiment_num=experiment_num,
        env=env,
        discount_factor=0.99,
        learning_rate=0.5,
        max_iter=5 * 10 ** 6,
        q_table0=Q0,
        q_table1=Q1
    )
    start_time = datetime.now()
    prob, margin_of_error = nash_q_agent.calculate_prob(n)  #both_leave
    # prob, margin_of_error = nash_q_agent.calculate_one_leave_prob(n)
    # print("prob, margin_of_error",prob, margin_of_error)

    end_time = datetime.now()
    print(f"Experiment {experiment_num} finished! Running time: {end_time - start_time}")
    return experiment_num, prob, margin_of_error  # 실험 번호와 결과 반환



with ProcessPoolExecutor() as executor:
    results = executor.map(run_experiment, experiment_range)  # 병렬 실행

# # 결과를 순서대로 저장
for experiment_num, prob, margin_of_error in results:
    probs[experiment_num] = prob
    errors[experiment_num] = margin_of_error

print("probs",probs)
print("errors",errors)

base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
probs_data_path = os.path.join(base_dir, f"Probs_both(exp0-42,n=100).json")
probs_data_path1 = os.path.join(base_dir, f"Errors_both(exp0-42,n=100).json")

with open(probs_data_path, 'w') as f:
    json.dump(probs, f)

with open(probs_data_path1, 'w') as f:
    json.dump(errors, f)




# 결과를 저장할 리스트 (0~29까지의 인덱스를 그대로 사용)
# probs1 = [None] * 30
# probs2 = [None] * 30
# errors1 = [None] * 30
# errors2 = [None] * 30
# n=100
# def run_experiment(experiment_num):
#     """각 실험을 실행하고 결과를 반환"""
#     print("experiment_num:",experiment_num,"n:",n)
#     q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
#     q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")

#     with open(q0_path, 'rb') as file:
#         Q0 = pickle.load(file)

#     with open(q1_path, 'rb') as f:
#         Q1 = pickle.load(f)

#     env = ParentalLeave(employees=[employee0, employee1])

#     nash_q_agent = NashQLearning(
#         experiment_num=experiment_num,
#         env=env,
#         discount_factor=0.99,
#         learning_rate=1,
#         max_iter=5 * 10 ** 6,
#         q_table0=Q0,
#         q_table1=Q1
#     )
#     start_time = datetime.now()
#     prob1, prob2, error1, error2 = nash_q_agent.calculate_each_prob(n)
#     end_time = datetime.now()
#     print(f"Experiment {experiment_num} finished! Running time: {end_time - start_time}")
#     return experiment_num, prob1, prob2, error1, error2  # 실험 번호와 결과 반환

# # 병렬 실행
# experiment_range = range(0, 30)  # 0부터 29까지
# with ProcessPoolExecutor() as executor:
#     results = executor.map(run_experiment, experiment_range)  # 병렬 실행

# # 결과를 순서대로 저장
# for experiment_num, prob1, prob2, error1, error2 in results:
#     probs1[experiment_num] = prob1
#     probs2[experiment_num] = prob2
#     errors1[experiment_num] = error1
#     errors2[experiment_num] = error2

# base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
# probs_data_path1 = os.path.join(base_dir, f"Probs1(c1=c2,n=100).json")
# probs_data_path2 = os.path.join(base_dir, f"Probs2(c1=c2,n=100).json")
# probs_data_path3 = os.path.join(base_dir, f"Errors1(c1=c2,n=100).json")
# probs_data_path4 = os.path.join(base_dir, f"Errors2(c1=c2,n=100).json")
# with open(probs_data_path1, 'w') as f:
#     json.dump(probs1, f)

# with open(probs_data_path2, 'w') as f:
#     json.dump(probs2, f)


# with open(probs_data_path3, 'w') as f:
#     json.dump(errors1, f)

# with open(probs_data_path4, 'w') as f:
#     json.dump(errors2, f)




# probs1 = [None] * 12
# probs2 = [None] * 12
# errors1 = [None] * 12
# errors2 = [None] * 12
# n=100
# def run_experiment(experiment_num):
#     """각 실험을 실행하고 결과를 반환"""
#     print("experiment_num:",experiment_num,"n:",n)
#     q0_path = os.path.join(base_dir, f"Q0_exp{experiment_num}.pickle")
#     q1_path = os.path.join(base_dir, f"Q1_exp{experiment_num}.pickle")

#     with open(q0_path, 'rb') as file:
#         Q0 = pickle.load(file)

#     with open(q1_path, 'rb') as f:
#         Q1 = pickle.load(f)

#     env = ParentalLeave(employees=[employee0, employee1])

#     nash_q_agent = NashQLearning(
#         experiment_num=experiment_num,
#         env=env,
#         discount_factor=0.99,
#         learning_rate=1,
#         max_iter=5 * 10 ** 6,
#         q_table0=Q0,
#         q_table1=Q1
#     )
#     start_time = datetime.now()
#     prob1, prob2, error1, error2 = nash_q_agent.calculate_each_prob(n)
#     end_time = datetime.now()
#     print(f"Experiment {experiment_num} finished! Running time: {end_time - start_time}")
#     return experiment_num, prob1, prob2, error1, error2  # 실험 번호와 결과 반환

# # 병렬 실행
# experiment_range = range(66, 78)  # 0부터 29까지
# with ProcessPoolExecutor() as executor:
#     results = executor.map(run_experiment, experiment_range)  # 병렬 실행

# # 결과를 순서대로 저장
# for experiment_num, prob1, prob2, error1, error2 in results:
#     probs1[experiment_num-66] = prob1
#     probs2[experiment_num-66] = prob2
#     errors1[experiment_num-66] = error1
#     errors2[experiment_num-66] = error2

# base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
# probs_data_path1 = os.path.join(base_dir, f"Probs1(exp66-77,n=100).json")
# probs_data_path2 = os.path.join(base_dir, f"Probs2(exp66-77,n=100).json")
# probs_data_path3 = os.path.join(base_dir, f"Errors1(exp66-77,n=100).json")
# probs_data_path4 = os.path.join(base_dir, f"Errors2(exp66-77,n=100).json")
# with open(probs_data_path1, 'w') as f:
#     json.dump(probs1, f)

# with open(probs_data_path2, 'w') as f:
#     json.dump(probs2, f)


# with open(probs_data_path3, 'w') as f:
#     json.dump(errors1, f)

# with open(probs_data_path4, 'w') as f:
#     json.dump(errors2, f)