import multiprocessing
from MARL import Employee, ParentalLeave, NashQLearning
from datetime import datetime
import os
num_cpus = os.cpu_count()#128

def run_experiment(experiment_num, alpha, delta, c1, c2):
    print("Experiment_num",experiment_num)
    employee0 = Employee((7, 3, 1, 8, 8))
    employee1 = Employee((7, 3, 1, 8, 8))
    
    yrs_for_pos = [3, 4, 4, 5, 5]
    max_yrs_pos = [6, 7, 7, 8, 8]
    
    env = ParentalLeave(employees=[employee0, employee1],max_yrs_pos=max_yrs_pos, yrs_for_pos=yrs_for_pos, c1=c1, c2=c2, alpha=alpha, delta=delta)
    nash_q_agent = NashQLearning(experiment_num=experiment_num, env=env, discount_factor=0.95, learning_rate=0.5, max_iter=10**7)

    start_time = datetime.now()
    q_table_employee0, q_table_employee1 = nash_q_agent.fit()
    end_time = datetime.now() 
    print(f"Experiment {experiment_num} finished! Running time: {end_time - start_time}")
    # print(f"Experiment {experiment_num} finished!")



# hyperparameter: (experiment_num, alpha, delta, c1, c2)
param_sets = [
    #c1 = c2
    # (0, 0, 0, 0, 0),
    # (1, 0, 0, 3300, 3300),
    # (2, 0, 0, 4150, 4150),
    # (3, 0, 0, 5000, 5000),
    # (4, 0, 0, 10000, 10000),
    # (5, 0, 0, 20000, 20000),

    # (6, 0, 0.1, 0, 0),
    # (7, 0, 0.1, 3300, 3300),
    # (8, 0, 0.1, 4150, 4150),
    # (9, 0, 0.1, 5000, 5000),
    # (10, 0, 0.1, 10000, 10000),
    # (11, 0, 0.1, 20000, 20000),

    # (12, 0, 0.2, 0, 0),
    # (13, 0, 0.2, 3300, 3300),
    # (14, 0, 0.2, 4150, 4150),
    # (15, 0, 0.2, 5000, 5000),
    # (16, 0, 0.2, 10000, 10000),
    # (17, 0, 0.2, 20000, 20000),

    # (18, 0.05, 0, 0, 0),
    # (19, 0.05, 0, 3300, 3300),
    # (20, 0.05, 0, 4150, 4150),
    # (21, 0.05, 0, 5000, 5000),
    # (22, 0.05, 0, 10000, 10000),
    # (23, 0.05, 0, 20000, 20000),

    # (24, 0.1, 0, 0, 0),
    # (25, 0.1, 0, 3300, 3300),
    # (26, 0.1, 0, 4150, 4150),
    # (27, 0.1, 0, 5000, 5000),
    # (28, 0.1, 0, 10000, 10000),
    # (29, 0.1, 0, 20000, 20000),
    
    (30, 0.05, 0.1, 0, 0),
    # (31, 0.05, 0.1, 3300, 3300),
    # (32, 0.05, 0.1, 4150, 4150),
    # (33, 0.05, 0.1, 5000, 5000),
    # (34, 0.05, 0.1, 10000, 10000),
    # (35, 0.05, 0.1, 20000, 20000),
  
    
    # (36, 0.1, 0.1, 0, 0),
    # (37, 0.1, 0.1, 3300, 3300),
    # (38, 0.1, 0.1, 4150, 4150),
    # (39, 0.1, 0.1, 5000, 5000),
    (40, 0.1, 0.1, 10000, 10000),
    (41, 0.1, 0.1, 20000, 20000),
   
    # c1 < c2
    (42, 0, 0, 0, 3300),
    (43, 0, 0, 0, 5000),
    (44, 0, 0, 0, 10000),
    (45, 0, 0, 3300, 5000),
    (46, 0, 0, 3300, 10000),
    (47, 0, 0, 10000, 5000),
    
    (48, 0.05, 0, 0, 3300),
    (49, 0.05, 0, 0, 5000),
    (50, 0.05, 0, 0, 10000),
    (51, 0.05, 0, 3300, 5000),
    (52, 0.05, 0, 3300, 10000),
    (53, 0.05, 0, 10000, 5000),
    
    (54, 0.1, 0, 0, 3300),
    (55, 0.1, 0, 0, 5000),
    (56, 0.1, 0, 0, 10000),
    (57, 0.1, 0, 3300, 5000),
    (58, 0.1, 0, 3300, 10000),
    (59, 0.1, 0, 10000, 5000),

    (60, 0, 0.1, 0, 3300),
    (61, 0, 0.1, 0, 5000),
    (62, 0, 0.1, 0, 10000),
    (63, 0, 0.1, 3300, 5000),
    (64, 0, 0.1, 3300, 10000),
    (65, 0, 0.1, 10000, 5000),

    (66, 0.05, 0.1, 0, 3300),
    (67, 0.05, 0.1, 0, 5000),
    (68, 0.05, 0.1, 0, 10000),
    (69, 0.05, 0.1, 3300, 5000),
    (70, 0.05, 0.1, 3300, 10000),
    (71, 0.05, 0.1, 10000, 5000),

    (72, 0.1, 0.1, 0, 3300),
    (73, 0.1, 0.1, 0, 5000),
    (74, 0.1, 0.1, 0, 10000),
    (75, 0.1, 0.1, 3300, 5000),
    (76, 0.1, 0.1, 3300, 10000),
    (77, 0.1, 0.1, 10000, 5000)

]


if __name__ == "__main__":
    # with multiprocessing.Pool(processes=len(param_sets)) as pool:
    #     pool.starmap(run_experiment, param_sets)

    batch_size = 20
    for i in range(0, len(param_sets), batch_size):
        batch = param_sets[i:i+batch_size]
        with multiprocessing.Pool(processes=batch_size) as pool:
            pool.starmap(run_experiment, batch)

