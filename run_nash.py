from MARL import Employee,ParentalLeave,NashQLearning
# from MARL_early_stopping import Employee,ParentalLeave,NashQLearning

from datetime import datetime

#(w_yrs,pos,opt,age,time)
employee0 = Employee((7, 3, 1, 8, 8)) 
employee1 = Employee((7, 3, 1, 8, 8))

if __name__ == "__main__":
    experiment_num=80
    alpha,delta=0,0 
    c1,c2 = 3300,3300
    print(experiment_num,alpha,delta)
    print(c1,c2)

    # Create an instance of the ParentalLeave environment
    env = ParentalLeave(employees=[employee0, employee1], c1=c1, c2=c2, alpha=alpha, delta=delta)
    
    start_time = datetime.now() 
    # Create an instance of the NashQLearning agent          
    #nash_q_agent = NashQLearning(experiment_num=0, env=env, discount_factor=0.99, learning_rate=1, max_iter=5)
    nash_q_agent = NashQLearning(experiment_num=experiment_num,env=env, discount_factor=0.99,learning_rate=0.5)                         
    # Train the agent and get the Q-tables
    q_table_employee0, q_table_employee1 = nash_q_agent.fit()
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time  
    print(f"Running time:{elapsed_time}")

   








