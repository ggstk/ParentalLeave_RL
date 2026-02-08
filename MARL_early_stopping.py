import os
import numpy as np
import random
import discrete_env
from tqdm import tqdm
import nashpy as nash
import pickle
import json
import warnings
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
# RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave"
save_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models"
Q_learning_curve_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/Q_learning_curve/"
Q_tables_dir="/home/zhaolixue/ZHAOLIXUE/ParentalLeave/Q_tables/"


class Employee:
    def __init__(self, initial_state=None, actions=('근무', '휴직 사용')):
        self.initial_state = initial_state
        self.actions = actions

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()


class ParentalLeave(discrete_env.DiscreteEnv):
    def __init__(self,
                 max_yrs_pos=(6, 6, 6, 8, 8),  # [6, 6, 6, 8, 8],# 각 직급에서 최대로 머물 수 있는 기간. 승진 지속 누락시 마지막 연차에 계속 머묾
                 yrs_for_pos=(4, 4, 4, 5, 5),  # [4, 4, 4, 5, 5],각 직급에서 승진까지 최소 소요기간.
                 max_years=25,
                 alpha=0,  # Adjustment factor based on remaining parental leave of the other one
                 delta=0,
                 c1=0,
                 c2=0,
                 employees=[Employee(), Employee()]):
        '''
        this class is a representation of the PrentalLeave
        max_yrs_pos(list):각 직급에서 최대로 머물 수 있는 기간. 승진 지속 누락시 마지막 연차에 계속 머묾
        yrs_for_pos(list):각 직급에서 승진까지 최소 소요기간.    신입-과장: 평균 8.4년, 신임과장-부장: 8.5년
        https://kosis.kr/statHtml/statHtml.do?orgId=389&tblId=DT_389_2013_038&conn_path=I2
        n_position(int):직급의 수  {사원, 대리, 과장, 차장, 부장}
        n_option(int):  {0번, 1번}, 육아휴직 남은 횟수
        max_age(int): {0,1,2,3,4,5,6,7,8, over 9}, 아이의 나이
        max_year(int):최대 근속 연수
        '''

        self.max_yrs_pos = max_yrs_pos
        self.yrs_for_pos = yrs_for_pos
        self.n_position = np.shape(self.max_yrs_pos)[0]  # {사원, 대리, 과장, 차장, 부장}, 직급
        self.n_options = 2  # {0번, 1번}, 육아휴직 남은 횟수
        self.max_age = 10
        self.max_years = max_years
        self.alpha = alpha
        self.delta = delta
        self.employees = employees
        self.c1 = c1
        self.c2 = c2
        add_years = True  # 휴직기간 가산. 법적으로 무조건 가산해야함.
        self.add_years = add_years
        self.annual_cost = 0  # 1년간 생활비
        self.states = np.sum(
            self.max_yrs_pos) * self.n_options * self.max_age * self.max_years + 1  # 상태의 가능한 경우의 수 + Terminal state
        nS = self.states ** 2  # 상태의 가능한 경우의 수 + Terminal state
        nA = 4
        joint_actions = [(0, 0), (0, 1), (1, 0),(1, 1)]
        self.joint_actions = joint_actions
        self.nS = nS
        self.nA = nA  # 0:("근무","근무"), 1:("근무","휴직 사용"),2:("휴직 사용","근무"),3:("휴직 사용","휴직 사용")
        self.action_name = {0: "근무", 1: "휴직 사용"}
        self.current_states = (self.employees[0].initial_state, self.employees[1].initial_state)
        P = {}
        self.P = P
        isd = np.zeros(nS)
        self.make_joint_states()
        self.state_index = -1
        
   
        super(ParentalLeave, self).__init__(nS, nA, P, isd)

    def get_employee_0(self):
        return self.employees[0]

    def get_employee_1(self):
        return self.employees[1]

    def make_joint_states(self):
        self.joint_states = [
            ((w_yrs, pos, 1, age, age),
             (w_yrs, pos, 1, age, age))
            for pos in range(self.n_position-1)  
            for w_yrs in range(self.max_yrs_pos[pos])  # 직급에서 최대로 머물 수 있는 기간
            for age in range(self.max_age-1)#234
            # for time in range(self.max_years)
        ]
        self.joint_states_num=len(self.joint_states)
        reset_states_path = os.path.join(save_dir, f"reset_states.json")
        with open(reset_states_path, 'w') as f:
            for state in self.joint_states:
                f.write(json.dumps(state)+"," + '\n')

        

    def random_joint_state(self):
        self.state_index = (self.state_index+1) % len(self.joint_states)
        selected_state = self.joint_states[-self.state_index]
        return selected_state

    def get_reward(self, pos0, pos1, opt0, opt1, age0, age1, time0, time1, action0, action1):
        if time0 == self.max_years:  # 직원0은 육아휴직을 사용할 수 없는 상태에서 육아휴직을 사용했다.
            reward0 = -self.annual_cost
            if time1 == self.max_years:  # 직원1은 육아휴직을 사용할 수 없는 상태에서 육아휴직을 사용했다.
                reward1 = -self.annual_cost
                return [reward0, reward1]
            else:
                if action1 == '근무':
                    reward1 = self.salary(pos1) - self.annual_cost
                    return [reward0, reward1]
                elif action1 == '휴직 사용':
                    if opt1 == 0 or age1 == self.max_age - 1:
                        reward1 = -self.annual_cost
                        return [reward0, reward1]
                    else:
                        reward1 = max(840.0, min(self.salary(pos1) * 0.8, 2150)) + self.c2
                        return [reward0, reward1]
        elif time1 == self.max_years:
            reward1 = -self.annual_cost
            if action0 == '근무':
                reward0 = self.salary(pos0) - self.annual_cost
                return [reward0, reward1]
            elif action0 == '휴직 사용':
                if opt0 == 0 or age0 == self.max_age - 1:
                    reward0 = -self.annual_cost
                    return [reward0, reward1]
                else:
                    reward0 = max(840.0, min(self.salary(pos0) * 0.8, 2150)) + self.c1
                    return [reward0, reward1]

        if action0 == '근무' and action1 == '근무':  # time == self.max_years,action = "근무'일 때?
            reward0 = self.salary(pos0) - self.annual_cost
            reward1 = self.salary(pos1) - self.annual_cost
            return [reward0, reward1]
        elif action0 == '근무' and action1 == '휴직 사용':
            reward0 = self.salary(pos0) - self.annual_cost
            if opt1 == 0 or age1 == self.max_age - 1:
                reward1 = -self.annual_cost  # 육아휴직 횟수 없거나 아이의 나이가 10을 넘을 때 보조금 같은 것 없어요.
            else:
                reward1 = max(840.0,min(self.salary(pos1) * 0.8, 2150)) + self.c2-self.annual_cost
            return [reward0, reward1]
        elif action0 == '휴직 사용' and action1 == '근무':
            if opt0 == 0 or age0 == self.max_age - 1:
                reward0 = -self.annual_cost
            else:
                reward0 = max(840.0,min(self.salary(pos0) * 0.8, 2150)) + self.c1-self.annual_cost
            reward1 = self.salary(pos1) - self.annual_cost
            return [reward0, reward1]
        elif action0 == '휴직 사용' and action1 == '휴직 사용':
            if opt0 == 0 or age0 == self.max_age - 1:
                reward0 = -self.annual_cost-self.annual_cost
            else:
                reward0 = max(840.0, min(self.salary(pos0) * 0.8, 2150)) + self.c1-self.annual_cost
            if opt1 == 0 or age1 == self.max_age - 1:
                reward1 = -self.annual_cost
            else:
                reward1 = max(840.0, min(self.salary(pos1) * 0.8, 2150)) + self.c2-self.annual_cost

            return [reward0, reward1]

    def salary(self, pos):
        # https://www.wage.go.kr/whome/wage/wagesearch.do?menuNo=102010200
        # For 5 levels

        if pos == 0:
            return 3000 # 2400 + U+ -> 1800 + U+
        elif pos == 1:
            return 3600 # 2880 + U+ -> 1800 + U+
        elif pos == 2:
            return 4700 # 3760 + U+ -> 1800 + U+
        elif pos == 3:
            return 5300 # 4240 + U+ -> 1800 + U+
        elif pos == 4:
            return 6600 # 5280 + U+ -> 1800 + U+

        # if pos == 0:
        #     return 4200
        # elif pos == 1:
        #     return 5200
        # elif pos == 2:
        #     return 6600
        # elif pos == 3:
        #     return 7600
        # elif pos == 4:
        #     return 9100

        # For 4 levels
        # if pos == 0:
        #     return 4500
        # elif pos == 1:
        #     return 6700
        # elif pos == 2:
        #     return 9200
        # elif pos == 3:
        #     return 11100

    # 승진 확률 함수
    def get_promo_prob(self, pos0, pos1, opt0, opt1):
        # https://donidang.tistory.com/1483
        # 사원 -대리 : 100%
        # 대리-과장: 60~70%
        # 과장-차장: 40~50%
        # 차장-부장: 30~0
        q_pos = [1.0, 0.65, 0.45, 0.35]

        # Calculate probabilities for Employee 0
        if pos0 < self.n_position - 1:
            q0 = q_pos[pos0] - self.delta * (1 - opt0)
        else:
            q0 = 1.0  # Highest position, assumed to be always promoted

        # Calculate probabilities for Employee 1
        if pos1 < self.n_position - 1:
            q1 = q_pos[pos1] - self.delta * (1 - opt1)
        else:
            q1 = 1.0  # Highest position, assumed to be always promoted

        if pos0 == pos1:
            if (opt0 == 0) and (opt1 == 1):
                q0 -= self.alpha
            elif (opt0 == 1) and (opt1 == 0):
                q1 -= self.alpha
        q0 = min(1, max(0, q0))
        q1 = min(1, max(0, q1))
        
        return [q0, q1]

    def step(self, actions):
        state0 = self.current_states[0]
        state1 = self.current_states[1]

        action0 = actions[0]
        action1 = actions[1]

        w_yrs0, pos0, opt0, age0, time0 = state0[0], state0[1], state0[2], state0[3], state0[4]
        w_yrs1, pos1, opt1, age1, time1 = state1[0], state1[1], state1[2], state1[3], state1[4]

        # update states
        n_age0 = min(age0 + 1, self.max_age - 1)
        n_age1 = min(age1 + 1, self.max_age - 1)

        n_time0 = min(time0 + 1, self.max_years)
        n_time1 = min(time1 + 1, self.max_years)

        n_w_yrs0 = min(w_yrs0 + 1, self.max_yrs_pos[pos0])
        n_w_yrs1 = min(w_yrs1 + 1, self.max_yrs_pos[pos1])

        done = True if (n_time0 >= self.max_years or n_time1 >= self.max_years) else False

        # calculate the reward
        rew = self.get_reward(pos0, pos1, opt0, opt1, age0, age1, time0, time1, self.action_name[action0],
                              self.action_name[action1])

        # state transtion for employee0
        if action0 == 1 and (age0 >= self.max_age - 1 or state0[2] == 0):
            n_s0 = (0, 0, 0, 0, self.max_years)
            next_state0 = (1.0, n_s0, 0.0, done)
        else:
            n_opt0 = opt0 - action0
            if self.employees[0].actions[action0] == '근무':  # a == 0
                if w_yrs0 == self.max_yrs_pos[pos0] - 1:
                    n_s0 = (w_yrs0, pos0, n_opt0, n_age0, n_time0)
                    next_state0 = (1.0, n_s0, rew[0], done)
                elif self.yrs_for_pos[pos0] - 1 <= w_yrs0 < self.max_yrs_pos[pos0] - 1:
                    promo_prob = self.get_promo_prob(pos0, pos1, state0[2], state1[2])[0]
                    n_s0_promo = (0, min(pos0 + 1, self.n_position - 1), n_opt0, n_age0, n_time0)
                    n_s0_no_promo = (n_w_yrs0, pos0, n_opt0, n_age0, n_time0)
                    next_state0 = [(promo_prob, n_s0_promo, rew[0], done),
                                   (1.0 - promo_prob, n_s0_no_promo, rew[0], done)]
                else:
                    n_s0 = (n_w_yrs0, pos0, n_opt0, n_age0, n_time0)
                    next_state0 = (1.0, n_s0, rew[0], done)
            elif self.employees[0].actions[action0] == '휴직 사용':  # a == 1
                if w_yrs0 == self.max_yrs_pos[pos0] - 1:
                    n_s0 = (w_yrs0, pos0, n_opt0, n_age0, n_time0)
                    next_state0 = (1.0, n_s0, rew[0], done)
                elif self.yrs_for_pos[pos0] - 1 <= w_yrs0 < self.max_yrs_pos[pos0] - 1:
                    promo_prob = self.get_promo_prob(pos0, pos1, state0[2], state1[2])[0]
                    if np.isclose(promo_prob, 1.0):
                        n_s0 = (0, min(pos0 + 1, self.n_position - 1), n_opt0, n_age0, n_time0)
                        next_state0 = (1.0, n_s0, rew[0], done)
                    else:# eliminates promotion eligibility     -->allow promotion review upon return
                        # n_s0 = (state0[0] + self.add_years, pos0, n_opt0, n_age0, n_time0)
                        # next_state0 = (1.0, n_s0, rew[0], done)
                        n_s0_promo = (0, min(pos0 + 1, self.n_position - 1), n_opt0, n_age0, n_time0)
                        n_s0_no_promo = (state0[0] + self.add_years, pos0, n_opt0, n_age0, n_time0)
                        next_state0 = [(promo_prob, n_s0_promo, rew[0], done),
                                   (1.0 - promo_prob, n_s0_no_promo, rew[0], done)]
                
                else:
                    n_s0 = (state0[0] + self.add_years, pos0, n_opt0, n_age0, n_time0)
                    next_state0 = (1.0, n_s0, rew[0], done)

        # state transition for employee1
        if action1 == 1 and (age1 >= self.max_age - 1 or state1[2] == 0):
            n_s1 = (0, 0, 0, 0, self.max_years)
            next_state1 = (1.0, n_s1, 0.0, done)
        else:
            n_opt1 = opt1 - action1
            if self.employees[1].actions[action1] == '근무':  # a == 0
                if w_yrs1 == self.max_yrs_pos[pos1] - 1:
                    n_s1 = (w_yrs1, pos1, n_opt1, n_age1, n_time1)
                    next_state1 = (1.0, n_s1, rew[1], done)
                elif self.yrs_for_pos[pos1] - 1 <= w_yrs1 < self.max_yrs_pos[pos1] - 1:
                    promo_prob = self.get_promo_prob(pos0, pos1, state0[2], state1[2])[1]
                    n_s1_promo = (0, min(pos1 + 1, self.n_position - 1), n_opt1, n_age1, n_time1)
                    n_s1_no_promo = (n_w_yrs1, pos1, n_opt1, n_age1, n_time1)
                    next_state1 = [(promo_prob, n_s1_promo, rew[1], done),
                                   (1.0 - promo_prob, n_s1_no_promo, rew[1], done)]
                else:
                    n_s1 = (n_w_yrs1, pos1, n_opt1, n_age1, n_time1)
                    next_state1 = (1.0, n_s1, rew[1], done)
            elif self.employees[1].actions[action1] == '휴직 사용':  # a == 1
                if w_yrs1 == self.max_yrs_pos[pos1] - 1:
                    n_s1 = (w_yrs1, pos1, n_opt1, n_age1, n_time1)
                    next_state1 = (1.0, n_s1, rew[1], done)
                elif self.yrs_for_pos[pos1] - 1 <= state1[0] < self.max_yrs_pos[pos1] - 1:
                    promo_prob = self.get_promo_prob(pos0, pos1, state0[2], state1[2])[1]
                    if np.isclose(promo_prob, 1.0):
                        n_s1 = (0, min(pos1 + 1, self.n_position - 1), n_opt1, n_age1, n_time1)
                        next_state1 = (1.0, n_s1, rew[1], done)
                    else:
                        # n_s1 = (w_yrs1 + self.add_years, pos1, n_opt1, n_age1, n_time1)
                        # next_state1 = (1.0, n_s1, rew[1], done)
                        n_s1_promo = (0, min(pos1 + 1, self.n_position - 1), n_opt1, n_age1, n_time1)
                        n_s1_no_promo = (n_w_yrs1, pos1, n_opt1, n_age1, n_time1)
                        next_state1 = [(promo_prob, n_s1_promo, rew[1], done),
                                    (1.0 - promo_prob, n_s1_no_promo, rew[1], done)]
                else:
                    n_s1 = (w_yrs1 + self.add_years, pos1, n_opt1, n_age1, n_time1)
                    next_state1 = (1.0, n_s1, rew[1], done)

        next_states = (next_state0, next_state1)

        return next_states, rew, done, {}

    def _reset(self):
        states = self.random_joint_state()
        return states

class NashQLearning:
    def __init__(self,
                 env=ParentalLeave(),
                 learning_rate=0.5,
                 max_iter=1000,
                 discount_factor=0.7,
                 decision_strategy="epsilon-greedy",
                 epsilon=0.99,
                 random_state=42,
                 q_table0={},
                 q_table1={},
                 continue_train=False,
                 experiment_num=0):
        self.env = env
        joint_actions = env.joint_actions
        self.joint_actions = joint_actions
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.discount_factor = discount_factor
        self.decision_strategy = decision_strategy
        self.epsilon = epsilon
        random.seed(random_state)
        self.q_table0 = q_table0
        self.q_table1 = q_table1
        self.experiment_num = experiment_num
        self.continue_train = continue_train
        single_q_value_path="/home/zhaolixue/ZHAOLIXUE/ParentalLeave/results/Parental_Leave_pnot1_U0_delta0BaseValue.xlsx"
        self.initial_q_value= pd.read_excel(single_q_value_path)
        self.initial_q_value["State"] = self.initial_q_value["State"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
)


    def _get_q_value(self, state, action, q_table):
        target_state=()
        if q_table == self.q_table0:
            target_state = state[0]
        elif q_table == self.q_table1:
            target_state = state[1]
        q_value = q_table.get((state, action))

        if q_value is None:
            if target_state[4] == 25:
                q_value=0
            else:
                value = self.initial_q_value.loc[self.initial_q_value["State"] == target_state, "Value"].values
                q_value = value[0]  

        return  q_value  

    def _compute_pi(self, states,q_table):
        """
            compute pi (nash)
        """
        if isinstance(states[0], tuple):
            states = ([states[0]], states[1])
        if isinstance(states[1], tuple):
            states = (states[0], [states[1]])
     
        pi_group = []
        for state0 in states[0]:
            for state1 in states[1]:
                q_values0 = {action: self._get_q_value((state0[1], state1[1]), action, self.q_table0) for action in
                             self.joint_actions}
                q_values1 = {action: self._get_q_value((state0[1], state1[1]), action, self.q_table1) for action in
                             self.joint_actions}
                q_values0_matrix = np.array(list(q_values0.values())).reshape(2, 2)
                q_values1_matrix = np.array(list(q_values1.values())).reshape(2, 2)

                game = nash.Game(q_values0_matrix, q_values1_matrix)

                # equilibria = game.lemke_howson(initial_dropped_label=0)
                # equilibria = game.support_enumeration()
                equilibria = game.lemke_howson_enumeration()
                # equilibria =game.vertex_enumeration()

                pi_list = list(equilibria)
                unique_equilibria = []
                for eq in pi_list:
                    if not any(np.array_equal(eq, u_eq) for u_eq in unique_equilibria):
                        unique_equilibria.append(eq)
            
                best_nash_q = -float('inf')  
                best_equilibrium = None  
                for eq in unique_equilibria:
                    pi, pi_o = eq #(array([0., 1.]), array([1., 0.]))
                    action0 = np.argmax(pi)
                    action1 = np.argmax(pi_o)
                    nash_q = self._get_q_value((state0[1], state1[1]), (action0, action1), q_table)

                    if nash_q >= best_nash_q:
                        best_nash_q = nash_q
                        best_equilibrium = eq
                pi_group.append(best_equilibrium)
        return pi_group

    def _compute_nash_q(self, state, pi_group, q):
      
        if isinstance(state[0], tuple):
            state = ([state[0]], state[1])
        if isinstance(state[1], tuple):
            state = (state[0], [state[1]])
        state = (state)
        nash_q = 0
        i = 0
        for t0 in state[0]:
            for t1 in state[1]:
                pi, pi_o = pi_group[i]
                action0 = np.argmax(pi)
                action1 = np.argmax(pi_o)
                nash_q += t0[0] * t1[0] * (
                        pi[action0] * pi_o[action1] * self._get_q_value((t0[1], t1[1]), (action0, action1), q))
                i += 1
        return nash_q

    def update_epsilon(self):
        self.epsilon *= 0.99
        if self.epsilon < 0.50:  #0.5、0.2
            self.epsilon = 0.50
        return self.epsilon

    def update_learning_rate(self):
        self.learning_rate *= 0.99 
        self.learning_rate = round(self.learning_rate, 4)
        if self.learning_rate < 0.05:  
            self.learning_rate = 0.05

    def _update_q_value(self, states, action, rewards, q_table, q_table_o, nash_q):
        old_q_value = self._get_q_value(states, action, q_table)
        
        if q_table==self.q_table0:
            reward=rewards[0]
            new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * nash_q - old_q_value)
            
            if states[0][3] == 9 or states[0][2] == 0:  
                if action[0] == 1:
                    self.q_table0[(states, action)] = 0
                    self.q_table1[(states, action)] = self._get_q_value(states,action, self.q_table1)
                    self.skip_update0 = True
                elif action[0] == 0:
                    if (states[0][0] == self.env.max_yrs_pos[states[0][1]] - 1
                            or states[0][1] == self.env.n_position - 1): 
                        #print(self.env.current_states)
                        distance1 = 24 - states[0][4]
                        q0_end = self.calculate_value(distance1, rewards[0], self.discount_factor)
                        self.q_table0[(states, action)] = q0_end
                        self.q_table1[(states, action)] = self._get_q_value(states,action, self.q_table1)
                    else:
                        q_table[(states, action)] = new_q_value
                        q_table_o[(states, action)] = self._get_q_value(states,action, q_table_o)
            else:
                q_table[(states, action)] = new_q_value
                # if self.env.c1 == self.env.c2:
                #     q_table_o[((states[1], states[0]), (action[1], action[0]))] = new_q_value

        elif q_table==self.q_table1:
            reward=rewards[1]
            new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * nash_q - old_q_value)
       
            if states[1][3] == 9 or states[1][2] == 0: 
                if action[1] == 1:
                    self.q_table0[(states, action)] = self._get_q_value(states, action, self.q_table0)
                    self.q_table1[(states, action)] = 0
                    
                elif action[1] == 0:
                    if (states[1][0] == self.env.max_yrs_pos[states[1][1]] - 1
                            or states[1][1] == self.env.n_position - 1):
                        distance2 = 24 - states[1][4]
                        q1_end = self.calculate_value(distance2, rewards[1], self.discount_factor)
                        self.q_table1[(states, action)] = q1_end
                        self.q_table0[(states,action)] = self._get_q_value(states,action, self.q_table0)
                    else:
                        q_table[(states, action)] = new_q_value
                        q_table_o[(states, action)] = self._get_q_value(states,action, q_table_o)
            else:
                q_table[(states, action)] = new_q_value
                # if self.env.c1 == self.env.c2:
                #     q_table_o[((states[1], states[0]), (action[1], action[0]))] = new_q_value
                        
        #q_table[(self.env.current_states, action)] = new_q_value
        #q_table_o[((self.env.current_states[1], self.env.current_states[0]), (action[1], action[0]))] = new_q_value
       

    def _select_actions(self, epsilon):
        if random.uniform(0, 1) < epsilon:
            action0 = random.randrange(len(self.env.employees[0].actions))
            action1 = random.randrange(len(self.env.employees[0].actions))
            #print(action0,action1)
        else:
            q_values0 = {action: self._get_q_value(self.env.current_states, action, self.q_table0) for action in
                         self.joint_actions}
            q_values1 = {action: self._get_q_value(self.env.current_states, action, self.q_table1) for action in
                         self.joint_actions}
            q_values0_matrix = np.array(list(q_values0.values())).reshape(2, 2)
            q_values1_matrix = np.array(list(q_values1.values())).reshape(2, 2)

            game = nash.Game(q_values0_matrix, q_values1_matrix)
            equilibriums = list(game.lemke_howson_enumeration())
            random_num=random.randrange(len(equilibriums))
            greedy_equilibrium = equilibriums[random_num]
            
            if len(np.where(greedy_equilibrium[0] == 1)[0]) == 0 or len(
                    np.where(greedy_equilibrium[1] == 1)[0]) == 0:  # No strict equilibrium found
                action0 = random.randrange(len(self.env.employees[0].actions))
                action1 = random.randrange(len(self.env.employees[0].actions))
            else:  # Select the movements corresponding to the nash equilibrium
                action0 = np.where(greedy_equilibrium[0] == 1)[0][0]
                action1 = np.where(greedy_equilibrium[1] == 1)[0][0]
            if self.env.current_states[0][2] == 0 or self.env.current_states[0][3] == 9:
                action0 = 0
            if self.env.current_states[1][2] == 0 or self.env.current_states[1][3] == 9:
                action1 = 0
        return action0, action1

    def calculate_value(self, distance, reward, discount_factor):
        if (distance <= 0):
            return reward
        else:
            return reward + discount_factor * self.calculate_value(distance - 1, reward, discount_factor)

    def collect_final_nash_trajectory(self):
        # self.env.reset(seed=0)
        # random.seed(0)
        # print("Collecting trajectory under Nash equilibrium...")
        trajectory = []
        #print("initial_state",self.env.current_states)
        while True:
            action0, action1 = self._select_actions(epsilon=0)
            action0, action1 = int(action0), int(action1)

            trajectory.append({
                "state": self.env.current_states,
                "action": (action0, action1),
            })

            next_states, rewards, done, info = self.env.step((action0, action1))
            
            if len(next_states[0]) == 2:
                i = categorical_sample([t[0] for t in next_states[0]], self.env.np_random)
                next_states = (next_states[0][i], next_states[1])
              
            if len(next_states[1]) == 2:
                i = categorical_sample([t[0] for t in next_states[1]], self.env.np_random)
                next_states = (next_states[0], next_states[1][i])

            self.env.current_states = (next_states[0][1], next_states[1][1])
            if done:
                end_state = trajectory[-1]["state"]
                # if end_state[0][2] == 1 and end_state[1][2] == 0:
                #     print(trajectory)
                break
        trajectory_path = os.path.join(save_dir, f"final_nash_trajectory_exp{self.experiment_num}.json")
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory, f, indent=4)
        return trajectory


    def calculate_prob(self,n):
        state_num=len(self.env.joint_states)
        probs=[]
        counter=0
        for num in range(n):
            counter=0
            for state in self.env.joint_states:
                self.env.reset(seed=num)
                random.seed(num)
                self.env.current_states=state
                trajectory=self.collect_final_nash_trajectory()
                end_state=trajectory[-1]["state"]
                if end_state[0][2]==0 and end_state[1][2]==0:
                    counter +=1
            prob=counter/state_num
            probs.append(prob)
             
        prob_mean = np.mean(probs)
        std_err = stats.sem(probs) #SE = SD / sqrt(n)
        confidence = 0.95 #95% CI
        t_value = stats.t.ppf((1 + confidence) / 2, df=len(probs)-1) 
        margin_of_error = t_value * std_err  
        return prob_mean,margin_of_error

    def calculate_one_leave_prob(self,n):
            state_num=len(self.env.joint_states)
            probs=[]
            counter=0
            for num in range(n):
                counter=0
                for state in self.env.joint_states:
                    self.env.reset(seed=num)
                    random.seed(num)
                    self.env.current_states=state
                    trajectory=self.collect_final_nash_trajectory()
                    end_state=trajectory[-1]["state"]
                    if end_state[0][2]==0 and end_state[1][2]==1:
                        counter +=1
                    if end_state[0][2]==1 and end_state[1][2]==0:
                        counter +=1
                prob=counter/state_num
                probs.append(prob)
                
            prob_mean = np.mean(probs)
            std_err = stats.sem(probs) #SE = SD / sqrt(n)
            confidence = 0.95 #95% CI
            t_value = stats.t.ppf((1 + confidence) / 2, df=len(probs)-1) 
            margin_of_error = t_value * std_err  
            return prob_mean,margin_of_error


    def calculate_each_prob(self, n):
        state_num = len(self.env.joint_states)
        counter1 = 0
        counter2 = 0
        probs1,probs2=[],[]

        for num in range(n):
            counter1,counter2=0,0
            for state in self.env.joint_states:
                self.env.reset(seed=num)
                random.seed(num)
                self.env.current_states = state
                trajectory = self.collect_final_nash_trajectory()
                end_state = trajectory[-1]["state"]
                if end_state[0][2] == 0 :
                    counter1 += 1
                if end_state[1][2] == 0:
                    counter2 += 1
            prob1=counter1/state_num
            prob2=counter2/state_num
            probs1.append(prob1)
            probs2.append(prob2)

        probs1_mean = np.mean(probs1)
        probs2_mean = np.mean(probs2)

        std_err1 = stats.sem(probs1) #SE = SD / sqrt(n)
        std_err2 = stats.sem(probs2) #SE = SD / sqrt(n)

        confidence = 0.95 #95% CI
        t_value1 = stats.t.ppf((1 + confidence) / 2, df=len(probs1)-1)  
        t_value2 = stats.t.ppf((1 + confidence) / 2, df=len(probs2)-1)  
        margin_of_error1 = t_value1 * std_err1  
        margin_of_error2 = t_value2 * std_err2  
    
        return probs1_mean,probs2_mean,margin_of_error1,margin_of_error2


    def fit(self, return_history=False):
        Q0, Q1 = self.q_table0, self.q_table1
        state_tracker = [self.env.current_states]
        trajectory = []
        studied_state=[]
        covergence_episode=[]
        
        update_num=5000

        for init_state in reversed(self.env.joint_states):
            self.epsilon = 0.99
            self.learning_rate = 0.5
            done_counter = 0
            # convergence 
            prev_q_vector = None
            stable_counter = 0
            tol = 1e-2
            required_stable_episodes = 200

            Q0_values1, Q0_values2, Q0_values3, Q0_values4, episodes = [], [], [], [], []
            Q1_values1, Q1_values2, Q1_values3, Q1_values4 = [], [], [], []
            for episode in range(update_num):
                self.env.current_states = init_state
                # print( "initial", self.env.current_states)
                done=False

                while not done:
                    # print(self.env.current_states)
                    action0, action1 = self._select_actions(epsilon=self.epsilon)
                    action0, action1 = (action0, action1)

                    # Use step function to get the next state, reward, and done
                    next_states, rewards, done, info = self.env.step((action0, action1))
                    reward0,reward1 =rewards

                    if self.env.current_states[0][3] == 9 or self.env.current_states[0][2] == 0:
                        if action0 == 1:
                            Q0[(self.env.current_states, (action0, action1))] = 0
                            Q1[(self.env.current_states, (action0, action1))] = self._get_q_value(self.env.current_states,
                                                                                        (action0, action1), Q1)
                            self.skip_update0 = True
                        elif action0 == 0:
                            if (self.env.current_states[0][0] == self.env.max_yrs_pos[self.env.current_states[0][1]] - 1
                                    or self.env.current_states[0][1] == self.env.n_position - 1): 
                                #print(self.env.current_states)
                                distance1 = 24 - self.env.current_states[0][4]
                                q0_end = self.calculate_value(distance1, reward0, self.discount_factor)
                                Q0[(self.env.current_states, (action0, action1))] = q0_end
                                Q1[(self.env.current_states, (action0, action1))] = self._get_q_value(self.env.current_states,
                                                                                                        (action0, action1), Q1)


                    if self.env.current_states[1][3] == 9 or self.env.current_states[1][2] == 0:  

                        if action1 == 1:
                            Q0[(self.env.current_states, (action0, action1))] = self._get_q_value(self.env.current_states,
                                                                                                    (action0, action1), Q0)
                            Q1[(self.env.current_states, (action0, action1))] = 0
                            self.skip_update1 = True

                        elif action1 == 0:
                            if (self.env.current_states[1][0] == self.env.max_yrs_pos[self.env.current_states[1][1]] - 1
                                    or self.env.current_states[1][1] == self.env.n_position - 1):

                                distance2 = 24 - self.env.current_states[1][4]
                                q1_end = self.calculate_value(distance2, reward1, self.discount_factor)
                                Q1[(self.env.current_states, (action0, action1))] = q1_end
                                Q0[(self.env.current_states, (action0, action1))] = self._get_q_value(self.env.current_states,
                                                        (action0, action1), Q0)
                    
                    trajectory.append((self.env.current_states, (action0, action1), rewards, next_states))
                

                    if len(next_states[0]) == 2:
                        i = categorical_sample([t[0] for t in next_states[0]], self.env.np_random)
                        next_states = (next_states[0][i], next_states[1])
                    if len(next_states[1]) == 2:
                        i = categorical_sample([t[0] for t in next_states[1]], self.env.np_random)
                        next_states = (next_states[0], next_states[1][i])
                    # Transition to the next state
                    self.env.current_states = (next_states[0][1], next_states[1][1])

                    if done:
                        studying_trajectory = []
                        for state, actions, rewards, next_states in trajectory:
                            if state in studied_state:
                                break
                            studying_trajectory.append((state, actions, rewards, next_states))
                        for state, actions, rewards, next_states in reversed(studying_trajectory):
                            action0,action1=actions
                            pi_group = self._compute_pi(next_states,Q0)
                            nash_q = self._compute_nash_q(next_states, pi_group, self.q_table0)
                            nash_q_o = self._compute_nash_q(next_states, pi_group, self.q_table1)

                            self._update_q_value(state,(int(action0), int(action1)), rewards, Q0, Q1, nash_q)
                            self._update_q_value(state,(int(action0), int(action1)), rewards, Q1, Q0, nash_q_o)

                        trajectory=[]
                        done_counter +=1
                        if done_counter % 10 == 0:
                            self.update_learning_rate()
                            self.update_epsilon()

                Q0_value1 = self._get_q_value(
                    init_state, (0, 0), Q0)
                Q0_value2 = self._get_q_value(
                   init_state, (0, 1), Q0)
                Q0_value3 = self._get_q_value(
                    init_state, (1, 0), Q0)
                Q0_value4 = self._get_q_value(
                    init_state, (1, 1), Q0)
                
                Q1_value1 = self._get_q_value(
                    init_state, (0, 0), Q1)
                Q1_value2 = self._get_q_value(
                   init_state, (0, 1), Q1)
                Q1_value3 = self._get_q_value(
                    init_state, (1, 0), Q1)
                Q1_value4 = self._get_q_value(
                    init_state, (1, 1), Q1)
                current_q_vector = [Q0_value1, Q0_value2, Q0_value3, Q0_value4, 
                                    Q1_value1, Q1_value2, Q1_value3, Q1_value4]
                
                if prev_q_vector is not None:
                    max_diff = max(abs(c - p) for c, p in zip(current_q_vector, prev_q_vector))
                    # print(max_diff)
                    

                    relative_diff = max_diff / (max(abs(x) for x in current_q_vector) + 1e-10)
                    # print(max_diff,relative_diff)
                    if relative_diff < 1e-5:  #0.001% 
                        stable_counter += 1
                    else:
                        stable_counter = 0

                    # if max_diff < tol:
                    #     stable_counter += 1
                    # else:
                    #     stable_counter = 0 #
                
                prev_q_vector = current_q_vector.copy()
                episodes.append(episode)
                Q0_values1.append(Q0_value1)
                Q0_values2.append(Q0_value2)
                Q0_values3.append(Q0_value3)
                Q0_values4.append(Q0_value4)
                
                Q1_values1.append(Q1_value1)
                Q1_values2.append(Q1_value2)
                Q1_values3.append(Q1_value3)
                Q1_values4.append(Q1_value4)
                #Early Stopping
                if stable_counter >= required_stable_episodes and episode>1000:
                    # print(f"State {init_state} converged early at episode {episode}")
                    covergence_episode.append(episode+1)
                    break
                if episode==update_num-1:
                    covergence_episode.append(episode+1)
                   
        
            learning_curve_data = {
                'episodes': episodes,
                'Q0_(0,0)': Q0_values1,
                'Q0_(0,1)': Q0_values2,
                'Q0_(1,0)': Q0_values3,
                'Q0_(1,1)': Q0_values4,
                'Q1_(0,0)': Q1_values1,
                'Q1_(0,1)': Q1_values2,
                'Q1_(1,0)': Q1_values3,
                'Q1_(1,1)': Q1_values4
            }
            learning_curve_data_path = os.path.join(Q_learning_curve_dir, f"learning_curve_data_exp{self.experiment_num}_{init_state}.json")
        
            with open(learning_curve_data_path, 'w') as f:
                json.dump(learning_curve_data, f)
                
            studied_state.append(init_state)
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            # x=learning_curve_data["episodes"]

            # ax1.plot(x, learning_curve_data['Q0_(0,0)'], label="Q0_(Work, Work)")
            # ax1.plot(x, learning_curve_data['Q0_(0,1)'], label="Q0_(Work, Leave)")
            # ax1.plot(x, learning_curve_data['Q0_(1,0)'], label="Q0_(Leave, Stay)")
            # ax1.plot(x, learning_curve_data['Q0_(1,1)'], label="Q0_(Leave, Leave)")
            # ax1.set_title("Learning Curve for Agent 0", fontsize=14)
            # ax1.set_xlabel("Episodes")
            # ax1.set_ylabel("Q-Value")
            # ax1.legend()
            # ax1.grid(True, linestyle='--', alpha=0.6)

            # ax2.plot(x, learning_curve_data['Q1_(0,0)'], label="Q1_(Stay, Stay)")
            # ax2.plot(x, learning_curve_data['Q1_(0,1)'], label="Q1_(Stay, Leave)")
            # ax2.plot(x, learning_curve_data['Q1_(1,0)'], label="Q1_(Leave, Stay)")
            # ax2.plot(x, learning_curve_data['Q1_(1,1)'], label="Q1_(Leave, Leave)")
            # ax2.set_title("Learning Curve for Agent 1", fontsize=14)
            # ax2.set_xlabel("Episodes")
            # ax2.set_ylabel("Q-Value")
            # ax2.legend()
            # ax2.grid(True, linestyle='--', alpha=0.6)

            # plt.tight_layout()
            # plt.show()

            # image_dir="/home/zhaolixue/ZHAOLIXUE/ParentalLeave/Q_learning_curve/"
            # learning_curve_image_path=os.path.join( image_dir,f"learning_curve_q0_exp{self.experiment_num}_{init_state}.png")
            # plt.savefig(learning_curve_image_path)
            # print("image is saved")
            # plt.close()
            
        convergence_speed_path = os.path.join(Q_learning_curve_dir, f"learning_curve_data_exp{self.experiment_num}_convergence_episode.json")
        with open(convergence_speed_path, 'w') as f:
            json.dump(covergence_episode, f)  
   
        # save q tables
        q0_path = os.path.join(Q_tables_dir, f"Q0_exp{self.experiment_num}.pickle")
        with open(q0_path, 'wb') as f:
            pickle.dump(Q0, f)
        q1_path = os.path.join(Q_tables_dir, f"Q1_exp{self.experiment_num}.pickle")
        with open(q1_path, 'wb') as f:
            pickle.dump(Q1, f)
        if return_history:
            print(state_tracker)
        return Q0, Q1












