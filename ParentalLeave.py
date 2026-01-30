"""
Module from the assignments for UC Berkeley's Deep RL course.
Modify frozen_lake.py for our purpose(2022.11.08)
"""

import numpy as np
import discrete_env

class ParentalLeave(discrete_env.DiscreteEnv):

    def __init__(self, max_yrs_pos =  [6, 6, 6, 8, 8], yrs_for_pos = [4, 4, 4, 5, 5], u_plus = 4000, max_years = 25, delta = 0):
        self.max_yrs_pos = max_yrs_pos  # 각 직급에서 최대로 머물 수 있는 기간. 승진 지속 누락시 마지막 연차에 계속 머묾
        self.yrs_for_pos = yrs_for_pos  # 각 직급에서 승진까지 최소 소요기간, https://kosis.kr/statHtml/statHtml.do?orgId=389&tblId=DT_389_2013_038&conn_path=I2
        # 신입-과장: 평균 8.4년, 신임과장-부장: 8.5년
        self.n_position = np.shape(self.max_yrs_pos)[0]  # {사원, 대리, 과장, 차장, 부장}, 직급
        self.n_options = 2  # {0번, 1번}, 육아휴직 남은 횟수
        self.max_age = 10  # {0,1,2,3,4,5,6,7,8, over 9}, 아이의 나이
        self.max_years = max_years
        # 상태 = (직급에서 머문 기간, 직급, 육아휴직 남은 횟수, 아이 나이, 총 근속 연수)

        self.delta = delta # 휴직 시 승진 discount
        add_years = True  # 휴직기간 가산. 법적으로 무조건 가산해야함.

        self.utility = u_plus # 육아휴직에 대한 개인이 생각하는 양의 가치
        self.annual_cost = 0 # 1년간 생활비

        nS = np.sum(self.max_yrs_pos) * self.n_options * self.max_age * self.max_years + 1 # 상태의 가능한 경우의 수 + Terminal state
        self.nS = nS
        nA = 2
        action_name = {0: "근무", 1: "휴직 사용"}

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        # def to_s(left, pos, opt, age, time):
        #     s_idx = 0
        #     s_idx += time
        #     s_idx += age * self.max_years
        #     s_idx += opt * (self.max_years * self.max_age)
        #     s_idx += left * (self.max_years * self.max_age * self.n_options)
        #     cumul = np.sum(self.max_yrs_pos[:pos]) if pos != 0 else 0
        #     s_idx += self.max_years * self.max_age * self.n_options * cumul
        #     if time == self.max_years:
        #         s_idx = nS - 1 # Terminal state with value zero
        #     return s_idx

        # 첫 상태 결정 확률
        isd = np.zeros(nS)
        # 예를 들어, 첫 상태가 (4년 남음, 사원, 2번 남음, 1살, 0 시점)
        isd[self.to_s(4, 0, 2, 0, 0)] = 1.0

        for time in range(self.max_years):
            n_time = time + 1
            done = True if time == self.max_years else False
            for pos in range(self.n_position):
                for w_yrs in range(self.max_yrs_pos[pos]):
                    for opt in range(self.n_options):
                        for age in range(self.max_age):
                            n_age = min(age + 1, self.max_age-1)
                            n_pos = min(pos + 1, self.n_position - 1) # if get promoted

                            c_s = self.to_s(w_yrs, pos, opt, age, time)
                            for a in range(nA):
                                rew = self.get_reward(pos, opt, age, time, action_name[a])  # r(s,a)
                                li = P[c_s][a]  # 빈 list []

                                if a == 1 and (age >= self.max_age-1 or opt == 0):  # 사용 불가
                                    n_s = self.to_s(0, 0, 0, 0, self.max_years)
                                    li.append((1.0, n_s, 0.0, done))  # Go to terminal state (with value 0)
                                else:
                                    n_opt = opt - a
                                    if action_name[a] == '근무': # a == 0
                                        # n_opt = 0 if age >= self.max_age - 2 else opt
                                        if w_yrs == self.max_yrs_pos[pos] - 1: # 승진 영구누락자
                                            n_s = self.to_s(w_yrs, pos, n_opt, n_age, n_time)
                                            li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                        elif self.yrs_for_pos[pos]-1 <= w_yrs and w_yrs < self.max_yrs_pos[pos]-1: # 승진 대상
                                            promo_prob = self.get_promo_prob(pos, opt)
                                            # 승진 시
                                            n_s_promo = self.to_s(0, n_pos, n_opt, n_age, n_time)
                                            # 미승진 시
                                            n_s_not_promo = self.to_s(w_yrs + 1, pos, n_opt, n_age, n_time)
                                            if n_s_promo != n_s_not_promo:
                                                li.append((promo_prob, n_s_promo, rew, done))  # n_s는 다음 상태
                                                li.append((1.0 - promo_prob, n_s_not_promo, rew, done))  # n_s는 다음 상태
                                            else:
                                                li.append((1.0, n_s_promo, rew, done))  # n_s는 다음 상태
                                        else: # 승진 대상 아님
                                            n_s = self.to_s(w_yrs + 1, pos, n_opt, n_age, n_time)
                                            li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                    elif action_name[a] == '휴직 사용': # a == 1, Assume that employee cannot be promoted during parental leave
                                        # n_opt = max(opt - 1, 0)
                                        if w_yrs == self.max_yrs_pos[pos] - 1: # 승진 영구누락자
                                            n_s = self.to_s(w_yrs, pos, n_opt, n_age, n_time)
                                            li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                        elif self.yrs_for_pos[pos] - 1 <= w_yrs and w_yrs < self.max_yrs_pos[pos] - 1:  # 승진 대상
                                            promo_prob = self.get_promo_prob(pos, opt)
                                            if np.isclose([promo_prob], [1.0])[0]: # 승진 확률 1.0이면 자동 승진
                                                n_s = self.to_s(0, n_pos, n_opt, n_age, n_time)
                                                li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                            else:
                                                n_s = self.to_s(w_yrs + add_years, pos, n_opt, n_age, n_time)
                                                li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                        else:  # 승진 대상 아님
                                            n_s = self.to_s(w_yrs + add_years, pos, n_opt, n_age, n_time)
                                            li.append((1.0, n_s, rew, done))  # n_s는 다음 상태

                                        # if w_yrs == 1: # 승진 대상,
                                        #     if add_years == True:  # 승진기간 가산 시
                                        #         promo_prob = self.get_promo_prob(pos, n_opt)
                                        #         # 승진 시
                                        #         n_s = to_s(self.l_for_promo-1, n_pos, n_opt, n_age, n_time)
                                        #         li.append((promo_prob, n_s, rew, done))  # n_s는 다음 상태
                                        #         # 미승진 시
                                        #         n_s = to_s(w_yrs-1, pos, n_opt, n_age, n_time)
                                        #         li.append((1.0-promo_prob, n_s, rew, done))  # n_s는 다음 상태
                                        #     else:  # 승진기간 미가산 시
                                        #         n_s = to_s(w_yrs, pos, n_opt, n_age, n_time)
                                        #         li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                        # elif w_yrs == 0: # 승진 누락자는 무조건 승진
                                        #     n_s = to_s(self.l_for_promo - 1, n_pos, n_opt, n_age, n_time)
                                        #     li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                        # elif w_yrs > 0:
                                        #     if add_years == True:  # 승진기간 가산 시
                                        #         n_s = to_s(w_yrs - 1, pos, n_opt, n_age, n_time)
                                        #         li.append((1.0, n_s, rew, done))  # n_s는 다음 상태
                                        #     else:  # 승진기간 미가산 시
                                        #         n_s = to_s(w_yrs, pos, n_opt, n_age, n_time)
                                        #         li.append((1.0, n_s, rew, done))  # n_s는 다음 상태

        super(ParentalLeave, self).__init__(nS, nA, P, isd)

    def to_s(self, left, pos, opt, age, time):
        s_idx = 0
        s_idx += time
        s_idx += age * self.max_years
        s_idx += opt * (self.max_years * self.max_age)
        s_idx += left * (self.max_years * self.max_age * self.n_options)
        cumul = np.sum(self.max_yrs_pos[:pos]) if pos != 0 else 0
        s_idx += self.max_years * self.max_age * self.n_options * cumul
        if time == self.max_years:
            s_idx = self.nS - 1 # Terminal state with value zero
        return s_idx

    def get_reward(self, pos, opt, age, time, action):
        if time == self.max_years:
            return -self.annual_cost
        if action == '근무':
            return self.salary(pos) - self.annual_cost
        elif action == '휴직 사용':
            if opt == 0 or age == self.max_age - 1:
                return -self.annual_cost
            else:
                # yr2024
                subsidy = max(70, min(self.salary(pos)/12*0.8, 150))
                annual_subsidy = subsidy * 12
                # # yr2025; 정부 ‘저출생 추세 반전을 위한 대책’ 발표
                # subsidy_first3 = max(70, min(self.salary(pos)/12*1.0, 250))
                # subsidy_4to6 = max(70, min(self.salary(pos)/12 * 1.0, 200))
                # subsidy_after = max(70, min(self.salary(pos)/12 * 0.8, 160))
                # annual_subsidy = subsidy_first3 * 3 + subsidy_4to6 * 3 + subsidy_after * 6
                # return annual_subsidy + self.get_utlity(age) - self.annual_cost
                return max(840.0, min(self.salary(pos)/12*0.8, 2150)) + self.get_utlity(age) - self.annual_cost

    def get_utlity(self, child_age):
        return self.utility

    # 직급별 salary 함수
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

        # # For 5 levels
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
    def get_promo_prob(self, pos, opt):
        # https://donidang.tistory.com/1483
        # 대리-과장: 60~70%
        # 과장-차장: 40~50%
        # 차장-부장: 30~40%
        # p < 1
        if pos < self.n_position - 1:
            q_pos = [1.0, 0.65, 0.45, 0.35]
            return max(0, q_pos[pos] - self.delta * (1-opt))
        else:
            return 1.0

        # # p = 1
        # return 1.0