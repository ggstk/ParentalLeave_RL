# from MARL import Employee,ParentalLeave,NashQLearning
# import pickle,os,json
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import pandas as pd
# import matplotlib.patches as mpatches

# employee0 = Employee((0, 2, 1, 0, 0)) #首先职位是4是没有升职的因素影响的
# employee1 = Employee((0, 2, 1, 0, 0))#(w_yrs,pos,opt,age,time) (0,0,1,0,0)

# base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models"
# q0_path = os.path.join(base_dir, f"Q0_exp13.pickle")
# q1_path = os.path.join(base_dir, f"Q1_exp13.pickle")
# #从文件中加载 Q0,Q1
# with open(q0_path, 'rb') as f:
#     Q0_loaded = pickle.load(f)
# with open(q1_path, 'rb') as f:
#     Q1_loaded = pickle.load(f)




# if __name__ == "__main__":
#     experiment_num=74
#     alpha,delta=1,0
#     c1,c2 = 0,3300
#     #494:q值是否同时更新
#     # Create an instance of the ParentalLeave environment
#     env = ParentalLeave(employees=[employee0, employee1], c1=c1, c2=c2, alpha=alpha, delta=delta)
    
#     # Create an instance of the NashQLearning agent           #35*10**5
#     start_time = datetime.now()    #nash_q_agent = NashQLearning(experiment_num=0, env=env, discount_factor=0.99, learning_rate=1, max_iter=5*10**6)
#     nash_q_agent = NashQLearning(experiment_num=experiment_num,env=env, discount_factor=0.99,learning_rate=0.5,max_iter = 2*10**6,
#                                 q_table0=Q0_loaded,q_table1=Q1_loaded)#
#     # Train the agent and get the Q-tables
#     #q_table_employee0, q_table_employee1 = nash_q_agent.fit()
#     nash_q_agent.reset_states_policy()
#     end_time = datetime.now()
#     elapsed_time = end_time - start_time  # 计算运行时间
#     print(f"Running time:{elapsed_time}")

#     base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models"
#     max_yrs_pos = [6, 6, 6, 8, 8]
#     yrs_for_pos = [4, 4, 4, 5, 5]
#     policy_num = 1
#     # Multiple sheets
#     policy1_path = os.path.join(base_dir, f"policy1_exp{experiment_num}.json")

#     # 从文件中加载 Q0,Q1
#     with open(policy1_path, 'rb') as file:
#         policy1 = json.load(file)

#     position_yrs = []

#     for a in range(np.shape(max_yrs_pos)[0] - 1):  # pos
#         for b in range(max_yrs_pos[a]):  # left for promo
#             # x_labels.append(f"{position_names[a]}, yr{b}")
#             # x_labels.append(rf"$x={a}$, $y_{{p,x}}$ = {b}")
#             position_yrs.append(f"{a + 1},{b}")
#     #print(len(position_yrs))

#     policies1 = [item['policy'] for item in policy1]#1:同时leave
   
#     # print(policies)
#     rows, cols = 9, len(position_yrs)
#     reshaped_data = [policies1[i::rows] for i in range(rows)]
#     df = pd.DataFrame(reshaped_data).transpose()  # 矩阵转置
#     fig, ax = plt.subplots(figsize=(4, 5))  # y-title 넣는 경우 (4,5) 아니면 (3.5,5)
#     # data = np.transpose(policies)
#     data = df
#     im = ax.imshow(data, cmap='gray', vmin=0, vmax=1)  # cmap
#     x_labels = np.arange(1, data.shape[1] + 1)
#     y_labels = np.arange(1, data.shape[0] + 1)
#     ax.set_xticks(np.arange(data.shape[1]), labels=x_labels, fontsize=6)  #
#     ax.set_yticks(np.arange(data.shape[0]), labels=position_yrs, fontsize=8)  #

#     #Policy1：두 직원의 opt은 (1,1);      Policy2：두 직원의 opt은 (1,0);    Policy3：두 직원의 opt은 (0,1)
#     plt.title(rf"$Policy$={policy_num},$\delta$={delta}, $\alpha$={alpha},$c1$ = {c1}, $c2$ = {c2}",fontsize=12)
#     # plt.title(rf"$\delta$={delta}, $U^{{+}}$ = {u_plus}, $y_c$={d}", fontsize = 14)

#     # Set Axis title
#     ax.set_xlabel(r"Age $t$", fontsize=12)
#     ax.set_ylabel(r"Position $x$, Service years $y_{p}$", fontsize=12)
#     ax.xaxis.set_label_position('top')
#     # # Add legend
#     # colormap used by imshow
#     colors = [im.cmap(im.norm(0)), im.cmap(im.norm(1))]
#     # create a patch (proxy artist) for every color
#     patches = [mpatches.Patch(facecolor=colors[0], label="Work", edgecolor='black'),
#                mpatches.Patch(facecolor=colors[1], label="Leave", edgecolor='black')]
#     # put those patched as legend-handles into the legend
#     # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.01), borderaxespad=0., ncol=2)

#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=True, bottom=False,
#                    labeltop=True, labelbottom=False)
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=40, ha="left",
#              rotation_mode="anchor")  # ha는 수평 정렬(right, left, center)
#     # Turn spines off and create white grid.
#     ax.spines[:].set_visible(False)
#     ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
#     ax.grid(which="minor", color="grey", linestyle='-', linewidth=1)
#     ax.tick_params(which="minor", bottom=False, left=False)

#     fig.tight_layout()
#     plt.savefig("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/results/" + f"exp{experiment_num}_policy{policy_num}.png")


import matplotlib.pyplot as plt
import os,json
import seaborn as sns
base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
probs_data_path = os.path.join(base_dir, f"Probs(c1=c2).json")
probs_data_path2 = os.path.join(base_dir, f"Probs2(c1<c2).json")

###1-a/b
with open(probs_data_path, 'r') as f:
     probs = json.load(f)

###2-1/b
with open(probs_data_path2, 'r') as f:
     probs2 = json.load(f)

#Make a plot
#1-a
# labels = [
#     "alpha=0;delta=0.1",
#     "alpha=0.05;delta=0.1",
#     "alpha=0.1;delta=0.1"
# ] 

#1-b
# labels = [
#     "alpha=0;delta=0",
#     "alpha=0;delta=0.1",
#     "alpha=0;delta=0.2"
# ] 

#2-a-1
# labels = [
#     "alpha=0;delta=0.1",
#     "alpha=0.05;delta=0.1",
#     "alpha=0.1;delta=0.1"
# ] 

#2-a-2
labels = [
    #"alpha=0;delta=0",
    "alpha=0.05;delta=0",
    #"alpha=0.1;delta=0",
    # "alpha=0;delta=0.1",
    "alpha=0.05;delta=0.1",
    #"alpha=0.1;delta=0.1"
] 

#2-b
# labels = [
#     "alpha=0;delta=0",
#     "alpha=0;delta=0.1",
# ] 


markers = ["o"] * 3
plt.figure(figsize=(11, 5))
sns.set_theme()
# #X=[0,3300,4150,5000,10000,20000]
# X=[3300,4150,5000,5000,10000]

# #for prob,label in zip(probs,labels):

# #ranges = [(6, 12), (18, 24), (24, 30)] #1-a alpha
# #ranges = [(0, 6), (6, 12), (12, 18)] #1-b delta
# #ranges = [(45, 50), (50, 55), (55, 60)] 
# #ranges = [(45, 50), (50, 55), (55, 60)] 
# for (m, n), label in zip(ranges, labels):
#     Y = [round(y * 100, 2) for y in probs[m:n]]
#     plt.plot(X, Y, marker="o", label=label)
#     line_color = plt.gca().lines[-1].get_color()

#     # 标注每个点的数值，错开位置
#     for i, (x, y) in enumerate(zip(X, Y)):
#         # offset = (-5, 5) if i % 2 == 0 else (5, -5)  # 交替偏移
#         # plt.text(x + offset[0], y + offset[1], f'({x},{y})', 
#         #          fontsize=10, ha='left', va='top', color=line_color)
#         plt.text(x, y, f'{y}%', fontsize=10, ha='right', va='bottom',color=line_color)

# # plt.xticks(range(0, 20, 1))
# plt.yticks(range(0, 100, 10))
# plt.legend()
# plt.xlabel("utility(10^4)")
# plt.ylabel("Probability_of_both_Leave(%)")
# plt.title("utility-probability(c1<c2;delta)")
# plt.show()
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"utility-probability(c1=c2;delta).png")
# plt.savefig(learning_curve_image_path)





#2-a
X=[1,2,3,4,5]
#ranges = [(0, 5), (5, 10), (10, 15)] 
#ranges = [(15, 20), (20, 25), (25, 30)] 
#ranges = [(0, 5), (10, 15)] 
ranges = [(5, 10), (20, 25)] 
#ranges = [(15, 20), (25, 30)] 
for (m, n), label in zip(ranges, labels):
    Y = [round(y * 100, 2) for y in probs2[m:n]]
    print(Y)
    plt.plot(X, Y, marker="o", label=label)
    line_color = plt.gca().lines[-1].get_color()

    # 标注每个点的数值，错开位置
    for i, (x, y) in enumerate(zip(X, Y)):
        # offset = (-5, 5) if i % 2 == 0 else (5, -5)  # 交替偏移
        # plt.text(x + offset[0], y + offset[1], f'({x},{y})', 
        #          fontsize=10, ha='left', va='top', color=line_color)
        plt.text(x, y, f'{y}%', fontsize=10, ha='right', va='bottom',color=line_color)

plt.xticks(range(1, 6, 1),labels=["c1=0,c2=3300","c1=0,c2=4150","c1=0,c2=5000","c1=3300,c2=5000","c1=3300,c2=10000"])
plt.yticks(range(0, 100, 10))
plt.legend()
plt.xlabel("utility(10^4)")
plt.ylabel("Probability_of_agent2_Leave(%)")
plt.title("utility-probability(c1<c2;delta)")
plt.show()
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"utility-probability(c1<c2;delta).png")
# plt.savefig(learning_curve_image_path)

