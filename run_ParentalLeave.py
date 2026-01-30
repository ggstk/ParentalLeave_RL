from VIandPI import Value_iteration, Policy_iteration
from ParentalLeave import ParentalLeave # 항상 최신 version을 ParentalLeave로 유지

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':

    # max_yrs_pos =  [6, 6, 6, 8, 8]
    # yrs_for_pos = [4, 4, 4, 5, 5]
    # u_plus = 4000
    # max_years = 25

    # Setting 2024.Nov
    yrs_for_pos =  [3, 4, 4, 5, 5]
    max_yrs_pos = [6, 7, 7, 8, 8]
    position_names = ['Staff', 'Assistant manager', 'Manager', 'Senior manager', 'Director']
    u_plus = 0 # 3300, 4150, 5000
    delta = 0 # 0, 0.1, 0.2
    max_years = 25
    discount_rate = 0.95
    # EXP_NAME = "p{}_U{}_delta{}".format("1", u_plus, delta)
    EXP_NAME = "p{}_U{}_delta{}".format("not1", u_plus, delta)

    env = ParentalLeave(max_yrs_pos = max_yrs_pos,
                        yrs_for_pos = yrs_for_pos,
                        u_plus= u_plus,
                        max_years = max_years,
                        delta = delta)

    # VI
    vi_model = Value_iteration(env, discount_rate = discount_rate)
    policy, value, q_value = vi_model.solve(max_iter=10000)
    print("Policy: ", policy)
    print("Value: ", value)

    # # PI
    # pi_model = Policy_iteration(env, discount_rate = 1.0)
    # policy, value = pi_model.solve(max_iter=10000)
    # print("Policy: ", policy)
    # print("Value: ", value)

    vl = value
    pol = policy
    df_vl = pd.DataFrame(vl)
    df_qval = pd.DataFrame(q_value)
    df_pol = pd.DataFrame(pol)


    ls = []

    for a in range(np.shape(max_yrs_pos)[0]):# pos
        for b in range(max_yrs_pos[a]): # left for promo
            for c in range(2):#opt
                for d in range(10):#age
                    for e in range(max_years):#time
                        ls.append((b,a,c,d,e))#[e,d,c,a,b]
    df = pd.DataFrame({'data': ls})
    df_1 = pd.concat([df, df_vl],axis=1)
    # df_2 = pd.concat([df_1, df_qval], axis=1)
    # df_3 = pd.concat([df_2, df_pol],axis=1)


    df_1.columns =["State",'Value'] #["state",'Value','Q-value_0','Q-value_1','Policy']
    print(df_1)

    df_1.to_excel('results/Parental_Leave_'+EXP_NAME+"BaseValue"+'.xlsx',index=False)
   
   
    # Multiple sheets
    # results = np.zeros((2, 10, max_years, sum(max_yrs_pos))) # [opt][age][work_yrs][position]
    # idx = 0
    # for a in range(np.shape(max_yrs_pos)[0]):# pos
    #     for b in range(max_yrs_pos[a]): # left for promo
    #         for c in range(2):#opt
    #             for d in range(10):#age
    #                 for e in range(max_years):#time
    #                     x_value = 0
    #                     if a != 0:
    #                         x_value += sum(max_yrs_pos[0:a])
    #                     x_value += b
    #                     opt_act = pol[idx]
    #                     if np.isclose(q_value[idx][0], q_value[idx][1]): # 동등할 경우
    #                         opt_act = 1
    #                     results[c][d][e][x_value] = opt_act
    #                     idx += 1

    # position_yrs = []
    # for a in range(np.shape(max_yrs_pos)[0]):# pos
    #     for b in range(max_yrs_pos[a]): # left for promo
    #         # x_labels.append(f"{position_names[a]}, yr{b}")
    #         # x_labels.append(rf"$x={a}$, $y_{{p,x}}$ = {b}")
    #         position_yrs.append(f"{a+1},{b}")

    # with pd.ExcelWriter('results/policy_multiple_sheets_'+EXP_NAME+'.xlsx') as writer:
    #     prev = results[1][0]
    #     for d in range(10):  # age
    #         data = results[1][d]
    #         if d == 0 or not np.array_equal(prev, data):
    #             df = pd.DataFrame(data) # opt = 1
    #             df.to_excel(writer, sheet_name='age{}'.format(d))

    #             # 그림
    #             fig, ax = plt.subplots(figsize=(4,5)) # y-title 넣는 경우 (4,5) 아니면 (3.5,5)
    #             data = np.transpose(data)
    #             im = ax.imshow(data, cmap = 'gray', vmin=0, vmax=1) #cmap
    #             x_labels = np.arange(1, data.shape[1] + 1)
    #             y_labels = np.arange(1, data.shape[0] + 1)
    #             ax.set_xticks(np.arange(data.shape[1]), labels=x_labels, fontsize=6) #
    #             ax.set_yticks(np.arange(data.shape[0]), labels=position_yrs, fontsize=8) #

    #             # plt.title(rf"$q(x,m)$=1, $U^{{+}}$ = {u_plus}, $y_c$={d}", fontsize = 12)
    #             plt.title(rf"$\delta$={delta}, $U^{{+}}$ = {u_plus}, $y_c$={d}", fontsize = 14)

    #             # Set Axis title
    #             ax.set_xlabel(r"Year $t$", fontsize = 12 )
    #             ax.set_ylabel(r"Position $x$, Service years $y_{p}$", fontsize = 12)
    #             ax.xaxis.set_label_position('top')
    #             # # Add legend
    #             # colormap used by imshow
    #             colors = [im.cmap(im.norm(0)), im.cmap(im.norm(1))]
    #             # create a patch (proxy artist) for every color
    #             patches = [mpatches.Patch(facecolor=colors[0], label="Work", edgecolor='black'),
    #                        mpatches.Patch(facecolor=colors[1], label="Leave", edgecolor='black')]
    #             # put those patched as legend-handles into the legend
    #             # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #             plt.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.01), borderaxespad=0., ncol=2)

    #             # Let the horizontal axes labeling appear on top.
    #             ax.tick_params(top=True, bottom=False,
    #                            labeltop=True, labelbottom=False)
    #             # Rotate the tick labels and set their alignment.
    #             plt.setp(ax.get_xticklabels(), rotation=40, ha="left",
    #                      rotation_mode="anchor") # ha는 수평 정렬(right, left, center)
    #             # Turn spines off and create white grid.
    #             ax.spines[:].set_visible(False)
    #             ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    #             ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    #             ax.grid(which="minor", color="grey", linestyle='-', linewidth=1)
    #             ax.tick_params(which="minor", bottom=False, left=False)

    #             fig.tight_layout()
    #             plt.savefig("results/" + f"{EXP_NAME}_child_age_{d}.png")
    #             # plt.show()
    #         prev = results[1][d]

