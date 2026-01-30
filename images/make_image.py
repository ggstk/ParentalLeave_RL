import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 打开并读取 JSON 文件

base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
probs_data_path = os.path.join(base_dir, f"Probs(c1=c2,n=100).json")
errors_data_path= os.path.join(base_dir, f"Errors(c1=c2,n=100).json")
probs_data_path2 = os.path.join(base_dir, f"Probs1(c1<c2,n=100).json")
probs_data_path3 = os.path.join(base_dir, f"Probs2(c1<c2,n=100).json")
errors_data_path2 = os.path.join(base_dir, f"Errors1(c1<c2,n=100).json")
errors_data_path3 = os.path.join(base_dir, f"Errors2(c1<c2,n=100).json")


with open(probs_data_path, 'r') as f:
     probs = json.load(f)
with open(probs_data_path2, 'r') as f:
     probs1 = json.load(f)     
with open(probs_data_path3, 'r') as f:
     probs2 = json.load(f) 
with open(errors_data_path, 'r') as f:
     errors = json.load(f) 
with open(errors_data_path3, 'r') as f:
     errors2 = json.load(f) 
print(errors)







#image1/image2----->C1=C2 
#1-a: delta=0.1일 때 alpha=0/0.05/0.1로 실험해서 alpha 영향 보여준다
#1-b: alpha=0,   delta=0/.1/0.2로 실험해서 delta영향을 보여준다


# # #1-a
labels = [
    "alpha=0;delta=0.1",
    "alpha=0.05;delta=0.1",
    "alpha=0.1;delta=0.1"
] 
# # labels = [
# #     "alpha=0;delta=0",
# #     "alpha=0.05;delta=0",
# #     "alpha=0.1;delta=0"
# # ] 

# #1-b
# # labels = [
# #     "alpha=0;delta=0",
# #     "alpha=0;delta=0.1",
# #     "alpha=0;delta=0.2"
# # ] 


markers = ["o"] * 3
plt.figure(figsize=(11, 5))
sns.set_theme()
X=[0,3300,4150,5000,10000,20000]


ranges = [(6, 12), (18, 24), (24, 30)] #1-a alpha
# ranges = [(0, 6), (6, 12), (12, 18)] #1-b delta
for (m, n), label in zip(ranges, labels):
    Y = [round(y * 100, 2) for y in probs[m:n]]
    plt.plot(X, Y, marker="o", label=label)
    errors1=[round(y * 100, 2) for y in errors[m:n]]
#     plt.errorbar(X, Y, yerr=errors1, fmt='-', capsize=5, label=label)
    line_color = plt.gca().lines[-1].get_color()
    for i, (x, y) in enumerate(zip(X, Y)):
        offset = (-5, 5) if i % 2 == 0 else (5, -5)  
     #    plt.text(x + offset[0], y + offset[1], f'({x},{y})', 
               #   fontsize=10, ha='left', va='top', color=line_color)
        plt.text(x, y, f'{y}%', fontsize=10, ha='right', va='bottom',color=line_color)

plt.yticks(range(0, 100, 10))
plt.legend()
plt.xlabel("utility(10^4)")
plt.ylabel("Probability_of_both_Leave(%)")
plt.title("utility-probability(c1=c2)")
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"image12.png")
# plt.savefig(learning_curve_image_path)
plt.show()

# plt.title("utility-probability(c1=c2;delta)")
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"utility-probability(c1=c2;delta).png")
# plt.savefig(learning_curve_image_path)
# plt.show()














#image3/image4----->C1<C2 
#2-a: delta=0.1일 때 alpha=0/0.05/0.1로 실험해서 alpha 영향이 보여주고
#2-b: 
     #alpha=0,     delta=0/0.1로 실험해서 delta영향을 보여준다
     #alpha=0.05,  delta=0/0.1로 실험해서 delta영향을 보여준다
     #alpha=0.1,   delta=0/0.1로 실험해서 delta영향을 보여준다


#2-a
# labels = [
#     "alpha=0;delta=0",
#     "alpha=0.05;delta=0",
#     "alpha=0.1;delta=0"
# ] 

# labels = [
#     "alpha=0;delta=0.1",
#     "alpha=0.05;delta=0.1",
#     "alpha=0.1;delta=0.1"
# ] 

# labels = [
#     "alpha=0;delta=0",
#      "alpha=0;delta=0.1",
#     "alpha=0.05;delta=0.1",
#     "alpha=0.1;delta=0.1"
# ] 


# #2-a
# X=[1,2,3,4,5]
# # ranges = [(0, 5), (5, 10), (10, 15)]  #2-a-1 delta=0
# # ranges = [(15, 20), (20, 25), (25, 30)] #2-a-2 delta=0.1

# ranges = [(0, 5),(15, 20),(20, 25), (25, 30)]  #2-a-1 delta=0
# # # # markers = ["o"] * 3
# plt.figure(figsize=(11, 5))
# sns.set_theme()

# for (m, n), label in zip(ranges, labels):
#     Y = [round(y * 100, 2) for y in probs2[m:n]]
   
# #     ci_low = Y - 1.96 * np.std(Y) / np.sqrt(len(Y)) #95%
# #     ci_high = Y + 1.96 * np.std(Y) / np.sqrt(len(Y))
# #     plt.fill_between(X, ci_low, ci_high, alpha=0.2)
#     plt.plot(X, Y, marker="o", label=label)
#     # 标注每个点的数值，错开位置
#     line_color = plt.gca().lines[-1].get_color()
#     for i, (x, y) in enumerate(zip(X, Y)):
#         # offset = (-5, 5) if i % 2 == 0 else (5, -5)  # 交替偏移
#         # plt.text(x + offset[0], y + offset[1], f'({x},{y})', 
#         #          fontsize=10, ha='left', va='top', color=line_color)
#         plt.text(x, y, f'{y}%', fontsize=10, ha='right', va='bottom',color=line_color)

# plt.xticks(range(1, 6, 1),labels=["c1=0,c2=3300","c1=0,c2=4150","c1=0,c2=5000","c1=3300,c2=5000","c1=3300,c2=10000"])
# plt.yticks(range(0, 100, 10))
# plt.legend()
# plt.xlabel("utility(10^4)")
# plt.ylabel("Probability_of_agent2_Leave(%)")
# plt.title("utility-probability(c1<c2;alpha+delta)")
# # plt.show()
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"utility-probability(alpha+delta).png")
# plt.savefig(learning_curve_image_path)


#2-b

# labels = [
#     "alpha=0;delta=0",    
#     "alpha=0;delta=0.1"] 

# labels = [
#     "alpha=0.05;delta=0",    
#     "alpha=0.05;delta=0.1"] 

# labels = [
#     "alpha=0.1;delta=0",    
#     "alpha=0.1;delta=0.1"] 

# # ranges = [(0, 5), (15, 20)]  #alpha=0,   delta=0/0.1로 실험해서 delta영향을 보여준다
# # ranges = [(5, 10), (20, 25)] #alpha=0.05,   delta=0/0.1로 실험해서 delta영향을 보여준다
# ranges = [(10, 15), (25, 30)] #alpha=0.1,   delta=0/0.1로 실험해서 delta영향을 보여준다

# plt.figure(figsize=(11, 5))
# sns.set_theme()
# X=[1,2,3,4,5]
# for (m, n), label in zip(ranges, labels):
#     Y = [round(y * 100, 2) for y in probs2[m:n]]
#     errors1 = [round(y * 100, 2) for y in errors2[m:n]]
#     plt.errorbar(X, Y, yerr=errors1, fmt='-', capsize=3, label=label,elinewidth=0.5)

# #     plt.plot(X, Y, marker="o", label=label)
#     line_color = plt.gca().lines[-1].get_color()

#     # 标注每个点的数值，错开位置
#     for i, (x, y) in enumerate(zip(X, Y)):
#         # offset = (-5, 5) if i % 2 == 0 else (5, -5)  # 交替偏移
#         # plt.text(x + offset[0], y + offset[1], f'({x},{y})', 
#         #          fontsize=10, ha='left', va='top', color=line_color)
#         plt.text(x, y, f'{y}%', fontsize=10, ha='right', va='bottom',color=line_color)

# plt.xticks(range(1, 6, 1),labels=["c1=0,c2=3300","c1=0,c2=4150","c1=0,c2=5000","c1=3300,c2=5000","c1=3300,c2=10000"])
# plt.yticks(range(0, 100, 10))
# plt.legend()
# plt.xlabel("utility(10^4)")
# plt.ylabel("Probability_of_agent2_Leave(%)")
# plt.title("utility-probability(c1<c2)")
# # plt.show()
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"image16.png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
# plt.savefig(learning_curve_image_path)


