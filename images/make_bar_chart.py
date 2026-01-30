import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 打开并读取 JSON 文件

base_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/saved_models/"
probs_data_path = os.path.join(base_dir, f"Probs(c1=c2,n=100).json")
errors_data_path = os.path.join(base_dir, f"Errors(c1=c2,n=100).json")

probs_data_path2 = os.path.join(base_dir, f"Probs1(c1<c2,n=100).json")
probs_data_path3 = os.path.join(base_dir, f"Probs2(c1<c2,n=100).json")
errors_data_path2 = os.path.join(base_dir, f"Errors1(c1<c2,n=100).json")
errors_data_path3 = os.path.join(base_dir, f"Errors2(c1<c2,n=100).json")

probs_data_path4 = os.path.join(base_dir, f"Probs1(c1=c2,n=100).json")
probs_data_path5 = os.path.join(base_dir, f"Probs2(c1=c2,n=100).json")
errors_data_path4 = os.path.join(base_dir, f"Errors1(c1=c2,n=100).json")
errors_data_path5 = os.path.join(base_dir, f"Errors2(c1=c2,n=100).json")

probs_data_path6 = os.path.join(base_dir, f"Probs1(exp60-65,n=100).json")
probs_data_path7 = os.path.join(base_dir, f"Probs2(exp60-65,n=100).json")
errors_data_path6 = os.path.join(base_dir, f"Errors1(exp60-65,n=100).json")
errors_data_path7 = os.path.join(base_dir, f"Errors2(exp60-65,n=100).json")




with open(probs_data_path, 'r') as f:
     probs = json.load(f)
with open(probs_data_path2, 'r') as f:
     probs1 = json.load(f)     
with open(probs_data_path3, 'r') as f:
     probs2 = json.load(f) 
with open(errors_data_path, 'r') as f:
     errors = json.load(f)


with open(errors_data_path2, 'r') as f:
     errors1 = json.load(f) 
with open(errors_data_path3, 'r') as f:
     errors2 = json.load(f) 

with open(probs_data_path4, 'r') as f:
     probs3 = json.load(f)     
with open(probs_data_path5, 'r') as f:
     probs4 = json.load(f) 

with open(errors_data_path4, 'r') as f:
     errors3 = json.load(f) 
with open(errors_data_path5, 'r') as f:
     errors4 = json.load(f) 


with open(probs_data_path6, 'r') as f:
     probs5 = json.load(f)     
with open(probs_data_path7, 'r') as f:
     probs6 = json.load(f) 
with open(errors_data_path6, 'r') as f:
     errors5 = json.load(f) 
with open(errors_data_path7, 'r') as f:
     errors6 = json.load(f)



# c1=c2
# labels = [
#     "alpha=0;delta=0.1",
#     "alpha=0.05;delta=0.1",
#     "alpha=0.1;delta=0.1"
# ] 

# labels = [
#     "alpha=0;delta=0",
#     "alpha=0;delta=0.1",
#     "alpha=0;delta=0.2"
# ] 

# X=[1,5,10]
# plt.figure(figsize=(5, 5))
# sns.set_theme()

# ranges = [[8,11],[20,23],[26,29]] #1-a alpha
# # ranges = [[9,11],[21,23],[27,29]]
# # ranges = [(0, 6), (6, 12), (12, 18)] #1-b delta

# bar_width = 1  # 柱状图宽度
# num_bars = len(ranges)  # 需要绘制的柱状图数量
# offsets = np.linspace(-bar_width, bar_width, num_bars)  # 计算偏移量

# line_x = []
# line_y = []

# for (m, n), label,offset in zip(ranges, labels,offsets):
#     Y = [round(y * 100, 2) for y in probs[m:n]]
#     error = [round(y * 100, 2) for y in errors[m:n]]
# #     plt.bar(X + offset, Y, width=bar_width, label=label)  # X 位置加上偏移量
#     plt.bar(X + offset, Y, yerr=error,capsize=2,width=bar_width, label=label,error_kw={'elinewidth': 0.5})  # X 位置加上偏移量
#     bar_x = X + offset  # 柱子的中心点
#     line_x.extend(bar_x)
#     line_y.extend(Y)
    
# plt.yticks(range(0, 100, 10))
# plt.xticks(X, [4150,5000,10000])
# plt.legend()
# i=0
# # for x,y in zip(line_x,line_y):
# #     # m=0+i*2
# #     # n=2+i*2
# #     # plt.plot(line_x[m:n],line_y[m:n], linestyle='-')
# #     # i += 1
# #     # line_color = plt.gca().lines[-1].get_color()
# #     for x, y in zip(line_x[m:n], line_y[m:n]):
# #         plt.text(x, y + 1, f"{y:.2f}%", ha='center', va='bottom',color=line_color,fontsize=8)
# #     #plt.plot(line_x, line_y, marker='o', linestyle='-', color='black', label="Trend Line")

# plt.xlabel("utility(10^4)")
# plt.ylabel("Probability_of_both_Leave(%)")
# plt.title("utility-probability(c1=c2)")
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"image17.png")
# plt.savefig(learning_curve_image_path)
# plt.show()
# plt.title("utility-probability(c1=c2;delta)")
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"utility-probability(c1=c2;delta).png")
# plt.savefig(learning_curve_image_path)




#c1<c2---->image1+image2

labels = [
    # "c1=0,c2=3300",
    # "c1=0,c2=4150",
    # "c1=0,c2=5000",
    # "c1=3300,c2=5000",
    # "c1=3300,c2=10000"
    
    "alpha=0;delta=0.1",
    "alpha=0.05;delta=0.1",
    "alpha=0.1;delta=0.1"
    
#     "alpha=0;delta=0",
#     "alpha=0.05;delta=0",
#     "alpha=0.1,delta=0"
] 

X=[1,15,30]
plt.figure(figsize=(5, 5))
sns.set_theme()

# ranges = [[0,1,2],[5,6,7],[10,11,12]] 
ranges = [[15,16,17],[20,21,22],[25,26,27]] 
# ranges = [[15,20,25],[16,21,26],[17,22,27],[18,23,28],[19,24,29]] 

bar_width = 2 # 柱状图宽度
num_bars = len(ranges)  # 需要绘制的柱状图数量
#offsets = np.linspace(-bar_width, bar_width, num_bars)  # 计算偏移量
offsets = np.linspace(-bar_width * (num_bars // 2), bar_width * (num_bars // 2), num_bars)  # 计算偏移量

line_x = []
line_y = []

for (m, n,l), label,offset in zip(ranges, labels,offsets):
    Y = [round(y * 100, 2) for y in [probs2[m],probs2[n],probs2[l]]]
    errors=np.array([round(y * 100, 2) for y in [errors2[m],errors2[n],errors2[l]]])
    plt.bar(X + offset, Y, yerr=errors,capsize=2,width=bar_width, label=label,error_kw={'elinewidth': 0.5})  # X 位置加上偏移量
    bar_x = X + offset  # 柱子的中心点
    line_x.extend(bar_x)
    line_y.extend(Y)
    
plt.yticks(range(0, 100, 10))
# plt.xticks(X, ["α=0,δ=0","α=0.05,δ=0","α=0.1,δ=0"])
# plt.xticks(X, ["α=0,δ=0.1","α=0.05,δ=0.1","α=0.1,δ=0.1"])
plt.xticks(X, ["c1=0,c2=3300","c1=0,c2=4150","c1=0,c2=5000"])

plt.legend(fontsize=8)
# print(line_x,line_y)
i=0
for x,y in zip(line_x,line_y):
    m=0+i*3
    n=1+i*3
    l=2+i*3
    #plt.plot([line_x[m],line_x[n],line_x[l]],[line_y[m],line_y[n],line_y[l]], linestyle='-')
    i += 1
    #line_color = plt.gca().lines[-1].get_color()
    # for x, y in zip([line_x[m],line_x[n],line_x[l]],[line_y[m],line_y[n],line_y[l]]):
    #     plt.text(x, y + 1, f"{y:.2f}%", ha='center', va='bottom',fontsize=7)
    if l==14:
        break
    #plt.plot(line_x, line_y, marker='o', linestyle='-', color='black', label="Trend Line")

plt.xlabel("utility(10^4)")
plt.ylabel("Probability_of_agent2_Leave(%)")
plt.title("utility-probability(c1<c2)")
learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"image2.png")
plt.savefig(learning_curve_image_path)
# plt.show()
# # plt.title("utility-probability(c1=c2;delta)")
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"utility-probability(bar2).png")
# plt.savefig(learning_curve_image_path)


# a) c1=0, c2= 3,300 / 4,150 / 5,000

#image3+image4+image5+image6+image7+image8
# b) c1=3,300, c2 = 3,300 / 5,000 / 10,000

# labels = [
#     "alpha=0;delta=0.1",
#     "alpha=0.05;delta=0.1",
#     "alpha=0.1;delta=0.1"
# ] 

# X=[1,15,30]
# plt.figure(figsize=(5, 5))
# sns.set_theme()

# ranges = [[19,3,10],[24,4,22],[29,5,28]] 
# # ranges = [[18,9,3],[23,21,4],[28,27,5]] 
# # ranges = [[7,18,19],[19,23,24],[25,28,29]] 

# bar_width = 2 # 柱状图宽度
# num_bars = len(ranges)  # 需要绘制的柱状图数量
# #offsets = np.linspace(-bar_width, bar_width, num_bars)  # 计算偏移量
# offsets = np.linspace(-bar_width * (num_bars // 2), bar_width * (num_bars // 2), num_bars)  # 计算偏移量

# line_x = []
# line_y = []

# for (m, n,l), label,offset in zip(ranges, labels,offsets):
#     Y = [round(y * 100, 2) for y in [probs1[m],probs6[n],probs3[l]]]
#     errors=np.array([round(y * 100, 2) for y in [errors1[m],errors6[n],errors3[l]]])
#     plt.bar(X + offset, Y, yerr=errors,capsize=2,width=bar_width, label=label,error_kw={'elinewidth': 0.5})  # X 位置加上偏移量
#     bar_x = X + offset  # 柱子的中心点
#     line_x.extend(bar_x)
#     line_y.extend(Y)
    
# plt.yticks(range(0, 100, 10))
# # plt.xticks(X, ["c1=0,c2=3300","c1=0,c2=4150","c1=0,c2=5000"])
# # plt.xticks(X, ["c1=3300,c2=3300","c1=3300,c2=5000","c1=3300,c2=10000"],fontsize=10)
# plt.xticks(X, ["c1=3300,c2=10000","c1=5000,c2=10000","c1=10000,c2=10000"],fontsize=9)

# plt.legend(fontsize=7)
# print(line_x,line_y)
# i=0

# plt.xlabel("utility(10^4)")
# plt.ylabel("Probability_of_agent1_Leave(%)")
# # plt.title("utility-probability(c1=3300)")
# plt.title("utility-probability(c2=10000)")
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"image7.png")
# plt.savefig(learning_curve_image_path)
# # plt.show()







#image9+image10

# labels = [
#     "c1=3300;c2=3300",
#     "c1=3300;c2=5000",
#     "c1=3300;c2=10000"
# ] 

# labels = [
#     "c1=3300;c2=5000",
#     "c1=5000;c2=5000",
#     "c1=10000;c2=5000"
# ] 

# labels = [
#     "c1=3300;c2=10000",
#     "c1=5000;c2=10000",
#     "c1=10000;c2=10000"
# ] 


# X=[1,15,30]
# plt.figure(figsize=(5, 5))
# sns.set_theme()


# # ranges = [[18,23,28],[9,21,27],[3,4,5]] 
# # ranges = [[7,19,25],[18,23,28],[19,24,29]] 
# ranges = [[19,24,29],[3,4,5],[10,22,28]] 

# # ranges=[[7,19,25]]
# bar_width = 2 # 柱状图宽度
# num_bars = len(ranges)  # 需要绘制的柱状图数量
# #offsets = np.linspace(-bar_width, bar_width, num_bars)  # 计算偏移量
# offsets = np.linspace(-bar_width * (num_bars // 2), bar_width * (num_bars // 2), num_bars)  # 计算偏移量

# line_x = []
# line_y = []

# pset=[probs2,probs5,probs4]
# eset=[errors2,errors5,errors4]
# for (m, n,l), label,offset,p,e in zip(ranges, labels,offsets,pset,eset):    
#     Y = [round(y * 100, 2) for y in [p[m],p[n],p[l]]]
#     errors=np.array([round(y * 100, 2) for y in [e[m],e[n],e[l]]])
#     plt.bar(X + offset, Y, yerr=errors,capsize=2,width=bar_width, label=label,error_kw={'elinewidth': 0.5})  # X 位置加上偏移量
#     bar_x = X + offset  # 柱子的中心点
#     line_x.extend(bar_x)
#     line_y.extend(Y)
    

# plt.yticks(range(0, 100, 10))
# # plt.xticks(X, ["c1=0,c2=3300","c1=0,c2=4150","c1=0,c2=5000"])
# # plt.xticks(X, ["c1=3300,c2=3300","c1=3300,c2=5000","c1=3300,c2=10000"],fontsize=10)
# plt.xticks(X, ["alpha=0;delta=0.1","alpha=0.05;delta=0.1","alpha=0.1;delta=0.1"],fontsize=9)

# plt.legend(fontsize=7)
# print(line_x,line_y)
# i=0

# plt.xlabel("utility(10^4)")
# plt.ylabel("Probability_of_agent2_Leave(%)")
# plt.title("utility-probability(c2=10000)")
# # plt.title("utility-probability(c2=10000)")
# learning_curve_image_path = os.path.join("/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/",f"image11.png")
# plt.savefig(learning_curve_image_path)
# # plt.show()

