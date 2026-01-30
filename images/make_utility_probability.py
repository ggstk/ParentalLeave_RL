# import os,pickle,json
# import matplotlib.pyplot as plt
# from MARL import Employee,ParentalLeave,NashQLearning
# import seaborn as sns
# base_dir = "C:/Users/SDOlab/Desktop/ParentalLeave/saved_models/"
#
#
# probs_path1 = os.path.join(base_dir, f"probs_n=10.json")
# probs_path2 = os.path.join(base_dir, f"setting=0.json")
# probs_path3 = os.path.join(base_dir, f"setting=1.json")
# probs_path4 = os.path.join(base_dir, f"setting=2.json")
#
# probs_path5 = os.path.join(base_dir, f"setting=3.json")
# probs_path6 = os.path.join(base_dir, f"setting=4.json")
# probs_path7 = os.path.join(base_dir, f"setting=5.json")
# probs_path8 = os.path.join(base_dir, f"setting=6.json")
#
#
# with open(probs_path1, 'r') as f:
#     probs = json.load(f)
#
# with open(probs_path2, 'r') as f:
#     probs2 = json.load(f)
#
# with open(probs_path3, 'r') as f:
#     probs3 = json.load(f)
#
# with open(probs_path4, 'r') as f:
#     probs4 = json.load(f)
#
# with open(probs_path5, 'r') as f:
#     probs5 = json.load(f)
#
# with open(probs_path6, 'r') as f:
#     probs6 = json.load(f)
#
# with open(probs_path7, 'r') as f:
#     probs7 = json.load(f)
#
# with open(probs_path8, 'r') as f:
#     probs8 = json.load(f)
#
# plt.figure(figsize=(11, 5))
# X = [i for i in range(11)]
# X.extend([15, 20])
# Y = probs
# Y = [round(y * 100, 2) for y in Y]
# sns.set_theme()
# plt.plot(X, Y,marker="o",label="alpha=0.1;delta=0.2(excluding position=0,1)")
#
# # for x, y in zip(X, Y):
# #     plt.text(x, y + 1, f"{y}%", ha='center', va='bottom', fontsize=9)
# X2 = [i for i in range(7)]
# Y2 = probs2
# Y2 = [round(y * 100, 2) for y in Y2]
# plt.plot(X2, Y2,marker="o",label="alpha=0;delta=0(excluding position=0,1)")
# # for x, y in zip(X2, Y2):
# #     plt.text(x, y + 1, f"{y}%", ha='center', va='bottom', fontsize=9)
#
# X3 = [i for i in range(7)]
# Y3 = probs3
# Y3 = [round(y * 100, 2) for y in Y3]
# plt.plot(X3, Y3,marker="o",label="alpha=0.05;delta=0.2(excluding position=0,1)")
# # for x, y in zip(X3, Y3):
# #     plt.text(x, y + 1, f"{y}%", ha='center', va='bottom', fontsize=9)
#
# X4 = [i for i in range(7)]
# Y4 = probs4
# Y4 = [round(y * 100, 2) for y in Y4]
# plt.plot(X4, Y4,marker="o",label="alpha=0.1;delta=0.1(excluding position=0,1)")
# # for x, y in zip(X4, Y4):
# #     plt.text(x, y + 1, f"{y}%", ha='center', va='bottom', fontsize=9)
#
# X5 = [i for i in range(11)]
# X5.extend([15, 20])
# Y5 = probs5
# Y5 = [round(y * 100, 2) for y in Y5]
# plt.plot(X5, Y5,marker="*",label="alpha=0.1;delta=0.2(excluding position=0)")
#
# X6 = [i for i in range(7)]
# Y6 = probs6
# Y6 = [round(y * 100, 2) for y in Y6]
# plt.plot(X6, Y6,marker="*",label="alpha=0;delta=0(excluding position=0)")
#
#
# X7 = [i for i in range(7)]
# Y7 = probs7
# Y7 = [round(y * 100, 2) for y in Y7]
# plt.plot(X7, Y7,marker="*",label="alpha=0.05;delta=0.2(excluding position=0)")
#
# X8 = [i for i in range(7)]
# Y8 = probs8
# Y8 = [round(y * 100, 2) for y in Y8]
# plt.plot(X8, Y8,marker="*",label="alpha=0.1;delta=0.1(excluding position=0)")
#
#
# plt.xticks(range(0, 20, 1))
# plt.legend()
# plt.xlabel("utility(10^4)")
# plt.ylabel("Probability_of_both_Leave(%)")
# plt.title("utility-probability")
# #plt.show()
#
# learning_curve_image_path = os.path.join("C:/Users/SDOlab/Desktop/ParentalLeave/images/",f"utility-probability1.png")
# plt.savefig(learning_curve_image_path)
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_json_data(base_dir, file_templates):
    """
    加载 JSON 文件并返回数据。

    参数:
    - base_dir (str): 基础目录路径。
    - file_templates (list of str): 文件名模板列表。

    返回:
    - list: 每个文件中加载的 JSON 数据。
    """
    data_list = []
    for template in file_templates:
        file_path = os.path.join(base_dir, template)
        try:
            with open(file_path, 'r') as f:
                data_list.append(json.load(f))
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            data_list.append([])
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {file_path}")
            data_list.append([])
    return data_list


# 基础目录和文件模板
base_dir = "C:/Users/SDOlab/Desktop/ParentalLeave/saved_models/"
file_templates = [
    "probs_n=10.json",
    "setting=0.json",
    "setting=1.json",
    "setting=2.json",
    "setting=3.json",
    "setting=4.json",
    "setting=5.json",
    "setting=6.json",
    "setting=7.json",
    "setting=8.json",
    "setting=9.json",
    "setting=10.json",
]

# 加载数据
probs_data = load_json_data(base_dir, file_templates)

# X 和对应标签
x_values = [
    [i for i in range(11)] + [15, 20],  # probs_n=10.json
    *[[i for i in range(7)] for _ in range(3)],  # setting=0 to setting=6
     [i for i in range(11)] + [15, 20],
     *[[i for i in range(7)] for _ in range(3)],
     [i for i in range(11)] +[15, 20],
    *[[i for i in range(7)] for _ in range(3)]


]
labels = [
    "alpha=0.1;delta=0.2(excluding position=0,1)",
    "alpha=0;delta=0(excluding position=0,1)",
    "alpha=0.05;delta=0.2(excluding position=0,1)",
    "alpha=0.1;delta=0.1(excluding position=0,1)",
    "alpha=0.1;delta=0.2(excluding position=0)",
    "alpha=0;delta=0(excluding position=0)",
    "alpha=0.05;delta=0.2(excluding position=0)",
    "alpha=0.1;delta=0.1(excluding position=0)",
    "alpha=0.1;delta=0.2",
    "alpha=0;delta=0",
    "alpha=0.05;delta=0.2",
    "alpha=0.1;delta=0.1",
]
markers = ["o"] * 4 + ["o"] * 4 +["o"] * 4

# 绘图
plt.figure(figsize=(11, 5))
sns.set_theme()

# for X, probs, label, marker in zip(x_values, probs_data, labels, markers):
#     Y = [round(y * 100, 2) for y in probs]
#     plt.plot(X, Y, marker=marker, label=label)

selected_indices = [1, 5, 9]  # 第 2, 5, 9 个数据点的索引（Python 从 0 开始）
X_selected = [x_values[i] for i in selected_indices]
Y_selected = [probs_data[i]  for i in selected_indices]
label_selected =[labels[i]  for i in selected_indices]
for X, probs,label in zip(X_selected, Y_selected,label_selected):
    Y = [round(y * 100, 2) for y in probs]
    plt.plot(X, Y, marker="o", label=label)
#plt.show()
plt.legend()

plt.xticks(range(0, 20, 1))
plt.yticks(range(0, 100, 10))
plt.legend()
plt.xlabel("utility(10^4)")
plt.ylabel("Probability_of_both_Leave(%)")
plt.title("utility-probability")
#plt.show()
learning_curve_image_path = os.path.join("C:/Users/SDOlab/Desktop/ParentalLeave/images/",f"utility-probability.png")
plt.savefig(learning_curve_image_path)