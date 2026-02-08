import pickle,os,json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import matplotlib.patches as mpatches

results_dir="/home/zhaolixue/ZHAOLIXUE/ParentalLeave/results/"
image_dir = "/home/zhaolixue/ZHAOLIXUE/ParentalLeave/images/"
#Load probability and error
#C1=C2
probs_dir_1=os.path.join(results_dir, f"Probs1(0-42).json")
error_dir_1=os.path.join(results_dir, f"Probs1(0-42).json")
probs_dir_2=os.path.join(results_dir, f"Probs2(0-42).json")
error_dir_2=os.path.join(results_dir, f"Probs2(0-42).json")

#c1 != c2
probs_dir_3=os.path.join(results_dir, f"Probs1(42-78).json")
error_dir_3=os.path.join(results_dir, f"Probs1(42-78).json")
probs_dir_4=os.path.join(results_dir, f"Probs2(42-78).json")
error_dir_4=os.path.join(results_dir, f"Probs2(42-78).json")

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
#c1=c2
probs1=load_data(probs_dir_1)
errors1=load_data(error_dir_1)
probs2=load_data(probs_dir_2)
errors2=load_data(error_dir_2)

#c1!=c2
probs3=load_data(probs_dir_3)
errors3=load_data(error_dir_3)
probs4=load_data(probs_dir_4)
errors4=load_data(error_dir_4)


#c1=c2
arry1=(np.array(probs1).reshape(7,6))*100
arry2=(np.array(probs2).reshape(7,6))*100
df1 = pd.DataFrame(arry1)
df2 = pd.DataFrame(arry2)
errors1=(np.array(errors1).reshape(7,6))*100
errors2=(np.array(errors2).reshape(7,6))*100
er1=pd.DataFrame(errors1)
er2=pd.DataFrame(errors2)

#c1 != c2
arry3=(np.array(probs3).reshape(6,6))*100
arry4=(np.array(probs4).reshape(6,6))*100
df3 = pd.DataFrame(arry3)
df4 = pd.DataFrame(arry4)
errors3=(np.array(errors3).reshape(6,6))*100
errors4=(np.array(errors4).reshape(6,6))*100
er3=pd.DataFrame(errors3)
er4=pd.DataFrame(errors4)


#Make a plot

#(1)image1：
def image1():
    labels1 = [
    "α=0; δ=0",
    "α=0; δ=0.1",
    "α=0; δ=0.2"]   #0-5,6-11,12-17
    
    X=[0,33,41.5,50,100,200]
    print(df2)
    df_selected = df2.iloc[[0,1,2]]
    print(df_selected)
    df_T = df_selected.T.copy()#df1.T.copy()
    df_T.index = X
    df_T.columns = labels1
    
    df_T.plot(marker="o",figsize=(6, 4))
    plt.xlabel(r"Utility $U^+_1=U^+_2(KRW,million)$",fontsize=10)
    plt.ylabel("Parental leave probability of Agent 2 (%)",fontsize=10)
    # plt.title("Utility-Probability",fontsize=11)
    plt.yticks(range(0,120,20),fontsize=10)
    plt.tight_layout()
    learning_curve_image_path = os.path.join(image_dir,f"image1.png")
    plt.savefig(learning_curve_image_path)
    plt.show()
    
   



def image2():
    labels2 = [
    "α=0; δ=0",
    "α=0.05; δ=0",
    "α=0.1; δ=0"]
    # 1-c
    labels3 = [
    "α=0; δ=0.1",
    "α=0.05; δ=0.1",
    "α=0.1; δ=0.1"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 2行1列，总尺寸设定
    X=[0,33,41.5,50,100,200]
    
    # image2_a
    df_selected1 = df2.iloc[[0, 3, 4]]
    df_T1 = df_selected1.T.copy()#df1.T.copy()
    df_T1.index = X
    df_T1.columns = labels2
    df_T1.plot(ax=axes[0], marker="o")
    # axes[0].set_title("Utility-Probability", fontsize=11)
    axes[0].set_xlabel(r"Utility $U^+_1 = U^+_2 (\mathrm{KRW}, million)$", fontsize=10)
    axes[0].set_ylabel("Parental leave probability of Agent 2 (%)")
    
    # image2_b
    df_selected2 = df2.iloc[[1, 5, 6]]
    df_T2 = df_selected2.T.copy()#df1.T.copy()
    df_T2.index = X
    df_T2.columns = labels3
    df_T2.plot(ax=axes[1], marker="o")
    # axes[1].set_title("Utility-Probability", fontsize=10)
    axes[1].set_xlabel(r"Utility $U^+_1 = U^+_2 (\mathrm{KRW}, million)$", fontsize=10)
    plt.yticks(range(0,120,20),fontsize=10)
    plt.tight_layout() 
    learning_curve_image_path = os.path.join(image_dir,f"image2.png")
    plt.savefig(learning_curve_image_path)
    plt.show()
        
    


def image3():
    #image3
    # delta=0일 때 alpha=0/0.05/0.1/     delta=0.1일 때 alpha=0/0.05/0.1
    # 0 vs 3,300 / 5000 / 10,000/
    # 3,300 vs 0 / 3,300 / 5,000 / 10,000
    # 5,000 vs 0 / 3,300 / 5,000 / 10,000
    
    X=[0,5,10,20]
    # (1)0 vs 3,300 / 5000/ 10,000/
    df_selected = df4.iloc[0:3, 0:3]
    er_selected = er4.iloc[0:3, 0:3]
    df_selected.columns = [r"$U^+_1=0,U^+_2=33$",r"$U^+_1=0,U^+_2=50$",r"$U^+_1=0,U^+_2=100$"]
    df_selected.index= [
        "α=0; δ=0",
        "α=0.05; δ=0",
        "α=0.1; δ=0"
    ]
    er_selected.columns = [r"$U^+_1=0,U^+_2=33$",r"$U^+_1=0,U^+_2=50$",r"$U^+_1=0,U^+_2=100$"]
    er_selected.index= [
        "α=0; δ=0",
        "α=0.05; δ=0",
        "α=0.1; δ=0"
    ]
    
    df_selected.T.plot(kind="bar",yerr=er_selected.T, capsize=3)
    plt.xlabel(r"$Utility (KRW,million)$",fontsize=10)
    plt.ylabel("Parental leave probability of Agent 2 (%)",fontsize=10)
    # plt.title("Utility-Probability",fontsize=11)
    plt.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.yticks(range(0,120,20),fontsize=10)
    plt.tight_layout()
    learning_curve_image_path = os.path.join(image_dir,f"image3.png")
    plt.savefig(learning_curve_image_path)
    plt.show()

def image4():
    #(2) 3,300 vs 0 / 3,300 / 5,000 / 10,000
    df_selected = pd.concat([df4.iloc[0:3, 0],df2.iloc[[0, 3, 4], 1].reset_index(drop=True)], axis=1)
    df_selected = pd.concat([df_selected,df3.iloc[0:3, 3:5]],axis=1)
    er_selected = pd.concat([er4.iloc[0:3, 0],er2.iloc[[0, 3, 4], 1].reset_index(drop=True)], axis=1)
    er_selected = pd.concat([er_selected,er3.iloc[0:3, 3:5]],axis=1)
    # print(df_selected)
    # print(er_selected)
    # df_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=33$",
    #     r"$U^+_1=33$" + "\n" + r"$U^+_2=33$",
    #     r"$U^+_1=50$" + "\n" + r"$U^+_2=33$",
    #     r"$U^+_1=100$" + "\n" + r"$U^+_2=33$"
    # ]
    
    
    df_selected.index= [
        "α=0; δ=0",
        "α=0.05; δ=0",
        "α=0.1; δ=0"
    ]
    er_selected.columns = [r"$U^+_1=0,U^+_2=33$",r"$U^+_1=33,U^+_2=33$",r"$U^+_1=50,U^+_2=33$",r"$U^+_1=100,U^+_2=33$"]
    # er_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=33$",
    #     r"$U^+_1=33$" + "\n" + r"$U^+_2=33$",
    #     r"$U^+_1=50$" + "\n" + r"$U^+_2=33$",
    #     r"$U^+_1=100$" + "\n" + r"$U^+_2=33$"
    # ]
    er_selected.index= [
        "α=0; δ=0",
        "α=0.05; δ=0",
        "α=0.1; δ=0"
    ]
    # print(df_selected)
    df_selected.T.plot(kind="bar",yerr=er_selected.T, capsize=3)
    plt.xlabel(r"$Utility (KRW,million)$",fontsize=10)
    plt.ylabel("Parental leave probability of Agent 2 (%)",fontsize=10)
    # plt.title("Utility-Probability",fontsize=11)
    plt.xticks(fontsize=10)
    plt.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.yticks(range(0,120,20),fontsize=10)
    plt.tight_layout()
    learning_curve_image_path = os.path.join("C:/Users/joyss/Desktop/ParentalLeave/images/",f"image4.png")
    plt.savefig(learning_curve_image_path)
    plt.show()

def image5():
    # (3)5,000 vs 0 / 3,300 / 5,000 / 10,000
    df_selected = pd.concat([df4.iloc[0:3, [1,3]],df2.iloc[[0, 3, 4], 3].reset_index(drop=True)], axis=1)
    df_selected = pd.concat([df_selected,df4.iloc[0:3, 5]],axis=1)
    er_selected = pd.concat([er4.iloc[0:3, [1,3]],er2.iloc[[0, 3, 4], 3].reset_index(drop=True)], axis=1)
    er_selected = pd.concat([er_selected,er4.iloc[0:3, 5]],axis=1)
    # print(df_selected)
    # df_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=3300$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=5000$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=10000$" + "\n" + r"$U^+_2=5000$"
    # ]
    df_selected.columns = [r"$U^+_1=0,U^+_2=50$",r"$U^+_1=33,U^+_2=50$",r"$U^+_1=50,U^+_2=50$",r"$U^+_1=100,U^+_2=50$"]
    
    df_selected.index = [
        "α=0; δ=0",
        "α=0.05; δ=0",
        "α=0.1; δ=0"
    ]
    # er_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=3300$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=5000$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=10000$" + "\n" + r"$U^+_2=5000$"
    # ]
    er_selected.columns = [r"$U^+_1=0,U^+_2=50$",r"$U^+_1=33,U^+_2=50$",r"$U^+_1=50,U^+_2=50$",r"$U^+_1=100,U^+_2=50$"]
    
    er_selected.index = [
        "α=0; δ=0",
        "α=0.05; δ=0",
        "α=0.1; δ=0"
    ]
    df_selected.T.plot(kind="bar",yerr=er_selected.T, capsize=3)
    plt.xlabel(r"$Utility (KRW,million)$",fontsize=10)
    plt.ylabel("Parental leave probability of Agent 2 (%)",fontsize=11)
    # plt.title("Utility-Probability",fontsize=11)
    plt.xticks(fontsize=10,rotation=0)
    plt.yticks(range(0,120,20),fontsize=10)
    plt.legend(loc="lower right")
    plt.tight_layout()
    learning_curve_image_path = os.path.join(image_dir,f"image5.png")
    plt.savefig(learning_curve_image_path)
    plt.show()

def image6():
    df_selected = df4.iloc[3:6, 0:3]
    er_selected = er4.iloc[3:6, 0:3]
    df_selected.columns = [r"$U^+_1=0,U^+_2=33$",r"$U^+_1=0,U^+_2=50$",r"$U^+_1=0,U^+_2=100$"]
    df_selected.index= [
        "α=0; δ=0.1",
        "α=0.05; δ=0.1",
        "α=0.1; δ=0.1"
    ]
    er_selected.columns = [r"$U^+_1=0,U^+_2=33$",r"$U^+_1=0,U^+_2=50$",r"$U^+_1=0,U^+_2=100$"]
    er_selected.index= [
        "α=0; δ=0.1",
        "α=0.05; δ=0.1",
        "α=0.1; δ=0.1"
    ]
    
    df_selected.T.plot(kind="bar",yerr=er_selected.T, capsize=3)
    plt.xlabel(r"$Utility (KRW,million)$",fontsize=10)
    plt.ylabel("Parental leave probability of Agent 2 (%)",fontsize=11)
    # plt.title("Utility-Probability",fontsize=11)
    plt.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.yticks(range(0,120,20),fontsize=10)
    plt.tight_layout()
    learning_curve_image_path = os.path.join("C:/Users/joyss/Desktop/ParentalLeave/images/",f"image6.png")
    plt.savefig(learning_curve_image_path)
    plt.show()

def image7():
    df_selected = pd.concat([df4.iloc[3:6, 0].reset_index(drop=True),df2.iloc[[1, 5, 6], 1].reset_index(drop=True)], axis=1)
    df_selected = pd.concat([df_selected,df3.iloc[3:6, 3:5].reset_index(drop=True)],axis=1)
    er_selected = pd.concat([er4.iloc[3:6, 0].reset_index(drop=True),er2.iloc[[1, 5, 6], 1].reset_index(drop=True)], axis=1)
    er_selected = pd.concat([er_selected,er3.iloc[3:6, 3:5].reset_index(drop=True)],axis=1)
    print(df_selected)
    print(er_selected)
    # df_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=3300$",
    #     r"$U^+_1=3300$" + "\n" + r"$U^+_2=3300$",
    #     r"$U^+_1=5000$" + "\n" + r"$U^+_2=3300$",
    #     r"$U^+_1=10000$" + "\n" + r"$U^+_2=3300$"
    # ]
    
    df_selected.columns= [r"$U^+_1=0,U^+_2=33$",r"$U^+_1=33,U^+_2=33$",r"$U^+_1=50,U^+_2=33$",r"$U^+_1=100,U^+_2=33$"]
    
    df_selected.index= [
        "α=0; δ=0.1",
        "α=0.05; δ=0.1",
        "α=0.1; δ=0.1"
    ]
    # er_selected.columns = [r"$U^+_1=0,U^+_2=3300$",r"$U^+_1=3300,U^+_2=3300$",r"$U^+_1=5000,U^+_2=3300$",r"$U^+_1=10000,U^+_2=3300$"]
    # er_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=3300$",
    #     r"$U^+_1=3300$" + "\n" + r"$U^+_2=3300$",
    #     r"$U^+_1=5000$" + "\n" + r"$U^+_2=3300$",
    #     r"$U^+_1=10000$" + "\n" + r"$U^+_2=3300$"
    # ]
    er_selected.columns = [r"$U^+_1=0,U^+_2=33$",r"$U^+_1=33,U^+_2=33$",r"$U^+_1=50,U^+_2=33$",r"$U^+_1=100,U^+_2=33$"]
    
    er_selected.index= [
        "α=0; δ=0.1",
        "α=0.05; δ=0.1",
        "α=0.1; δ=0.1"
    ]
    print(df_selected)
    df_selected.T.plot(kind="bar",yerr=er_selected.T, capsize=3)
    plt.xlabel(r"$Utility (KRW,million)$",fontsize=10)
    plt.ylabel("Parental leave probability of Agent 2 (%)",fontsize=10)
    # plt.title("Utility-Probability",fontsize=11)
    plt.xticks(fontsize=10,rotation=0)
    plt.legend(loc="lower right")
    plt.yticks(range(0,120,20),fontsize=10)
    plt.tight_layout()
    learning_curve_image_path = os.path.join("C:/Users/joyss/Desktop/ParentalLeave/images/",f"image7.png")
    plt.savefig(learning_curve_image_path)
    plt.show()

def image8():
    df_selected = pd.concat([df4.iloc[3:6, [1,3]].reset_index(drop=True),df2.iloc[[1, 5, 6], 3].reset_index(drop=True)], axis=1)
    df_selected = pd.concat([df_selected,df4.iloc[3:6, 5].reset_index(drop=True)],axis=1)
    er_selected = pd.concat([er4.iloc[3:6, [1,3]].reset_index(drop=True),er2.iloc[[1, 5, 6], 3].reset_index(drop=True)], axis=1)
    er_selected = pd.concat([er_selected,er4.iloc[3:6, 5].reset_index(drop=True)],axis=1)
    print(df_selected)
    # df_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=3300$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=5000$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=10000$" + "\n" + r"$U^+_2=5000$"
    # ]
    df_selected.columns = [r"$U^+_1=0,U^+_2=50$",r"$U^+_1=33,U^+_2=50$",r"$U^+_1=50,U^+_2=50$",r"$U^+_1=100,U^+_2=50$"]
    
    df_selected.index = [
        "α=0; δ=0.1",
        "α=0.05; δ=0.1",
        "α=0.1; δ=0.1"
    ]
    # er_selected.columns = [
    #     r"$U^+_1=0$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=3300$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=5000$" + "\n" + r"$U^+_2=5000$",
    #     r"$U^+_1=10000$" + "\n" + r"$U^+_2=5000$"
    # ]
    er_selected.columns = [r"$U^+_1=0,U^+_2=50$",r"$U^+_1=33,U^+_2=50$",r"$U^+_1=50,U^+_2=50$",r"$U^+_1=100,U^+_2=50$"]
    
    er_selected.index = [
        "α=0; δ=0.1",
        "α=0.05; δ=0.1",
        "α=0.1; δ=0.1"
    ]
    df_selected.T.plot(kind="bar",yerr=er_selected.T, capsize=3)
    plt.xlabel(r"$Utility (KRW,million)$",fontsize=10)
    plt.ylabel("Parental leave probability of Agent 2 (%)",fontsize=10)
    # plt.title("Utility-Probability",fontsize=11)
    plt.xticks(fontsize=10,rotation=0)
    plt.yticks(range(0,120,20),fontsize=10)
    plt.legend(loc="lower right")
    plt.tight_layout()
    learning_curve_image_path = os.path.join("C:/Users/joyss/Desktop/ParentalLeave/images/",f"image8.png")
    plt.savefig(learning_curve_image_path)
    plt.show()