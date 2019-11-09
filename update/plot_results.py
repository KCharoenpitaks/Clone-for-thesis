import numpy as np
import json
from pprint import pprint
import matplotlib.pyplot as plt

number = ["36","37","39","40","41"]
#number = str(np.arange(38))
##############################################################################
def plot(name,time, Average_win): #def plot(name,time, Average_win,color):
    t = Time#np.arange(1,time,1) 
    plt.figure('Episode Rwards of '+ map_name + ' for ' + Name)
    plt.plot(t, Average_win) #plt.plot(t, Average_win,color)
    plt.xlabel('Episode')
    plt.ylabel('Rewards of ' + map_name + ' for ' + Name)
    plt.title("Winning Probility on "+ map_name )
    #plt.show()   
##########################################################################


for num in number:
    #directory = 'Desktop/RL/results/sacred/'+number+'/info.json'
    directory = 'results/sacred/'+num+'/info.json'
    directory2 = 'results/sacred/'+num+'/config.json'
    #print(directory)
    
    
    with open(directory) as json_file:
        info_data = json.load(json_file)
        
    with open(directory2) as json_file2:
        config_data = json.load(json_file2)
        
    pprint(config_data)
    
    Time = info_data["battle_won_mean_T"]
    Average_win = info_data["battle_won_mean"]
    Name = config_data["name"]
    map_name = config_data["env_args"]["map_name"] 
    
    #########################################
    algor_type = config_data ["running_mode"]
    if algor_type == "0":
        algor_type = "Intrinsic_Reward_Noise"
    elif algor_type == "1":
        algor_type = "Count-based_Intrinsic_Reward"
    elif algor_type == "2":
        algor_type = "RND with current state input"
    elif algor_type == "3":
        algor_type = "RND with previous state and action input"
    elif algor_type == "4":
        algor_type = "Reward Prediction Network"
    else:
        algor_type = "No Intrinsic Value"
    plot(Name,Time,Average_win)
        


#plot(Name,Time,Average_win) #plot(Name,Time,Average_win,'r')

plt.show()  




