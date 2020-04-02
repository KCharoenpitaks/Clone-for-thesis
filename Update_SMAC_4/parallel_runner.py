from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import torch.nn as nn
#from utils import RunningMeanStd

device = th.device("cpu")
# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        #self.Mode = "2"
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        #sprint("self.batch_size=",self.batch_size)
        self.Mode = str(self.args.running_mode)
        #print("self.Mode=",self.Mode)

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                            for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        #print(self.env_info) #self.env_info,obs_shape, obs_shape,n_actions
        self.t = 0
        self.temp_input3 = 0

        self.t_env = 0
        self.n_agents = self.env_info["n_agents"]
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.lr = 0.002/self.env_info["n_agents"]
        self.betas = (0.9, 0.999)
        self.log_train_stats_t = -100000
        ### Use this one only
        self.RND_shape = self.env_info["obs_shape"] + self.env_info["n_agents"]
        #print("self.RND_shape",self.RND_shape)
        self.RND_net = RNDforPPO(self.RND_shape,self.env_info["n_actions"],64)
        self.RND_net_optimizer = th.optim.Adam(self.RND_net.parameters(),
                                              lr=self.lr, betas=self.betas)
        self.MseLoss1 = nn.MSELoss()
        
        self.RND_shape_all = self.env_info["obs_shape"]*self.env_info["n_agents"]
        self.RND_net_all = RNDforPPO(self.RND_shape_all,self.env_info["n_actions"],64)
        self.RND_net_optimizer_all = th.optim.Adam(self.RND_net_all.parameters(),
                                              lr=self.lr, betas=self.betas)
        
        ###
        """
        self.RND_net2 = RNDforPPO2(self.env_info["obs_shape"],self.env_info["n_actions"],64)
        self.RND_net_optimizer2 = th.optim.Adam(self.RND_net2.parameters(),
                                              lr=self.lr, betas=self.betas)
        self.MseLoss2 = nn.MSELoss()
        ###
        self.RPN_net3 = Reward_prediction(self.env_info["obs_shape"],self.env_info["n_actions"],64)
        self.RPN_net_optimizer3 = th.optim.Adam(self.RPN_net3.parameters(),
                                              lr=self.lr, betas=self.betas)
        self.MseLoss3 = nn.MSELoss()
        """
        self.rms_int= RunningMeanStd()
        #self.rms_all = RunningMeanStd()
        
    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()
        #print()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back

        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            #print("//////////////////////////////////////////////////////////////////////////////////////////////data", data)
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        #print("end loop+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()
        count = MemoryCount()
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()#.to(device)

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            #print("time =",self.t,"actions=",actions)
            
            # Send actions to each env
            action_idx = 0
            #print("self.parent_conns=",self.parent_conns)
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            if (self.Mode =="2" or self.Mode =="5"):
                post_transition_data = {
                    "reward": [],
                    "terminated": [],
                    "intrinsic_reward": []
                }
            else:
                post_transition_data = {
                    "reward": [],
                    "terminated": []
                }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                #print("idx=", idx)
                #print("parent_conn=", parent_conn)
                
                if not terminated[idx]:
                    data = parent_conn.recv()
                    
                    #print("data=",data)
                    #print("self.parent_conns=",self.parent_conns)
                    #print("idx=",idx)
                    #print("data obs=",data["obs"])
                    #print("data state=",data["state"])
                    # Remaining data for this current timestep
                    #print("reward for parallel =",data["reward"])
                    ####################################################################
                    #print("idx=",idx,"reward =",data["reward"],)
                    """
                    if self.Mode == "0": # Intrinsic_reward_noise
                        data["reward"] = data["reward"] + get_intrinsic_reward_noise(data["reward"],self.t,50,0.99)
                        if self.t%50 == 0:
                            print("00000000000000000000")
                    
                    elif self.Mode == "1": #Count-based
                        reward_temp = 0
                        for i in range(len(data["obs"])):
                            #print("i=",i,"state =",data["obs"][i],)
                            #print("i=",i,"reward =",data["reward"],)
                            temp_rew = get_intrinsic_count_based(count,data["obs"][i],1,256)
                            reward_temp += temp_rew
                            #print(temp_rew)
                         
                        #print(data["reward"],"======")
                        data["reward"] = data["reward"] + reward_temp/len(data["obs"])
                        #print(data["reward"])
                        if self.t%50 == 0:
                            print("1111111111111111")
                            """
                    if (self.Mode =="2" or self.Mode =="5"): # RND1
                        reward_temp = 0
                        #print("range of data obs=",range(len(data["obs"])))
                        #print("data obs=", data["obs"])
                        temp_input = []
                        temp_input2 = []
                        self.temp_input3 = []
                        temp_list = []
                        self.temp_input_all = []
                        for i in range(len(data["obs"])):
                            
                            temp_input = data["obs"][i]
                            temp_input2 = np.eye(self.n_agents)[i]
                            #print("temp_input=",temp_input)
                            #temp_input.append(th.eye(self.n_agents))
                            #print("eye = ",np.eye(self.n_agents))

                            self.temp_input3 = np.concatenate((temp_input,temp_input2))
                            #print("temp_input3=",self.temp_input3)
                            
                            temp_rew = get_intrinsic_reward_RND1(self.temp_input3,self.RND_net,5)
                            temp_rew =temp_rew.data.cpu().numpy()
                            temp_list.append(temp_rew)
                            

                            #reward_temp += temp_rew
                            """
                            #print("i=",i,"state =",data["obs"][i],)
                            #print("i=",i,"reward =",data["reward"],)
                            temp_rew = get_intrinsic_reward_RND1(data["obs"][i],self.RND_net,5)
                            temp_rew =temp_rew.data.cpu().numpy()
                            reward_temp += temp_rew
                            #print("JJJJJJJJJJJJJJJJJJJJJJJJ")
                            #print("i=",i,"temp_reward",temp_rew)  
                            #print("i=",i,"out =",self.RND_net.predictor_RND(data["obs"][i]))
                            """
                            
                        self.temp_input_all = np.array(data["obs"]).reshape(-1)
                        temp_rew_all = get_intrinsic_reward_RND1(self.temp_input_all,self.RND_net_all,5)
                        #print("temp_list",temp_list.shape)
                        temp_list.append(temp_rew_all.data.numpy()/3)
                        #print("temp_list2",temp_list.shape)
                        #data["reward"] = data["reward"] + reward_temp/len(data["obs"]) 
                        data["intrinsic_reward"] = np.array(temp_list).reshape(1,-1)
                        
                        #print("SSSSSSSSSSSSSSSSSSSSSSSSSS", data["intrinsic_reward"].shape)
                        self.rms_int.update(data["intrinsic_reward"])
                        #print("self.rms_int.var",self.rms_int.var)

                        data["intrinsic_reward"] = data["intrinsic_reward"]/(np.sqrt(self.rms_int.var)+0.0000001)
                        #print("SSSSSSSSSSSSSSSSSSSSSSSSSS 222222", data["intrinsic_reward"].shape)
                        
                        #r1_int_list/np.sqrt(reward_rms1.var)
                        #print("rewards=",data["reward"])
                        if self.t%200 == 0:
                            if self.Mode =="2":
                                print("22222222222222")
                            elif self.Mode =="5":
                                print("55555555555555")
                                
                            """
                    elif self.Mode =="3": # RND2
                        reward_temp = 0
                        for i in range(len(data["obs"])):
                            #print("i=",i,"state =",data["avail_actions"][i],)
                            #print("i=",i,"state =",data,)
                            #print("i=",i,"actions_chosen=",actions_chosen["actions"][0][0][i].numpy())
                            #print("i=",0,"actions_chosen=",actions_chosen["actions"][0][0][0].numpy())
                            #print("i=",1,"actions_chosen=",actions_chosen["actions"][0][0][1].numpy())
                            #print("i=",i,"reward =",data["reward"],)
                            #print("i=",i,"data_obs",data["obs"][i])
                            #print("i=",i,"avail_actions",data["avail_actions"][i])
                            temp_rew = get_intrinsic_reward_RND2(data["obs"][i],actions_chosen["actions"][0][0][i],self.RND_net2,5)
                            temp_rew =temp_rew.data.cpu().numpy()
                            reward_temp += temp_rew
                            #print("i=",i,"temp_reward",temp_rew)  

                            #print("i=",i,"out =",self.RND_net.predictor_RND(data["obs"][i]))
                        data["reward"] = data["reward"] + reward_temp/len(data["obs"]) 
                        if self.t%50 == 0:
                            print("3333333333333333")
                    elif self.Mode =="4": # RPN
                        reward_temp = 0
                        for i in range(len(data["obs"])):
                            #print("i=",i,"state =",data["avail_actions"][i],)
                            #print("i=",i,"state =",data,)
                            #print("i=",i,"actions_chosen=",actions_chosen["actions"][0][0][i].numpy())
                            #print("i=",0,"actions_chosen=",actions_chosen["actions"][0][0][0].numpy())
                            #print("i=",1,"actions_chosen=",actions_chosen["actions"][0][0][1].numpy())
                            #print("i=",i,"reward =",data["reward"],)
                            #print("i=",i,"data_obs",data["obs"][i])
                            #print("i=",i,"avail_actions",data["avail_actions"][i])
                            temp_rew = get_intrinsic_reward_RPN3(data["obs"][i],actions_chosen["actions"][0][0][i],self.RPN_net3,data["reward"],5)
                            temp_rew =temp_rew.data.cpu().numpy()
                            reward_temp += temp_rew
                            #print("i=",i,"temp_reward",temp_rew)  

                            #print("i=",i,"out =",self.RND_net.predictor_RND(data["obs"][i]))
                        data["reward"] = data["reward"] + reward_temp/len(data["obs"]) 
                        if self.t%50 == 0:
                            print("4444444444444444")
                            """
                    elif self.Mode=="normal":
                        
                        if self.t%50 == 0:
                            
                            print("Normal")

                        pass
                        
                    else:
                        if self.t%50 == 0:
                            print("Pass")
                        pass
                    #data["reward"] = data["reward"].data.numpy()
                    #print("datarewards",data["reward"])
                    episode_returns[idx] += data["reward"]
                    #print("episode_returns idx =",episode_returns[idx])
                    #episode_returns[idx] = episode_returns[idx].data.numpy()
                    episode_lengths[idx] += 1
                    ##################################################  
                        
                    #print("idx=",idx,"state =",data["obs"],) #find state
                    if self.Mode == "2": #RND1
                        loss_RND1 = self.RND_net.RND_diff(self.temp_input3).sum()#self.MseLoss1(RND_Net_values.detach(), RND_predictor_values)
                        self.RND_net_optimizer.zero_grad()
                        loss_RND1.backward()          
                        self.RND_net_optimizer.step()
                    elif self.Mode == "3": #RND2
                        loss_RND2 = self.RND_net2.RND_diff(data["obs"],actions_chosen["actions"][0][0]).sum()#reward_temp
                        self.RND_net_optimizer2.zero_grad()
                        loss_RND2.backward()          
                        self.RND_net_optimizer2.step()
                    elif self.Mode == "4": #RPN3
                        #print("HIIIIIII")
                        loss_RPN3 = self.RPN_net3.diff(data["obs"],actions_chosen["actions"][0][0],reward_temp).sum()#reward_temp
                        self.RPN_net_optimizer3.zero_grad()
                        loss_RPN3.backward()          
                        self.RPN_net_optimizer3.step()
                    elif self.Mode == "5": #RPN3
                        #print("HIIIIIII")
                        loss_RND_all = self.RND_net_all.RND_diff(self.temp_input_all).sum()#self.MseLoss1(RND_Net_values.detach(), RND_predictor_values)
                        self.RND_net_optimizer_all.zero_grad()
                        loss_RND_all.backward()          
                        self.RND_net_optimizer_all.step()
                    
                    ###############################################
                    
                    #print("(data[reward],)", (data["reward"],))
                    #print("(data[intrinsic_reward],)", data["intrinsic_reward"].shape)
                    
                    
                    post_transition_data["reward"].append((data["reward"],))
                    if (self.Mode =="2" or self.Mode =="5"):
                        post_transition_data["intrinsic_reward"].append(data["intrinsic_reward"],)
                    else:
                        pass
                    
                    #print("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                    
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    
                    #print("obs data = ", data["obs"])
            #print("post_transition_data[reward]=",post_transition_data["reward"])
            #print("pre_transition_data[obs]=",pre_transition_data["obs"])
             

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            #print("self.batch=",self.batch["reward"][0]) # number is equal to number of parallel

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        #print("cur_returns=",cur_returns)
        #cur_returns = cur_returns.data.numpy()
        #episode_returns.data.numpy()
        #print("episode_returns=",episode_returns)
        
        cur_returns.extend(episode_returns)
        #print("cur_returns=",cur_returns)
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        print(returns)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def get_intrinsic_count_based(memcount,states,multiple,bucket_size):
    rewards_intrinsic = {}
    count = {}
    memcount.add(str(divide_count(states,bucket_size)))
    count = memcount.count(str(divide_count(states,bucket_size)))
    rewards_intrinsic = 0.1/np.sqrt(count)*multiple
    return rewards_intrinsic

class MemoryCount():
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer2 = {}
    
    def add(self, experience):
        try:
            self.buffer2[experience] += 1
        except:
            self.buffer2[experience] = 1
        
    def delete(self):
        self.buffer2 = {}
    
    def count(self, b):
        return self.buffer2[b]
    def leng(self):
        return len(self.buffer2)
    
def divide_count(state, bucket_size):
    out = state//bucket_size
    return out

def get_intrinsic_reward_noise(episode_reward,timesteps,multiple,discount):
    episode_reward = max(episode_reward,1)
    rewards_intrinsic = np.random.normal(0,0.1)*np.power(discount,timesteps)*multiple*np.log(episode_reward)
    return rewards_intrinsic

def get_intrinsic_reward_RND1(states,network,multiple):
    rewards_intrinsic = network.RND_diff(states)*multiple
    return rewards_intrinsic

def get_intrinsic_reward_RND2(states,actions,network,multiple):
    rewards_intrinsic = network.RND_diff(states,actions)*multiple
    return rewards_intrinsic

def get_intrinsic_reward_RPN3(states,actions,network,rewards,multiple):
    rewards_intrinsic = network.diff(states,actions,rewards)*multiple
    return rewards_intrinsic

class RNDforPPO(nn.Module):
    def __init__(self, state_dim,action_dim, n_latent_var):
        super(RNDforPPO, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        self.MseLoss = nn.MSELoss()
        
        self.RND_NN_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
        self.Predictor_NN_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
    
    def forward_RND(self, state):
        state = th.from_numpy(np.array(state)).float()
        state = state
        value = self.RND_NN_layer(state)
        return th.squeeze(value)
    
    def predictor_RND(self, state):
        state = th.from_numpy(np.array(state)).float()
        #state= state
        #print(state)
        value = self.Predictor_NN_layer(state)
        return th.squeeze(value)
    
    def RND_diff(self,state):
        #print(state)
        #state = np.array(state)
        predictor = self.predictor_RND(state)
        forward = self.forward_RND(state)
        diff = self.MseLoss(forward,predictor.detach())
        diff.data.numpy()
        return diff#.data.numpy()
    
class RNDforPPO2(nn.Module):
    def __init__(self, state_dim,action_dim , n_latent_var):
        super(RNDforPPO2, self).__init__()
        #self.affine = nn.Linear(state_dim, n_latent_var)
        self.MseLoss = nn.MSELoss()
        self.action_dim=action_dim


        self.RND_NN_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
        self.Predictor_NN_layer = nn.Sequential(
                nn.Linear(int(state_dim)+int(action_dim), n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
                
    
    def forward_RND(self, state):
        state = th.from_numpy(np.array(state)).float()
        value = self.RND_NN_layer(state)
        return th.squeeze(value)
    
    def predictor_RND(self, state,action):
        
        #print(action)
        #print(self.action_dim)
        action = to_categorical(action,self.action_dim)
        action = th.from_numpy(action).float()
        #action = torch.tensor(action.clone().detach()).float()
        action = th.squeeze(action,0)

        
        state = th.from_numpy(np.array(state)).float()
        state= state
        #print("state=",state.shape)
        #print("action=",action.shape)
        state_action = th.cat((state, action), -1)

        value = self.Predictor_NN_layer(state_action)
        return th.squeeze(value)
    
    def RND_diff(self,state,action):
        action = action.cpu()
        action = np.array(action)
        predictor = self.predictor_RND(state,action)
        forward = self.forward_RND(state)
        diff = self.MseLoss(forward,predictor.detach())
        diff.data.numpy()
        return diff#.data.numpy()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class Reward_prediction(nn.Module):
    def __init__(self, state_dim,action_dim ,n_latent_var):
        super(Reward_prediction, self).__init__()
        #self.affine = nn.Linear(state_dim, n_latent_var)
        self.MseLoss = nn.MSELoss()
        self.action_dim=action_dim

        self.Predictor_NN_layer = nn.Sequential(
                nn.Linear(int(state_dim)+int(action_dim), n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1),
                )
    
    def predictor_RND(self, state,action):

        action = to_categorical(action,self.action_dim)
        action = th.from_numpy(action).float()
        #action = torch.tensor(action.clone().detach()).float()
        action = th.squeeze(action,0)

        
        state = th.from_numpy(np.array(state)).float()
        state= state
        state_action = th.cat((state, action), -1)
        value = self.Predictor_NN_layer(state_action)
        return th.squeeze(value)
    
    def diff(self, state, action,reward):
        action = action.cpu()
        action = np.array(action)
        predictor = self.predictor_RND(state,action)
        reward = th.tensor(reward)
        diff = self.MseLoss(reward,predictor)

        #print(diff)
        return diff#.data.numpy()
    
    
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]#len(x) #x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        #print(self.count)
        #print(batch_count)
        tot_count = self.count + int(batch_count)

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
