from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import torch.nn as nn


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        #self.Mode = "2"
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.Mode = self.args.running_mode

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

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.lr = 0.002/self.env_info["n_agents"]
        self.betas = (0.9, 0.999)
        self.log_train_stats_t = -100000
        ###
        self.RND_net = RNDforPPO(self.env_info["obs_shape"],self.env_info["n_actions"],64)
        self.RND_net_optimizer = th.optim.Adam(self.RND_net.parameters(),
                                              lr=self.lr, betas=self.betas)
        self.MseLoss1 = nn.MSELoss()
        ###
        self.RND_net2 = RNDforPPO2(self.env_info["obs_shape"],self.env_info["n_actions"],64)
        self.RND_net_optimizer2 = th.optim.Adam(self.RND_net2.parameters(),
                                              lr=self.lr, betas=self.betas)
        self.MseLoss2 = nn.MSELoss()
        ###
        self.RPN_net3 = Reward_prediction(self.env_info["obs_shape"],self.env_info["n_actions"],64)
        self.RPN_net_optimizer3 = th.optim.Adam(self.RPN_net3.parameters(),
                                              lr=self.lr, betas=self.betas)
        self.MseLoss3 = nn.MSELoss()

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
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

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
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            #print("time =",self.t,"actions=",actions)
            
            # Send actions to each env
            action_idx = 0
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
                    #print("idx=",idx,"data=",data)
                    # Remaining data for this current timestep
                    #print("------",data["reward"])
                    ####################################################################
                    #print("idx=",idx,"reward =",data["reward"],)
                    
                    if self.Mode == "0": # Intrinsic_reward_noise
                        data["reward"] = data["reward"] + get_intrinsic_reward_noise(data["reward"],self.t_env,50,0.99)
                    
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
                    
                    elif self.Mode =="2": # RND1
                        reward_temp = 0
                        for i in range(len(data["obs"])):
                            #print("i=",i,"state =",data["obs"][i],)
                            #print("i=",i,"reward =",data["reward"],)
                            temp_rew = get_intrinsic_reward_RND1(data["obs"][i],self.RND_net,5)
                            reward_temp += temp_rew
                            #print("i=",i,"temp_reward",temp_rew)  
                            #print("i=",i,"out =",self.RND_net.predictor_RND(data["obs"][i]))
                        data["reward"] = data["reward"] + reward_temp/len(data["obs"]) 
                        
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
                            reward_temp += temp_rew
                            #print("i=",i,"temp_reward",temp_rew)  

                            #print("i=",i,"out =",self.RND_net.predictor_RND(data["obs"][i]))
                        data["reward"] = data["reward"] + reward_temp/len(data["obs"]) 
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
                            reward_temp += temp_rew
                            #print("i=",i,"temp_reward",temp_rew)  

                            #print("i=",i,"out =",self.RND_net.predictor_RND(data["obs"][i]))
                        data["reward"] = data["reward"] + reward_temp/len(data["obs"]) 
                        
                    else:
                        pass
            
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    ##################################################  
                        
                    #print("idx=",idx,"state =",data["obs"],) #find state
                    if self.Mode == "2": #RND1
                        loss_RND1 = reward_temp#self.MseLoss1(RND_Net_values.detach(), RND_predictor_values)
                        self.RND_net_optimizer.zero_grad()
                        loss_RND1.backward()          
                        self.RND_net_optimizer.step()
                    elif self.Mode == "3": #RND2
                        loss_RND2 = reward_temp
                        self.RND_net_optimizer2.zero_grad()
                        loss_RND2.backward()          
                        self.RND_net_optimizer2.step()
                    elif self.Mode == "4": #RPN3
                        #print("HIIIIIII")
                        loss_RPN3 = reward_temp
                        self.RPN_net_optimizer3.zero_grad()
                        loss_RPN3.backward()          
                        self.RPN_net_optimizer3.step()
                    
                    ###############################################
                    post_transition_data["reward"].append((data["reward"],))
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

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

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

        cur_returns.extend(episode_returns)

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
        state = th.from_numpy(state).float()
        state = state
        value = self.RND_NN_layer(state)
        return th.squeeze(value)
    
    def predictor_RND(self, state):
        state = th.from_numpy(state).float()
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
        return diff
    
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
        state = th.from_numpy(state).float()
        value = self.RND_NN_layer(state)
        return th.squeeze(value)
    
    def predictor_RND(self, state,action):
        
        #print(action)
        #print(self.action_dim)
        action = to_categorical(action,self.action_dim)
        action = th.from_numpy(action).float()
        #action = torch.tensor(action.clone().detach()).float()
        action = th.squeeze(action,0)

        
        state = th.from_numpy(state).float()
        state= state
        #print("state=",state.shape)
        #print("action=",action.shape)
        state_action = th.cat((state, action), -1)

        value = self.Predictor_NN_layer(state_action)
        return th.squeeze(value)
    
    def RND_diff(self,state,action):
        action = np.array(action)
        predictor = self.predictor_RND(state,action)
        forward = self.forward_RND(state)
        diff = self.MseLoss(forward,predictor.detach())
        return diff


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

        
        state = th.from_numpy(state).float()
        state= state
        state_action = th.cat((state, action), -1)
        value = self.Predictor_NN_layer(state_action)
        return th.squeeze(value)
    
    def diff(self, state, action,reward):
        action = np.array(action)
        predictor = self.predictor_RND(state,action)
        reward = th.tensor(reward)
        diff = self.MseLoss(reward,predictor)
        #print(diff)
        return diff