import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop
import numpy as np


class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.Mode = str(self.args.running_mode)

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        #print("self.agent_params=",self.agent_params)
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        #print("episode batch=",EpisodeBatch)
        #print("batch=",batch,"--------------------------------------------------------------------------")
        #print("batch[intrinsic_reward]=",batch["intrinsic_reward"],"--------------------------------------------------------------------------")
        #print("batch[reward]=",batch["reward"],"--------------------------------------------------------------------------")
        #print("shape of batch[reward]=",batch["actions"].shape,"--------------------------------------------------------------------------")
        max_t = batch.max_seq_length
        
        rewards = batch["reward"][:, :-1]
        #print("rewards =",rewards.shape)
        #print("len rewards =",len(rewards))
        actions = batch["actions"][:, :]
        #print("actions =",actions.shape)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        #print("mask =",mask.shape)
        #print("len mask =",len(mask))
        avail_actions = batch["avail_actions"][:, :-1]
        

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        #print("mask2 =",mask.shape)

        q_vals, critic_train_stats, int_adv = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)
        #print("q_vals =",q_vals.shape)
        actions = actions[:,:-1]
        #print("actions2 =",actions.shape)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            #print("t=",t,"agent_outs=",agent_outs)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        #print("mac_out=",mac_out.shape)
        #print("mac_out shape =",mac_out.size())

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0
        #print("mac_out2=",mac_out.shape)
        #print("mac_out shape2 =",mac_out.size())

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()
        #print("baseline=",baseline.shape)

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        
        advantages = th.FloatTensor([0.0])
        #torch.clamp(a, min=-0.5, max=0.5)
        advantages = (q_taken - baseline).detach()
        #print("advantages",advantages)
        #print("int_adv",int_adv.shape)
        #print("advantages",advantages.shape)
        ##################################################### individual Intrinsic Reward
        advantages = advantages.reshape(-1)
        if self.Mode =="2":
            #int_adv = batch["intrinsic_reward"][:, :-1, :].reshape(-1)

            clip_ratio = 2
            int_adv = int_adv.view(-1)
            
            for t in range(len(advantages)):
                #print("adv shape =",advantages[t])
                #print("int_adv shape =",int_adv[t])
                
                int_adv_clipped = th.clamp(int_adv[t],min = clip_ratio*-advantages[t].item(), max =clip_ratio*advantages[t].item())
                advantages[t] = advantages[t]+ int_adv_clipped
            
        
        #print("advantages after",advantages)
        
        ##################################################### Combined Intrinsic Reward
        #print("batchzzzz = ",batch["intrinsic_reward"][:, :-1, 3])
        elif self.Mode =="5":
            #print("batch all =", th.cat((batch["intrinsic_reward"][:, :-1, :],batch["intrinsic_reward"][:, :-1, :],batch["intrinsic_reward"][:, :-1, :]),0).reshape(-1).shape)
            #print("batch soze =", batch["intrinsic_reward"][:, :-1, :].shape)
            #print("advantages  =", advantages.shape)
            #temp = []
            #int_adv = batch["intrinsic_reward"][:, :-1, :]
            #for p in range(self.n_agents-1):
            #    int_adv = th.cat((int_adv,batch["intrinsic_reward"][:, :-1, :]),0)
            #int_adv = int_adv.view(-1)

                
            #int_adv = th.cat((batch["intrinsic_reward"][:, :-1, :],batch["intrinsic_reward"][:, :-1, :],batch["intrinsic_reward"][:, :-1, :]),1).reshape(-1)
            clip_ratio = 2
            #for t in range(len(advantages)):

            int_adv = int_adv.view(-1)
            
            #print("adv shape =",advantages.shape)
            #print("int_adv shape =",int_adv.shape)
            for t in range(len(advantages)):
                int_adv_clipped = th.clamp(int_adv[t],min = clip_ratio*-advantages[t].item(), max =clip_ratio*advantages[t].item())
                advantages[t] = advantages[t]+ int_adv_clipped
            #print("int_adv shape =",int_adv_clipped.shape)
            #print("advantages=",advantages.shape)
            
        else:
            pass
        
        
        
        #print("advantages after",advantages)
        ###################################################################################
        #print("int_adv",int_adv.shape)
        #print("batch[intrinsic_reward]",batch["intrinsic_reward"].shape)
        #print("batch[reward]",batch["reward"].shape)
        #print("log_pi_taken",log_pi_taken.shape)
        #print("advantages",advantages.shape)
        
        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()
        #print("self.agent_optimiser=",self.agent_optimiser)
        # Optimise agents
        #print(self.critic.parameters())
        #print(self.agent_optimiser.parameters())
        
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        #print("batch obs =",batch["obs"][0][0])
        
        #print("test" , self.target_critic(batch))
        target_q_vals , _ = self.target_critic(batch)
        #target_q_vals = target_q_vals[:,:]
        #print("target_q_vals=",target_q_vals.shape)
        #print("_=",_.shape)
        #print("shape target_q_vals=",target_q_vals.shape)
        #print("batch obs =",batch["obs"])
        #print("size batch obs =",batch["obs"].size())
        #print("rewards", rewards)
        #print("size of rewards", rewards.shape)
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)
        
        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)
        #print("targets=",targets)
        
        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }
        
        #adv_int = []
        adv_int = []
        adv_int_ = 0
        for t in reversed(range(rewards.size(1))):
            #print("mask_t before=",mask[:, t])
            mask_t = mask[:, t].expand(-1, self.n_agents)
            #print("mask_t after=",mask_t)
            if mask_t.sum() == 0:
                continue

            q_t, q_int = self.critic(batch, t) # may be implement in here
            
            #print("batch check what inside =",batch)
            #print("q_t=",q_t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            #print("q_vals=",q_vals)
            #print("q_vals shpae=",q_vals.shape)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            #print("q_taken=",q_taken)
            targets_t = targets[:, t]
            #print("targets_t=",targets_t)

            td_error = (q_taken - targets_t.detach())
            
            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t
            #print("len r" ,rewards.size(1))
            q_int = q_int[:,:,0,:] #can change 0 to others
            ###################################################
            if self.Mode =="2":
                int_adv = batch["intrinsic_reward"][:, t, :].reshape(-1) # int_adv = batch["intrinsic_reward"][:, :-1, :].reshape(-1)
                int_adv = th.tensor(int_adv)
                
                int_adv = int_adv.view(-1)
                q_int = q_int.reshape(-1)
                int_adv_norm = (int_adv - th.mean(int_adv))/th.sqrt(th.var(int_adv)+0.000001)
                adv_int_ = int_adv_norm.float().detach() - q_int.float()
                adv_int.append(adv_int_)
                #adv_int = th.stack(adv_int,adv_int_, dim=0)
                #adv1_int = (adv1_int - torch.mean(adv1_int))/torch.sqrt(torch.var(adv1_int)+0.000001)
            
            elif self.Mode =="5":
                int_adv = batch["intrinsic_reward"][:, t, :] #int_adv = batch["intrinsic_reward"][:, :-1, :]
                int_adv = th.tensor(int_adv)
                #int_adv = int_adv.detach()
                #print("int_adv =" ,int_adv.shape)
                for p in range(self.n_agents-1):
                    int_adv = th.cat((int_adv,batch["intrinsic_reward"][:, t, :]),0)
                int_adv = int_adv.view(-1)
                q_int = q_int.reshape(-1)
                int_adv_norm = (int_adv - th.mean(int_adv))/th.sqrt(th.var(int_adv)+0.000001)
                
                #print("int_adv_norm =" ,int_adv_norm.shape)
                #print("q_int =" ,q_int.shape)
                adv_int_ = int_adv_norm.float().detach() - q_int.float()
                adv_int.append(adv_int_)
                #adv_int = th.stack([adv_int_], dim=0)
            
            masked_td_error_int = adv_int_ * mask_t.reshape(-1)
            ##################################################
            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum() + (masked_td_error_int ** 2).sum() / mask_t.sum() 
            self.critic_optimiser.zero_grad()
            loss.backward(retain_graph=True) #retain_graph=True
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)
            
            #adv_int.append(adv_int_)
            #adv_int[t] = adv_int_
        if (self.Mode =="2" or self.Mode =="5"):
            adv_int = th.stack(adv_int)
        else:
            adv_int = 0
        #print("adv_int_sum = ",len(adv_int))
        #print("adv_int_sum = ",adv_int.shape)
        #adv_int = th.FloatTensor(adv_int)
        return q_vals, running_log, adv_int

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
