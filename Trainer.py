import torch,os,sys,logging,json
from Environment import  ParallelEnvironment
from Policy import Actor,Critic
from collections import defaultdict
import numpy as np
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, ParallelEnv,TransformedEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from logging.handlers import SysLogHandler
#os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' Cannot be used for parallel
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = SysLogHandler(address='/dev/log')
formatter = logging.Formatter('%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Trainer :
    def __init__(self):
        
        self.checkpoint_path = 'data/checkpoint.pt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        #Hyper Params
        self.num_cells = 256
        self.lr = 3e-4
        self.max_grad_norm = 1.0
        self.frames_per_batch = 400
        self.total_frames = 10000000
        # PPO Params
        self.sub_batch_size = 256
        self.epochs = 10
        self.clip_epsi = 0.2
        self.gamma = 0.99
        self.lmda = 0.95
        self.entropy = 0.05
        self.transformed_env = ParallelEnv(
            4,
            [
                lambda: TransformedEnv(
                    ParallelEnvironment('1'),
                    Compose()
                ),
                lambda: TransformedEnv(
                    ParallelEnvironment('2'),
                    Compose()
                ),
                lambda: TransformedEnv(
                    ParallelEnvironment('3'),
                    Compose()
                ),
                lambda: TransformedEnv(
                    ParallelEnvironment('4'),
                    Compose()
                )  
            ]
        )
        check_env_specs(self.transformed_env)   
        self.policy_module = TensorDictModule(Actor().to(self.device),
                                              in_keys=['observation'],
                                              out_keys=['logits']
                                            )
    
        self.action_module = ProbabilisticActor(self.policy_module,
                                                in_keys=['logits'],
                                                out_keys=['action'],
                                                distribution_class = torch.distributions.Categorical,
                                                return_log_prob = True
                                            )
        self.value_module = ValueOperator(Critic().to(self.device),
                                          in_keys=['observation'],
                                          out_keys=['state_value'])
        
        self.collector = SyncDataCollector(self.transformed_env,
                                           self.action_module,
                                           frames_per_batch=self.frames_per_batch,
                                           total_frames=self.total_frames,
                                           device= self.device,
                                           split_trajs=True)
        
        self.replay = ReplayBuffer(storage=LazyTensorStorage(max_size=self.frames_per_batch),
                                   sampler=SamplerWithoutReplacement(),
                                )
        
        self.adv_module = GAE(gamma=self.gamma,
                              lmbda=self.lmda,
                              value_network=self.value_module,
                              average_gae=True,
                              device=self.device)
        
        self.loss_module = ClipPPOLoss(actor_network=self.action_module,
                                       critic_network=self.value_module,
                                       clip_epsilon=self.clip_epsi,
                                       entropy_coef=self.entropy,
                                       critic_coef=1.0,
                                       )
        
        self.optim = torch.optim.Adam(self.loss_module.parameters(),self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,self.total_frames // self.frames_per_batch,0.0)
        self.start_iteration = 0
        self.trained_frames = 0

        if os.path.exists(self.checkpoint_path):
            print("Checkpoint found. Loading model...")
            self.load_checkpoint(self.checkpoint_path)
        else:
            print("No checkpoint found. Starting fresh.")


    def save_checkpoint(self, path="data/checkpoint.pt"):
        checkpoint = {
            "actor": self.action_module.state_dict(),
            "critic": self.value_module.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "iteration": self.start_iteration,
            "trained_frames": self.trained_frames,
        }

        torch.save(checkpoint, path)
        tqdm.write(f"[Checkpoint] Saved to {path}")


    def load_checkpoint(self, path="checkpoint/checkpoint.pt"):
        checkpoint = torch.load(path, map_location=self.device)

        self.action_module.load_state_dict(checkpoint["actor"])
        self.value_module.load_state_dict(checkpoint["critic"])
        self.optim.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.start_iteration = checkpoint.get("iteration", 0)
        self.trained_frames = checkpoint.get("trained_frames", 0)


        print(f"[Checkpoint] Loaded from {path}")
        print(f"Resuming from iteration {self.start_iteration}")
        print(f"Frames trained: {self.trained_frames}")

    def log_metrics(self,avg_reward, init_reward, step_count, lr, policy_loss, critic_loss, entropy):
        data = {
            "avg_reward": avg_reward,
            "init_reward": init_reward,
            "step_count": step_count,
            "lr": lr,
            "policy_loss": policy_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
        }
        logger.info(json.dumps(data))

    def train(self):
        logs = defaultdict(list)
        pbar = tqdm(total=self.total_frames - self.trained_frames,dynamic_ncols=False,file=sys.stderr)
        eval_str = ''

        for i,tensordict_data in enumerate(self.collector,start=self.start_iteration):
            logs["loss_objective"] = []
            logs["loss_critic"] = []
            logs["loss_entropy"] = []
            for _ in range(self.epochs):
                #os.system('clear')
                self.adv_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay.empty()
                self.replay.extend(data_view)
                for _ in range(self.frames_per_batch // self.sub_batch_size):
                    subdata = self.replay.sample(self.sub_batch_size)
                    loss_vals = self.loss_module(subdata.to(self.device))
                    loss_value = (loss_vals['loss_objective']
                                  + loss_vals['loss_critic']
                                  + loss_vals['loss_entropy']
                                )
                    logs["loss_objective"].append(loss_vals["loss_objective"].item())
                    logs["loss_critic"].append(loss_vals["loss_critic"].item())
                    logs["loss_entropy"].append(loss_vals["loss_entropy"].item())
                    logs["adv_mean"].append(subdata["advantage"].mean().item())
                    logs["adv_std"].append(subdata["advantage"].std().item())
                    self.optim.zero_grad()
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(),self.max_grad_norm)
                    self.optim.step()
            


            logs["policy_loss"].append(np.mean(logs['loss_objective']))
            logs["critic_loss"].append(np.mean(logs['loss_critic']))
            logs["entropy_loss"].append(np.mean(logs['loss_entropy']))
            ppo_str = (
                f"policy_loss={logs['policy_loss'][-1]:.4f}, "
                f"critic_loss={logs['critic_loss'][-1]:.4f}, "
                f"entropy={logs['entropy_loss'][-1]:.4f}"
            )
            
            logs['reward'].append(tensordict_data['next','reward'].mean().item())
            pbar.update(tensordict_data.numel())
            self.trained_frames += tensordict_data.numel()
            self.start_iteration = i + 1
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(
                tensordict_data["next", "envsteps"].max().item()
            )
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs['lr'].append(self.optim.param_groups[0]['lr'])
            lr_str = f"lr policy : {logs['lr'][-1]: 4.4f}"
            if i > 0 and i % 10 == 0 :
                self.save_checkpoint(self.checkpoint_path)
                #torch.cuda.empty_cache()
                '''with set_exploration_type(ExplorationType.DETERMINISTIC),torch.no_grad():
                    eval_rollout = self.transformed_env.rollout(250,self.action_module)
                    logs['eval reward'].append(eval_rollout['next','reward'].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                logs["eval step_count"].append(eval_rollout["envsteps"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout'''
            #logger.info("||".join([eval_str, cum_reward_str, stepcount_str, lr_str,ppo_str]))
            self.log_metrics(logs['reward'][-1],logs['reward'][0],logs['step_count'][-1],logs['lr'][-1],logs['policy_loss'][-1],logs['critic_loss'][-1],logs['entropy_loss'][-1])
            #pbar.set_postfix_str("||".join([eval_str, cum_reward_str, stepcount_str, lr_str,ppo_str]))
            self.scheduler.step()


        



        


