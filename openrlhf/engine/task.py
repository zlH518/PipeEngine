import ray
import asyncio

from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
    RewardModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.ppo_critic import CriticModelActor
from openrlhf.utils import get_strategy

from tracer import  tracepoint_module_setup, TracePoint

class Task:
    """
    Task class for training a single task.
    It contains:
    - vllm_engines
    - actor_model
    - critic_model
    - ref_model
    - reward_model
    """
    def __init__(self, args, task_id, pgs, task_num):
        self.args = args
        self.task_id = task_id
        self.pgs = pgs
        self.task_num = task_num
        self.strategy = get_strategy(args)
        self.strategy.print(args)
        self.offset:dict = {
            "actor": 0,
            "vllm": 1,
            "critic": None,
            "ref": 2,
            "reward": 3
        }

    
    async def init_task(self, locks):
        args = self.args
        self.actor_model = RayActorGroup(
            self.task_id,
            args.actor_num_nodes,
            args.actor_num_gpus_per_node,
            PolicyModelActor,
            pg=self.pgs["actor"],
            num_gpus_per_actor=0.5,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
            offset = self.offset["actor"]
        )

        # init vLLM engine for text generation
        self.vllm_engines = None
        if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
            max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            if args.colocate_all_models and not args.async_train:
                assert (
                    args.actor_num_nodes * args.actor_num_gpus_per_node
                    == args.vllm_num_engines * args.vllm_tensor_parallel_size
                ), (
                    f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                    f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                    f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
                )

            if args.agent_func_path:
                from openrlhf.trainer.ray.vllm_engine_async import LLMRayActorAsync as LLMRayActor
            else:
                from openrlhf.trainer.ray.vllm_engine import LLMRayActor

            self.vllm_engines = await create_vllm_engines(
                args.vllm_num_engines,
                args.vllm_tensor_parallel_size,
                args.pretrain,
                args.seed,
                args.full_determinism,
                args.enable_prefix_caching,
                args.enforce_eager,
                max_len,
                self.pgs["rollout"],
                args.vllm_gpu_memory_utilization,
                args.vllm_enable_sleep,
                LLMRayActor,
                args.agent_func_path,
                self.offset["vllm"],
                self.task_id,
            )

        if args.init_kl_coef <= 0:
            self.ref_model = None
        else:
            self.ref_model = RayActorGroup(
                self.task_id,
                args.ref_num_nodes,
                args.ref_num_gpus_per_node,
                ReferenceModelActor,
                pg=self.pgs["ref"],
                num_gpus_per_actor=0.5,
                duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
                offset=self.offset["ref"]
            )


        if args.critic_pretrain:
            self.critic_model = RayActorGroup(
                self.task_id,
                args.critic_num_nodes,
                args.critic_num_gpus_per_node,
                CriticModelActor,
                pg=self.pgs["critic"],
                num_gpus_per_actor=0.5,
                duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
                offset=self.offset["critic"]
            )
        else:
            self.critic_model = None

        # multiple reward models
        if not args.remote_rm_url:
            self.reward_pretrain = args.reward_pretrain
            self.reward_model = RayActorGroup(
                self.task_id,
                args.reward_num_nodes,
                args.reward_num_gpus_per_node,
                RewardModelActor,
                pg=self.pgs["reward"],
                num_gpus_per_actor=0.5,
                duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
                offset=self.offset["reward"]
            )
        else:
            self.reward_model = None
        
        self.ppo_trainer=None
        tracepoint_module_setup()


        tp = TracePoint(f"m-{self.task_id}: init-trainer", "1")
        tp.begin()
        if self.args.async_train:
            from openrlhf.trainer.ppo_trainer_async import PPOTrainerAsync as PPOTrainer
        else:
            from openrlhf.trainer.ppo_trainer import PPOTrainer
        # init PPO trainer (Single controller)
        self.ppo_trainer = PPOTrainer.remote(
            self.task_id,
            locks,
            self.args.pretrain,
            self.strategy,
            self.actor_model,
            self.critic_model,
            self.reward_model,
            self.ref_model,
            self.vllm_engines,
            prompt_split=self.args.prompt_split,
            eval_split=self.args.eval_split,
            # generate kwargs
            do_sample=True,
            prompt_max_len=self.args.prompt_max_len,
            max_new_tokens=self.args.generate_max_len,
            max_length=self.args.max_len,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
        )
        # training update steps
        self.max_steps = ray.get(self.ppo_trainer.get_max_steps.remote())
        tp.end()
        tp = TracePoint(f"m-{self.task_id}: init-model", "1")
        tp.begin()
        # init reference/reward/actor model
        refs = []
        if self.ref_model is not None:
            refs.extend(self.ref_model.async_init_model_from_pretrained(self.strategy, self.args.pretrain))
        refs.extend(self.actor_model.async_init_model_from_pretrained(self.strategy, self.args.pretrain, self.max_steps, self.vllm_engines))
        if not self.args.remote_rm_url:
            refs.extend(self.reward_model.async_init_model_from_pretrained(self.strategy, self.reward_pretrain))
        await asyncio.gather(*refs)

        if self.args.critic_pretrain:
            # critic scheduler initialization depends on max_step, so we have to init critic after actor
            # TODO: use first reward model as critic model
            refs.extend(self.critic_model.async_init_model_from_pretrained(self.strategy, self.args.critic_pretrain, self.max_steps))
            asyncio.gather(*refs)
        tp.end()

    async def train(self):
        assert self.ppo_trainer is not None, "Trainer is not initialized"
        # train actor and critic model
        tp = TracePoint(f"m-{self.task_id}: begin-train", "1")
        tp.begin()
        await self.ppo_trainer.fit.remote()
        tp.end()
        # # save model
        # await asyncio.gather(*self.actor_model.async_save_model())

        # if self.args.critic_pretrain and self.args.save_value_network:
        #     await asyncio.gather(*self.critic_model.async_save_model())


