import ray
import asyncio

from openrlhf.utils import SingletonMeta
from openrlhf.trainer.ray.create_placement_group import create_placement_groups

from tqdm import tqdm

from .task import Task
from tracer import TracePoint, tracepoint_module_setup

class PipeEngine(metaclass=SingletonMeta):
    def __init__(self, tasks_args):
        self.tasks_args = tasks_args
        self.task_num = len(tasks_args)
        self.tasks = []
        
        self.load_ckpt_lock = asyncio.Lock()
        self.rollout_lock = asyncio.Lock()
        self.make_experiences_lock = asyncio.Lock()
        self.train_actor_critic_lock = asyncio.Lock()
        self.save_train_info_lock = asyncio.Lock()
        self.locks = [self.load_ckpt_lock, self.rollout_lock, self.make_experiences_lock, self.train_actor_critic_lock, self.save_train_info_lock]

        self.pgs = create_placement_groups(self.tasks_args[0])

        self.tasks = [Task(args, task_id+1, self.pgs, self.task_num) for task_id, args in enumerate(self.tasks_args)]
        tracepoint_module_setup()


    async def init_tasks(self):
        # breakpoint()
        await asyncio.gather(*[task.init_task(self.locks) for task in self.tasks])


    async def run(self):
        await asyncio.gather(*[self.task_fit(task_id) for task_id in range(self.task_num)])


    async def task_fit(self, task_id) -> None:
        print("Starting concurrent training tasks...")

        args = await self.tasks[task_id].ppo_trainer.get_args.remote()
        print(f"Task {task_id}: Starting...")

        async with self.load_ckpt_lock:
            steps = await self.tasks[task_id].ppo_trainer.load_ckpt.remote()
            print(f"Task {task_id}: Initial steps: {steps}")

        for episode in range(0, args.num_episodes):
            tp = TracePoint(f"m-{task_id}: episode-{episode}", "1")
            tp.begin()
            prompts_dataloader = await self.tasks[task_id].ppo_trainer.get_prompts_dataloader.remote()
            pbar = tqdm(
                range(prompts_dataloader.__len__()),
                desc=f"Task {task_id} Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
                leave=False
            )

            number_of_samples = 0
            for _, rand_prompts, labels in prompts_dataloader: 
                async with self.rollout_lock:
                    tp2 = TracePoint(f"m-{task_id}: episode-{episode}--samples-{number_of_samples}", "1")
                    tp2.begin()
                    rollout_samples = await self.tasks[task_id].ppo_trainer.rollout.remote(rand_prompts, labels, number_of_samples)
                    pbar.update()

                async with self.make_experiences_lock:
                    sample0, experiences = await self.tasks[task_id].ppo_trainer.make_experiences.remote(rollout_samples)
                
                async with self.train_actor_critic_lock:
                    status = await self.tasks[task_id].ppo_trainer.train_actor_critic.remote(steps)

                async with self.save_train_info_lock:
                    await self.tasks[task_id].ppo_trainer.save_train_info.remote(status=status, args=args, steps=steps, sample0=sample0, experiences=experiences, episode=episode)

                steps += 1
                number_of_samples += 1
                tp2.end()

            pbar.close()
            print(f"Task {task_id}: Episode {episode + 1} finished, current steps: {steps}")
            tp.end()

        print(f"Task {task_id}: All episodes completed.")

        _wandb = await self.tasks[task_id].ppo_trainer.get_wandb.remote()
        _tensorboard = await self.tasks[task_id].ppo_trainer.get_tensorboard()
        if _wandb is not None:
            self.tasks[task_id].ppo_trainer.finish_wandb.remote()
        if _tensorboard is not None:
            self.tasks[task_id].ppo_trainer.close_tensorboard.remote()

        print("All concurrent training tasks finished.")