import ray
import asyncio

from openrlhf.utils import SingletonMeta
from openrlhf.trainer.ray.create_placement_group import create_placement_groups

from .task import Task

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


    async def init_tasks(self):
        # breakpoint()
        for task in self.tasks:
            await task.init_trainer(self.locks)
            task.init_model()


    async def run(self):
        print("Starting concurrent training tasks...")

        await asyncio.gather(*[task.train() for task in self.tasks])

        print("All concurrent training tasks finished.")