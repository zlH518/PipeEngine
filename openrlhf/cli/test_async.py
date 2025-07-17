import asyncio
import os
import time

from tqdm import tqdm
from tracer import TracePoint, tracepoint_module_setup

class Testtrainer:
    def __init__(self):
        self.load_ckpt_lock = asyncio.Lock()
        self.rollout_lock = asyncio.Lock()
        self.make_experiences_lock = asyncio.Lock()
        self.train_actor_critic_lock = asyncio.Lock()
        self.save_train_info_lock = asyncio.Lock()
        self.ppo_train_lock = asyncio.Lock()

    async def load_ckpt(self, task_id):
        async with self.load_ckpt_lock:
            tracepoint_module_setup()

            tp = TracePoint(f"load-ckpt-{task_id}", "1")
            tp.begin()
            await asyncio.sleep(5)
            steps = 0
            tp.end()
            return steps

    async def rollout(self, task_id):
        async with self.rollout_lock:
            number_of_samples_tp = TracePoint(f"number of samples", "1")
            number_of_samples_tp.begin()
            rollout_samples_tp = TracePoint(f"rollout-samples-{task_id}", "1")
            rollout_samples_tp.begin()
            await asyncio.sleep(10)
            rollout_samples = None
            rollout_samples_tp.end()
            number_of_samples_tp.end()
            return rollout_samples
    
    async def make_experiences(self, task_id):
        async with self.make_experiences_lock:
            experiences_tp = TracePoint(f"make_experiences-{task_id}", "1")
            experiences_tp.begin()
            # experiences = self.experience_maker.make_experience_batch(rollout_samples)
            await asyncio.sleep(5)
            experiences = None
            experiences_tp.end()

            # sample0 = self.tokenizer.batch_decode(
            #     experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
            # )
            sample0 = None
            print(sample0)
            save_experiences_tp = TracePoint(f"save-experiences-actor-critic-{task_id}", "1")
            save_experiences_tp.begin()
            # refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
            # if self.critic_model_group is not None:
            #     refs.extend(
            #         self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
            #     )
            # ray.get(refs)
            save_experiences_tp.end()
            return sample0, experiences
    
    async def train_actor_critic(self, task_id):
        async with self.train_actor_critic_lock:
            train_tp = TracePoint(f"train-actor-critic-{task_id}", "1")
            train_tp.begin()
            await asyncio.sleep(15)
            status = None
            # status = self.ppo_train(steps)
            train_tp.end()
            return status
    
    async def save_train_info(self, task_id):
        async with self.save_train_info_lock:
            save_info_tp = TracePoint(f"save_train_info-{task_id}", "1")
            save_info_tp.begin()
            # if "kl" in status:
            #     self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

            print(f"✨ Global step {task_id}")
            # status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]

            # logs/checkpoints
            # client_states = {
            #     "global_step": steps,
            #     "episode": episode,
            #     "data_loader_state_dict": self.prompts_dataloader.state_dict(),
            # }
            # self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)
            await asyncio.sleep(5)
            save_info_tp.end()

    async def _run_single_task(self, task_id: int):
        """
        运行单个训练任务的完整流程。
        """
        print(f"Task {task_id}: Starting...")

        steps = await self.load_ckpt(task_id)
        print(f"Task {task_id}: Initial steps: {steps}")

        for episode in range(0, 1):

            number_of_samples = 0
            rollout_samples = await self.rollout(task_id)

            sample0, experiences = await self.make_experiences(task_id)
            status = await self.train_actor_critic(task_id)
            await self.save_train_info(task_id=task_id)

            steps += 1
            number_of_samples += 1

            print(f"Task {task_id}: Episode {episode + 1} finished, current steps: {steps}")

        print(f"Task {task_id}: All episodes completed.")


    async def fit(self) -> None:
        print("Starting concurrent training tasks...")

        await asyncio.gather(
            self._run_single_task(1),
            self._run_single_task(2)
        )

        print("All concurrent training tasks finished.")


if __name__ == "__main__":
    trainer = Testtrainer()
    asyncio.run(trainer.fit())
