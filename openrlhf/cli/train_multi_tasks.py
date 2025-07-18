import ray
import copy
import asyncio

from openrlhf.utils import load_task_args
from openrlhf.engine import PipeEngine

NUM_TASKS=2

async def main():
    args = load_task_args()
    # initialize ray if not initialized
    if not ray.is_initialized():
        print("--"*50)
        # ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "RAY_DEBUG_POST_MORTEM": "1"}})
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    tasks_args = []
    tasks_args.append(args)
    tasks_args = [copy.deepcopy(args) for _ in range(NUM_TASKS)]
    pipeEngine = PipeEngine(tasks_args)
    await pipeEngine.run()


if __name__ == "__main__":
    asyncio.run(main())
    print("finish")