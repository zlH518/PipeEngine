import os
import queue
import asyncio
from typing import Any, List

import ray
from ray.util.placement_group import placement_group
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.logging_utils import init_logger

from .utils import get_bundle_indices, ray_noset_visible_devices

from tracer import tracepoint_module_setup, TracePoint

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


class BaseLLMRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        kwargs.pop("agent_func_path", None)
        noset_visible_devices = ray_noset_visible_devices()
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.requests = {}
        self.response_queues = queue.Queue()

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.kwargs = kwargs
        self.task_id = kwargs.pop("task_id")

        import vllm

        if vllm.__version__ >= "0.9.0":
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        os.environ["GLOBAL_RANK"] = str(kwargs.pop("offset"))
        tracepoint_module_setup()


@ray.remote
class LLMRayActor(BaseLLMRayActor):
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        import vllm
        tp = TracePoint(f"m-{self.task_id}: init-vllm.LLM")
        tp.begin()
        self.llm = vllm.LLM(*args, **self.kwargs)
        tp.end()

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        tp = TracePoint(f"{self.task_id}: wake-up-engine")
        tp.begin()
        self.llm.sleep(level=level)
        tp.end()

    def wake_up(self):
        tp = TracePoint(f"{self.task_id}: wake-up-engine")
        tp.begin()
        self.llm.wake_up()
        tp.end()

    def add_requests(self, sampling_params, prompt_token_ids):
        """
        Process requests from rank0 and generate responses.
        Since only rank0 will send requests, we don't need to track actor ranks.
        """
        tp = TracePoint(f"{self.task_id}: put-request-in-queue", "1")
        tp.begin()
        from vllm.inputs import TokensPrompt

        requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
        responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
        self.response_queues.put(responses)
        tp.end()

    def get_responses(self):
        """
        Return the responses for the actor with the given rank
        """
        tp = TracePoint(f"{self.task_id}: get-responses", "1")
        tp.begin()
        ans = self.response_queues.get()
        tp.end()
        return ans

def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg= tuple[PlacementGroup, list[int]],
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    llm_actor_cls=LLMRayActor,
    agent_func_path=None,
    offset: int =None,
    task_id: int=None,
):
    assert task_id is not None, "task id must be set"
    tracepoint_module_setup()
    import vllm

    assert vllm.__version__ > "0.8.2", "OpenRLHF only supports vllm > 0.8.2"

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = 0.5
    assert shared_pg is not None

    shared_pg, reordered_bundle_indices = shared_pg       

    for i in range(num_engines):
        tp = TracePoint(f"m-{task_id}: init-engine", "1")
        tp.begin()
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[int(i*num_gpus)],
        )

        vllm_engines.append(
            llm_actor_cls.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                full_determinism=full_determinism,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.5,
                enable_sleep_mode=vllm_enable_sleep,
                agent_func_path=agent_func_path,
                offset=offset,
                task_id=task_id,
            )
        )
        tp.end()

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


async def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return refs
