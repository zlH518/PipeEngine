
_ROOT_LOGGER_PATH = "/root/PipeEngine_v0_add_log/experiments/log/"

def torch_available():
    """
    Check if PyTorch is available in the environment.
    
    Returns:
        bool: True if PyTorch is available, False otherwise.
    """
    try:
        import torch
        import torch.distributed

        return True
    except ImportError:
        return False


