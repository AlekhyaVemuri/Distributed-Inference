# Distributed Inference for Diffusers

## Introduction
Distributed inference in PyTorch involves deploying a diffusers model across multiple devices(targetting multi-GPUs in this sample) or even multiple machines to accelerate the inference process, especially for large models or high-throughput scenarios.

### Key approaches to distributed inference in PyTorch:
- Data Parallelism
- Model Parallelism

## Data Parallelism
On distributed setups, you can run inference across multiple GPUs with [Huggingface Accelerate](https://huggingface.co/docs/accelerate/index) or [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html), which is useful for **generating with multiple prompts in parallel**.

### Huggingface Accelerate
- Simplifies the process of setting up the distributed environment
- To begin, create a Python file and initialize an accelerate.PartialState to create a distributed environment.
- Your setup is automatically detected so you don’t need to explicitly define the rank or world_size.
- Move the DiffusionPipeline to distributed_state.device to assign a GPU to each process.
- Now use the split_between_processes utility as a context manager to automatically distribute the prompts between the number of processes.
  - Pass a list of prompts
- Run the below command:
  - ```bash
    uv run accelerate test.py --num-processes=2 
    ```
    OR
    ```bash
    uv run accelerate test.py 
    ```
    > If accelerate package is configured by default with available number of GPUs, --num-processes parameter takes the default value.

### PyTorch Distributed


## Model Parallelism or Model Sharding
- Model sharding is a technique that distributes models across GPUs when the models don’t fit on a single GPU.
- The `balanced` strategy evenly distributes the model on all available GPUs.
  > device_map = "balanced"
![Screenshot 2025-06-24 173837](https://github.com/user-attachments/assets/86df8e78-0392-4a2b-96e2-ad62635ec272)
