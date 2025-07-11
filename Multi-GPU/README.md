# Distributed Inference on Multi-GPUs using PyTorch XPU Backend

## Introduction
As deep learning models grow in complexity and demand, distributed inference offers a powerful solution to optimize performance, scale inference capabilities, and handle high-throughput workloads. 
By distributing the model and/or the inference workload across multiple GPUs, either on a single machine or across a cluster of machines, this approach enables faster predictions and the ability to serve larger models that might otherwise exceed the memory capacity of a single XPU.

This sample focuses on common approaches and tools used for multi-GPU distributed inference, specifically leveraging PyTorch's XPU backend.

## Contents
- [Key Approaches](./README.md#key-approaches)
- [Data Parallelism](./README.md#data-parallelism)
  - [Huggingface Accelerate](./README.md#hugging-face-accelerate)
    - [Run the `Data Parallelism` Sample](./README.md#run-the-data-parallelism-sample)
- [Model Parallelism or Model Sharding](./README.md#model-paralelllism-or-model-sharding)
  - [Run the `Model Parallelism` Sample](./README.md#run-the-model-parallelism-sample)
- [XPU Usage](./README.md#xpu-usage)
- [License](./README.md#license)

## Key Approaches
- Data Parallelism
- Model Parallelism

## Data Parallelism
In distributed setups, you can perform inference across multiple GPUs using either [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). Data parallelism is particularly useful for **generating outputs with multiple prompts in parallel**.

<img alt="image" width=600 src="./assets/data-parallelism-workflow.png"/>

### Hugging Face Accelerate
- Simplifies the process of setting up and managing the distributed environment.
- To begin, create a Python script and initialize an `accelerate.PartialState` object to define your distributed environment.
- Your hardware setup is automatically detected, eliminating the need to explicitly define `rank` or `world_size` parameters.
- Move your `DiffusionPipeline` or model to `distributed_state.device` to ensure it's assigned to the correct XPU for each process.
- Utilize the `split_between_processes` utility as a context manager. This automatically distributes a given list of prompts among the available processes.

#### **Run the `Data Parallelism` Sample:**
1. Configure accelerate with the following command:
   ```bash
    uv run accelerate config
   ```
   1. Choose Multi-GPU and appropriate options for correct configuration as shown below:
      
    <img alt="image" width=600 src="./assets/accelerate-config.png"/>
    
2. Run the sample:
    ```bash
    uv run accelerate launch data-parallelism-accelerate.py --num-processes=2
    ```
    OR (if `accelerate` is configured to use all available GPUs by default)
    ```bash
    uv run accelerate launch data-parallelism-accelerate.py
    ```
    > **Note:** The `--num-processes` parameter can often be omitted if `accelerate` is pre-configured to utilize all available GPUs. 
3. Output are generated in parallel as multiple prompts are passed.

    <img alt="image" width=600 src="./assets/data-parallelism-output.png"/>
   

## Model Parallelism or Model Sharding
Model sharding is a crucial technique for distributing large models across multiple XPUs when their entire footprint does not fit on a single XPU's memory.

<img alt="image" width=600 src="./assets/model-parallelism-workflow.png"/>

- For Hugging Face models, a common strategy for model sharding is to use `device_map="balanced"`.
- This configuration intelligently distributes the model's layers across all available XPUs to ensure an even load and optimize memory usage.
 <img alt="image" width=600 src="./assets/model-sharding.png"/> 
!

### Run the `Model Parallelism` Sample:
Commands to run:
```bash
uv run model-parallelism-flux.py
```

## XPU Usage:
- XPU Usage for Data Parallelism using Accelerate
  
  <img alt="image" width=600 src="./assets/data-parallelism-xpu.png"/>
  
- Model Sharding
  
  <img alt="image" width=600 src="./assets/model-sharding-xpu.png"/>


## License:
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.
Third party program Licenses can be found here: [third-party-programs.txt.](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

