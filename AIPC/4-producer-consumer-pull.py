import multiprocessing
import openvino as ov
import openvino_genai as ov_genai
import huggingface_hub as hf_hub
import os
import time
import sys
from collections import defaultdict

MODEL_ID = "OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov"
MODEL_PATH = "Mistral-7B-Instruct-v0.2-int4-cw-ov" # Local directory to save the model

try:
    core = ov.Core()
    AVAILABLE_DEVICES = [d for d in core.available_devices]
    if not AVAILABLE_DEVICES:
        print("Main Process: WARNING: No suitable OpenVINO devices found. Defaulting to ['CPU'].", file=sys.stderr, flush=True)
        AVAILABLE_DEVICES = ["CPU"]
    print(f"Main Process: Discovered OpenVINO devices: {AVAILABLE_DEVICES}", flush=True)
except Exception as e:
    print(f"Main Process: ERROR: Could not get OpenVINO devices: {e}. Defaulting to ['CPU'].", file=sys.stderr, flush=True)
    AVAILABLE_DEVICES = ["CPU"]

def download_model(model_id: str, model_path: str) -> str:
    """
    Downloads the OpenVINO model from Hugging Face Hub if it doesn't already exist locally.
    """
    if not os.path.exists(model_path):
        print(f"Main Process: Downloading model {model_id} to {model_path}...", flush=True)
        try:
            hf_hub.snapshot_download(model_id, local_dir=model_path, local_dir_use_symlinks=False)
            print("Main Process: Model download complete.", flush=True)
        except Exception as e:
            print(f"Main Process: ERROR: Model download failed: {e}", file=sys.stderr, flush=True)
            sys.exit(1)
    else:
        print(f"Main Process: Model already exists at {model_path}, skipping download.", flush=True)
    return model_path

def load_openvino_pipeline(model_path: str, device_name: str):
    """
    Loads and compiles the OpenVINO LLMPipeline for a specific device.
    Enables model caching to speed up subsequent runs on the same device.
    """
    print(f"[{device_name} Consumer Init] Initializing OpenVINO on device: {device_name}", flush=True)
    
    core = ov.Core()
    available_devices = core.available_devices
    print(f"[{device_name} Consumer Init] Available OpenVINO devices from core: {available_devices}", flush=True)
    if device_name not in available_devices and device_name != "AUTO":
        print(f"[{device_name} Consumer Init] WARNING: Requested device '{device_name}' not found in available devices. This consumer might fail.", file=sys.stderr, flush=True)

    print(f"[{device_name} Consumer Init] Attempting to load LLMPipeline for {device_name}. This may take time...", flush=True)
    
    start_load_time = time.time()
    
    # Set cache directory for the specific device.
    cache_dir = f"./ov_model_cache_{device_name.lower()}"
    core.set_property(device_name, {"CACHE_DIR": cache_dir})
    print(f"[{device_name} Consumer Init] Model caching enabled at: {cache_dir}", flush=True)

    # Set NPU_TURBO for NPU speedup
    if device_name == "NPU":
        core.set_property(device_name, {"NPU_TURBO": "YES"})
        print(f"[{device_name} Consumer Init] NPU_TURBO set to YES.", flush=True)

    pipe = ov_genai.LLMPipeline(model_path, device_name)
    end_load_time = time.time()
    print(f"[{device_name} Consumer Init] Model loaded successfully on {device_name} in {end_load_time - start_load_time:.2f} seconds.", flush=True)
    return pipe

def consumer(device_name: str, model_path: str, task_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
    """
    Consumer process: loads OpenVINO pipeline and pulls tasks from a shared queue.
    """
    try:
        print(f"[{device_name} Consumer] Process starting. PID: {os.getpid()}", flush=True)
        pipe = load_openvino_pipeline(model_path, device_name)

        generation_config = ov_genai.GenerationConfig(
            max_new_tokens=100, 
            stop_strings=set(["<|endoftext|>"]),
            include_stop_str_in_output=False 
        )

        while True:
            task = task_queue.get() 
            if task is None: 
                print(f"[{device_name} Consumer] Received termination signal. Terminating.", flush=True)
                break

            item_id, info_type, prompt_text = task

            print(f"[{device_name} Consumer] Generating for {info_type} for item '{item_id}': '{prompt_text}'", flush=True)
            
            start_time = time.time()
            generated_text = pipe.generate(prompt_text, generation_config=generation_config)
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"[{device_name} Consumer] Generated {info_type} on {device_name} ({generation_time:.2f}s). Result length: {len(generated_text)}", flush=True)
            
            output_queue.put((item_id, info_type, generated_text.strip(), generation_time))

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[{device_name} Consumer] FATAL ERROR: {e}\n{error_trace}", file=sys.stderr, flush=True)
        output_queue.put(("ERROR", "ERROR", f"Consumer failed: {e}", 0))

def producer(model_path: str, devices: list, menu_items: list):
    """
    Distributes text generation tasks to a central queue for consumers to pull.
    """
    if not devices:
        print("Main Process: No devices specified for parallelism. Exiting.", file=sys.stderr, flush=True)
        return

    print(f"Main Process: OpenVINO version: {ov.__version__}", flush=True)
    print(f"Main Process: Targeting devices: {devices}", flush=True)

    task_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    processes = []
    for device in devices:
        p = multiprocessing.Process(
            target=consumer,
            args=(device, model_path, task_queue, output_queue)
        )
        processes.append(p)
        print(f"Main Process: Starting consumer process for device: {device}", flush=True)
        p.start()
        time.sleep(1) 

        if not p.is_alive():
            print(f"Main Process: WARNING: Consumer for {device} died immediately after starting. Check its logs.", file=sys.stderr, flush=True)
            if device in devices: 
                devices.remove(device) 
            print(f"Main Process: Remaining active devices: {devices}", flush=True)
    
    active_devices = [p_info[0] for p_info in zip(devices, processes) if p_info[1].is_alive()]
    if not active_devices:
        print("Main Process: No active consumer processes. Cannot distribute tasks. Exiting.", file=sys.stderr, flush=True)
        return

    all_tasks_to_send = []
    task_info_types = {
        "description": "Generate an enticing and short description for a cafe menu item: {item_name} ({item_category}).",
        "calorie_count": "Generate the approximate calorie count(e.g., '250 calories', '180 kcal') for a single serving of a cafe item: {item_name} ({item_category})."
    }

    for item in menu_items:
        item_name = item['name']
        item_category = item['category']
        item_id = f"{item_category}-{item_name}" 

        for info_type, prompt_template in task_info_types.items():
            prompt_text = prompt_template.format(item_name=item_name, item_category=item_category)
            all_tasks_to_send.append((item_id, info_type, prompt_text))

    print(f"Main Process: Populating central task queue with {len(all_tasks_to_send)} generation tasks...", flush=True)
    
    for task in all_tasks_to_send:
        task_queue.put(task)
    
    total_tasks_expected = len(all_tasks_to_send)
    raw_results = []
    print(f"Main Process: Waiting to collect {total_tasks_expected} results from consumers...", flush=True)
    
    for _ in range(total_tasks_expected):
        result = output_queue.get()
        raw_results.append(result)

    print("Main Process: All tasks processed. Signalling consumers to terminate...", flush=True)
    for _ in active_devices: 
        task_queue.put(None)

    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            print(f"Main Process: WARNING: Consumer process {p.pid} did not terminate gracefully. Terminating forcefully.", file=sys.stderr, flush=True)
            p.terminate()
            p.join()

    grouped_results = defaultdict(dict)
    for item_id, info_type, generated_text, gen_time in raw_results:
        if item_id == "ERROR":
            print(f"Main Process: Received error from worker: {generated_text}", file=sys.stderr, flush=True)
            continue
        grouped_results[item_id][info_type] = generated_text

    print("\n--- Cafe Menu Information Generated ---", flush=True)
    for item_id, info_dict in grouped_results.items():
        category, name = item_id.split('-', 1)
        print(f"\n--- {name.upper()} ({category.upper()}) ---", flush=True)
        print(f"  Description: {info_dict.get('description', 'N/A')}", flush=True)
        print(f"  Calorie Count: {info_dict.get('calorie_count', 'N/A')}", flush=True)

    print("\n--- Cafe Menu Generation Complete ---", flush=True)
    print("Main Process: Exiting main process.", flush=True)


if __name__ == "__main__":
    total_time_start = time.time()
    print("--- Starting Cafe Menu Data Generation using Data Parallelism ---", flush=True)

    cafe_menu_items = [
        {'name': 'Espresso', 'category': 'Beverage'},
        {'name': 'Blueberry Muffin', 'category': 'Food'},
        {'name': 'Iced Caramel Macchiato', 'category': 'Beverage'},
        {'name': 'Vegan Chocolate Chip Cookie', 'category': 'Food'},
        {'name': 'Spinach and Feta Croissant', 'category': 'Food'},
        {'name': 'Green Tea Latte', 'category': 'Beverage'},
        {'name': 'Avocado Toast', 'category': 'Food'},
        {'name': 'Mango Smoothie', 'category': 'Beverage'},
        {'name': 'Salmon Bagel', 'category': 'Food'},
        {'name': 'Lavender Earl Grey Tea', 'category': 'Beverage'},
        {'name': 'Gluten-Free Brownie', 'category': 'Food'},
        {'name': 'Chai Latte', 'category': 'Beverage'},
    ]

    print(f"\nUser has provided {len(cafe_menu_items)} menu items for information generation.", flush=True)
    print("Each item will generate a description and calorie count.", flush=True)
    print("These tasks will be distributed across available OpenVINO devices for parallel processing.\n", flush=True)

    model_path_downloaded = download_model(MODEL_ID, MODEL_PATH)

    producer(model_path_downloaded, AVAILABLE_DEVICES, cafe_menu_items)
    total_time_end = time.time()
    total_time=total_time_end-total_time_start
    print(f"Total Inference time: ({total_time:.2f}s)")
    
    print("--- Cafe Menu Data Generation Demonstration Complete ---", flush=True)