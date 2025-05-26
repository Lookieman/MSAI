import os
import sys
import gc
import torch
import logging
import bitsandbytes as bnb
from pathlib import Path
from typing import Optional
import torch.version
from transformers import BitsAndBytesConfig

# Create a package-level logger
logger = logging.getLogger("rag_framework")

def setup_logger(log_dir: Path, log_file: Optional[str] = "rag_system.log", 
                 log_level: int = logging.INFO) -> logging.Logger:
    #Set up and configure the package logger.

    log_file = log_dir / log_file
    
    # Configure the logger only if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter and add it to handlers
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def setup_working_dir() -> str:

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        main_dir = Path('/content/drive/MyDrive/AI6130_Grp/')
        log_dir = Path('/content/drive/MyDrive/AI6130_Grp/log')
        papers_dir = Path('/content/drive/MyDrive/AI6130_Grp/papers')
        models_dir = Path('/content/drive/MyDrive/AI6130_Grp/models')      
        reference_dir = Path('/content/drive/MyDrive/AI6130_Grp/reference')
      
    except ImportError:
        main_dir = Path('C:/Users/luqma/AI6130/AI6130_Grp/')
        log_dir = Path('C:/Users/luqma/AI6130/AI6130_Grp/log')
        papers_dir = Path('C:/Users/luqma/AI6130/AI6130_Grp/papers')
        models_dir = Path('C:/Users/luqma/AI6130/AI6130_Grp/models')
        reference_dir = Path('C:/Users/luqma/AI6130/AI6130_Grp/reference')
    main_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    
    return main_dir, log_dir, papers_dir, models_dir, reference_dir

def detect_environment():
    try:
        import google.colab
        return "Colab"
    except ImportError:
        if 'ipykernel' in sys.modules:
            return "Local "
        else:
            return "Other"
        
def clear_memory() -> bool:
    #Clear GPU/MPS memory and run garbage collection.
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

def get_device() -> str:
    #Determine the best available computational device.
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # For Apple Silicon Macs
    else:
        return "cpu"
    
def time_function(func):
    #Decorator to measure the execution time of a function
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = logging.getLogger("rag_framework")
        logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

def setup_quantization_env() -> bool:
    if not torch.cuda.is_available():
        logger.warning("Cuda not available, quantization wil not work")
        return False
    
    logger.info(f"Cuda version: {torch.version.cuda}")

    #set environment variable
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['TRANSFORMERS_CACHE'] = "content hf_cache"
    return True

def verify_ref_data(reference_dir: Path, tasks=['Summarization', 'qa']) -> bool:
    
    all_files_exist = True

    for task in tasks:
        if task == 'qa':
            qa_file = reference_dir / "qa_pairs.json"
            if not qa_file.exists():
                logger.warning(f"Missing QA Ref file:{qa_file}")
                all_files_exist = False

        else:
            summary_files = list(reference_dir.glob('*_summary.json'))

            if not summary_files.exists():
                logger.warning(f"No summary reference files found in {reference_dir}")
                all_files_exist = False
            else:
                logger.info(f" Found len{summary_files} summary ref files")

                for summary_file in summary_files:
                    logger.debug(f"found summary reference files: {summary_file}")
    
    if all_files_exist:
        logger.info (f"all necessary refernce files found in {reference_dir}")
    else:
        logger.warning("Some reference_files are missing.Check log for details")

    return all_files_exist

def init_env():
    #Initialize environment for specified model

    #Setup directory
    
    main_dir, log_dir, papers_dir, results_dir, reference_dir = setup_working_dir()

    setup_logger(log_dir)

    quant_ready = setup_quantization_env()
    ref_ready = verify_ref_data(reference_dir)

    #Check if papers_dir has content

    papers_exist = len(list(papers_dir.glob("*.pdf"))) > 0

    if not papers_exist:
        logger.warning(f"No pdf files found in {papers_dir}")
    
    env_ready = quant_ready and ref_ready and papers_exist

    if not env_ready:
        if not papers_exist:
            logger.error(f"Environment setup failed.No pdf files in {papers_dir}. Cannot proceed")
        elif not ref_ready:
            logger.error(f"Environment setup failed. Missing some reference data in {reference_dir} Cannot proceed")    
        elif not quant_ready:
            logger.error("Environment setup failed. Setup quantization failed. Please check the logs")
        return None


    return {
        'main_dir': main_dir,
        'log_dir': log_dir,
        'papers_dir': papers_dir,
        'results_dir': results_dir,
        'reference_dir': reference_dir,
        'device': get_device()
    }


