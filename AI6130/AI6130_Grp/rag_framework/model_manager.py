import torch
import gc
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import dspy

from .utils import(
    init_env,
    logger,
    clear_memory,
    detect_environment
) 

from rag_system import RAGSystem

class ModelManager():
    def __init__(self, default_quant_bits = 4, mem_limit = 0.1):
 
        self. models = {
        'tiny': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'gemma': 'google/gemma-2-9b-it',
        'mistral': 'mistralai/Mistral-7B-v0.1'
        }

        self.model_sizes = {
        'tiny': 2.2,    # ~1.1B parameters
        'gemma': 18.0,  # ~9B parameters  
        'mistral': 90.0 # ~47B parameters (8x7B architecture)
        }   

        self.quantization_strategy = {
            'tiny': None,
            'gemma': '4bit',
            'mistral': '4bit'
        }      
        
        self.rag_system = None
        self.prompt_templates = None
        env_info = init_env()
        
        if env_info is None:
            logger.error('Environment initialization failed. ModelManager cannot proceed')
            raise RuntimeError("Environment setup failed")
        else:

            self.device = env_info['device']
            self.main_dir = env_info['main_dir']
            self.log_dir = env_info['log_dir']
            self.papers_dir =env_info['papers_dir']
            self.models_dir = env_info['models_dir']
            self.reference_dir = env_info ['reference_dir']
        
        
        self.model_results_dirs = {}
        for model_key in self.models.keys():
            model_results_dir = self.main_dir / "results" / model_key
            model_results_dir.mkdir(parents=True, exist_ok=True)
            self.model_results_dirs[model_key] = model_results_dir
            logger.info(f"Created results directory for {model_key}:{model_results_dir}")

        logger.info(f"ModelManager initialized with device:{self.device}")
        logger.info(f"Available models:{list(self.models.keys())}")
    
    def _create_quantization_strategy(self, model_key: str) -> Optional[BitsAndBytesConfig]:
        #define quantization based on model size

        strategy = self.quantization_strategy.get(model_key)

        if strategy is None:
            logger.info(f"No quantization for {model_key} model")
        
        if strategy == '4bit':
            logger.info(f"Using 4bit quantization for {model_key} model")

            return  BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_compute_dtype=torch.float,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8
            )
        
        return None
    
    def _setup_device_map(self,model_key: str) -> str:
        #Setup device mapping based on available hardware and model size
        model_size = self.model_sizes[model_key]

        if self.device == "cuda":

            if model_size > 5.0:
                return "auto"
            else:
                return "cuda"
            
        elif self.device == "mps":
            return "mps"
        else:
            return "cpu"

    def _setup_dspy_lm(self, model, tokenizer, model_key:str):
        try:
            self.dspy_lm = dspy.HFModel(
                model=model,
                tokenizer=tokenizer,
                model_name=model_key
            )
                

            dspy.settings.configure(lm=self.dspy_lm)
            logger.info(f"DSPy language model configured for {model_key}")

        except Exception as e:
            logger.warning(f"Failed to setup DSPy for {model_key}:str(e)")
            logger.info("continuing without DSPy integration")
        
        def unload_curr_model():

            if self.current_model is None:
                logger.info ("No model currently loaded")
                return
            
            logger.info ("Unloadeding model: {self.current_model_name}")

            #Clear DSPy config
            dspy.settings.configure(lm=None)
            delattr(self, 'dspy_lm')

            del self.current_model
            del self.current_tokenizer

            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            self.current_device = None

            clear_memory()
            logger.info("Model unloaded and memory cleared")

    def get_current_model_info(self) ->Dict[str, any]:
        if self.current_model is None:
            return {"status": "no_model_loaded"}
        
        return{
                "status": "model_loaded",
                "model_name": self.current_model_name,
                "model_path": self.models[self.current_model_name],
                "model_size": self.model_sizes[self.current_model_name],
                "quantization": self.quantization_strategies[self.current_model_name],
                "device": self.current_device
            }

    def is_model_loaded(self, model_key: str = None) -> bool:
        if model_key is None:
            return Fsldr
        return 

    def load_model(self, model_key: str) -> bool:
        #load specified model with appropriate quantization strategy
        
        if model_key not in self.models:
            logger.error(f"Unknown model key passed: {model_key}")
            return False

        #unload model before loading new one
        if self.current_model is not None:
            self.unload_current_model()
        
        model_path = self.models[model_key]
        logger.info(f"Loading model: {model_key} {model_path}")

        try:
            #create quantization config
            quantization_config = self._create_quantization_strategy(model_key)
            device_map = self._setup_device_map(model_key)

            #load tokenizer
            logger.info(f"Load tokenizer for {model_key}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            #handle tokenizer padding
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info (f"loading model {model_key} with device_map: {device_map}")

            model_kwargs = {
                'pretrained_model_name_or_path': model_path,
                'device_map': device_map,
                'torch_dtype': torch.float16,
                'trust_remote_code': True
            }

            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config
                logger.info(f'Using quantizatin for {model_key}')

            model = AutoModelForCausalLM(**model_kwargs)

            self._setup_dspy_lm(model, tokenizer, model_key)
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_key
            self.current_device = device_map

            logger.info(f"Successfully loaded {model_key}")
            logger.info(f"Model device: {next(model.parameters()).device}")

            return True
        except Exception as e:
            logger.error(f"Failed to load model{model_key}: {str(e)}")
            clear_memory()
            return False

    def setup_rag_system(self, papers_dir)->bool:

        
        papers_loaded = False
        success = False

        self.rag_system = RAGSystem(
            embedding_model_name = "BAAI/bge-small-en-v1.5",
            chunk_size = 1500,
            chunk_overlap = 300
        )

        papers_loaded = self.rag_system.ingest_and_index(papers_dir=papers_dir)

        if papers_loaded:
            success = True
            logger.info ("RAG System successfully initialized and papers were successfully ingested and indexed")
        else:
            logger.error("Error while loading and indexing papers. Please check log on issue")
            success = False

        return success

    def get_rag_system(self):
        # Check if RAG system is properly initialized
        if hasattr(self, 'rag_system') and self.rag_system is not None:
            if self.rag_system.index is not None:
                return self.rag_system
            else:
                logger.error("RAG system exists but index is not built")
                return None
        else:
            logger.error("RAG system not initialized")
            return None    

    
    def create_prompt_templates(self) -> Dict:
        templates = {}

        logger.info("Using DSPy templates for prompt structuring")

        class SummarizationTemplate(dspy.Signature):
            context = dspy.InputField(desc="Retrieved text chunks from the scientific paper")
            paper_name = dspy.InputField(desc="Name of the paper being summarized")
            summary = dspy.OutputField("Comprehensive summary covering key concepts, formulas and implications")
        
        class QATemplate(dspy.Signature):
            context = dspy.InputField(desc="Retrieved text chunks relevant to the question")
            question = dspy.InputField(desc="Question to be answered")
            answer = dspy.OutputField(desc="Detailed answer based on the context provided")

        templates['summarization'] = SummarizationTemplate
        templates['qa'] = QATemplate
        
        # Store templates as instance variable
        self.prompt_templates = templates
        logger.info(f"Created prompt templates")       

        return templates
    
    def generate_summary(self, paper_name, context_chunks):

        summary_text =""
        success = False

        if not paper_name or not context_chunks:
            logger.error ("Invalid inputs: paper name or context chunks empty")
            return summary_text, success
        
        try:
            #extract text content from chunk
            context_text = []

            for chunk in context_chunks:
                if isinstance(chunk,dict) and 'content' in chunk:
                    context_text.append(chunk['content'])
                elif isinstance(chunk, str):
                    context_text.append(chunk)
            
            combined_context = "\n\n".join(context_text)            
            
            #Check if model is loaded
            if not hasattr(self, 'current_model') or self.current_model is None:
                logger.error("No model currently loaded")
                return summary_text, success
            
            #Get summarization prompt template
            if not hasattr(self, 'prompt_templates') or 'summarization' not in self.prompt_templates:
                logger.error("Summarization template not available")
                return summary_text, success                

            summarize_module = dspy.Predict(self.prompt_templates['summarization'])
            response = summarize_module(context=combined_context, paper_name=paper_name)
            summary_text = response.summary

            #Clean and format summary text
            summary_text = summary_text.strip()
            success = True
            logger.info(f"Successfully generated summary for {paper_name}")

        except Exception as e:
            logger.error(f"Error generating summary for {paper_name}: {str(e)}")

        return summary_text, success
    
    def generate_qa_response(self, question, context_chunks ):
        answer_text = ""
        success = False

        if not question  or not context_chunks:
            logger.error ("Invalid inputs: paper name or context chunks empty")
            return answer_text, success
        
        try:
            
            context_text = []

            for chunk in context_chunks:
                if isinstance(chunk, dict) and 'content' in chunk:
                    context_text.append(chunk['content'])
                elif isinstance(chunk, str):
                    context_text.append(chunk)

            #combine context chunks into single context string
            combined_context = "\n\n".join(context_text) 
            
            #Check if model is loaded
            if not hasattr(self, 'current_model') or self.current_model is None:
                logger.error("No model currently loaded")
                return answer_text, success
            
            # Get QA prompt template
            if not hasattr(self, 'prompt_templates') or 'qa' not in self.prompt_templates:
                logger.error("QA template not available")
                return answer_text, success
            
            #Use DSPy template

            qa_module = dspy.Predict(self.prompt_templates['qa'])
            response = qa_module(context=combined_context, question=question)
            answer_text = response.answer

            #Clean and format answer text
            answer_text = answer_text.strip()
            success = True
            logger.info("Successfully generated answer for question {question[:50]}...")

        except Exception as e:
            logger.error(f"Error generating summary for {question}: {str(e)}")


        return answer_text, success   


    def process_paper(paper_name, qa_questions=None):

        return
def run_evaluation_pipeline(papers_dir, qa_questions_dict=None):
    all_results = {}
    models_to_test = self.model_configs.keys()

    return all_results

    

        


    