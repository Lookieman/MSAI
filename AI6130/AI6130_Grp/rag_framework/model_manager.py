import torch
import gc
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

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
            model_results_dir = Path(self.main_dir / "results" / model_key)
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

    def _load_qa_questions(self, paper_name: str) -> list :

        qa_questions = []
        qa_file_path = None

        try:
            qa_file_path = self.reference_dir / "qa_pairs.json"
        
            if not qa_file_path.exists():
                logger.warning(f"QA pairs file not found: {qa_file_path}")
                return qa_questions
            #Load QA pairs from json file

            with open(qa_file_path, 'r', encoding = 'utf-8') as file:
                qa_data = json.load(file)
            
            paper_key = paper_name.replace('.pdf', '')

            #look for matching paper in QA data
            if paper_key in qa_data:
                qa_pairs = qa_data[paper_key]
                #Extract question
                qa_questions = [pair.get('question', '') for pair in qa_pairs if 'question' in pair]
                logger.info(f"Loaded {len(qa_questions)} QA questions for {paper_name}")
            else:
                logger.warning(f"No QA questions found for paper key: {paper_key}")

        except Exception as e:
            logger.error(f"Error loading QA questions for {paper_name}: {str(e)}")

        return qa_questions

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
            return self.current_model is not None
        return self.current_model == model_key
        

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

    def get_rag_system(self)-> RAGSystem | None:
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

    
    def create_prompt_templates(self) -> list:
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
    
    def generate_summary(self, paper_name, context_chunks) -> tuple[str, bool]:

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
    
    def generate_qa_response(self, question, context_chunks ) -> str:
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


    def process_paper(self, paper_name, qa_questions=None) -> Dict[str, any]:
        
        #Initialize variables
        paper_results = {"summarization":{},"qa":{}}
        rag_system = None
        summary_context = []

        #Get RAG System instance
        rag_system = self.get_rag_system()
        if rag_system is None:
            logger.error(f"RAG system not available for processing {paper_name}")
            return paper_results
        
        try:
            logger.info(f"Processing paper: {paper_name}")
            
            # Process Summarization Task
            logger.info(f"Generating summary for {paper_name}")

            #Create generic summarization query
            summary_query = "Please summarize the paper for me that describes the main concepts described here"

            # Retrieve relevant context using RAG system (top_k=5 for comprehensive summary)
            summary_context = rag_system.query(summary_query, top_k=5)

            if summary_context:
                #Generate summary using context
                summary_text, summary_success = self.generate_summary(paper_name, summary_context)
            
                #store summary result 
                paper_results['summarization'] ={
                    "paper_name": paper_name,
                    "query": summary_context,
                    "context_chunks_count": len(summary_context),
                    "summary": summary_text,
                    "success": summary_success,
                    "context_sources":  [chunk.get('metadata', {}) for chunk in summary_context],
                    "retrieved_contexts": [chunk.get('content', '') for chunk in summary_context]
                }

                if summary_success:
                    logger.info(f"Successfully generated summary for {paper_name}")
                else:
                    logger.warning(f"Summary generation failed for {paper_name}")    
            else:
                logger.warning(f"No context retrieved for summarization of {paper_name}")
                paper_results["summarization"]={
                    "paper_name": paper_name,
                    "query": summary_query,
                    "context_chunks_count": 0,
                    "summary": "",
                    "success": False,
                    "context_sources": [],
                    "retrieved_contexts": []
                }
            
            #Process QA task - load question from json

            if qa_questions is None:
                qa_questions = self._load_qa_questions(paper_name)
            
            if qa_questions and len(qa_questions) > 0:
                logger.info(f"Processing {len(qa_questions)} QA questions for {paper_name}")
                qa_results = []

                for question in qa_questions:
                    try:

                        #Retrieve relavant context for specific questions
                        qa_context = rag_system.query(question, top_k=3)

                        if qa_context:
                            #Generate answer using retrieved context
                            answer_test, qa_success = self.generate_qa_response(question, qa_context)
                        
                            #Store individual QA result
                            qa_results = {
                                "question": question,
                                "answer": answer_test,
                                "success": qa_success,
                                "context_chunks_count": len(qa_context),
                                "context_sources": [chunk.get('metadata', {}) for chunk in qa_context],
                                "retrieved_contexts": [chunk.get('content', '') for chunk in qa_context]
                            }

                            qa_results.append(qa_results)
                            
                            if qa_success:
                                logger.info(f"Successfully answered question: {question[:50]}...")
                            else:
                                logger.warning(f"Failed to answer question: {question[:50]}...")
                        else:
                            logger.warning(f"No context retrieved for question: {question[:50]}...")
                            qa_results.append({
                                "question": question,
                                "answer": "",
                                "success": False,
                                "context_chunks_count": 0,
                                "context_sources": [],
                                "retrieved_contexts": []
                            })
                    except Exception as e:
                        logger.error(f"Error processing question '{question[:50]}...': {str(e)}")
                        qa_results.append({
                            "question": question,
                            "answer": "",
                            "success": False,
                            "error": str(e),
                            "context_chunks_count": 0,
                            "context_sources": [],
                            "retrieved_contexts": []
                        })
                    
                paper_results["qa"] = qa_results

                successful_qa = sum(1 for result in qa_results if result["success"])
                logger.info(f"Completed QA processing for {paper_name}: {successful_qa}/{len(qa_questions)} questions answered successfully")

            else:
                logger.info(f"No QA questions found for {paper_name}")
            logger.info(f"Completed processing paper: {paper_name}")

        except Exception as e:     
            logger.error(f"Error processing paper {paper_name}: {str(e)}")
            paper_results["error"] = str(e)           
                   
        return paper_results
    

    def run_evaluation_pipeline(self, qa_questions_dict=None)->list:
        
        all_results = {}
        models_to_test = self.model_configs.keys()
        pdf_files = []
        rag_setup_success = False

        #Initialize results
        model_results = {}
        model_load_success = False

        logger.info("Starting RAG evaluation pipeline")

        logger.info("Setting up RAG system...")
        rag_setup_success = self. setup_rag_system(self.papers_dir)

        if not rag_setup_success:
            logger.error("Rag system setup failed. Pipeline cannot proceed")
            return all_results
    
        #Create prompt templates
        self.create_prompt_templates()

        #Get pdf files from paper_dir
        try:
            
            pdf_files = [f.name for f in self.papers_dir.glob("*.pdf")]
            if not pdf_files:
                logger.error(f"No PDF files found in {self.papers_dir}")
                return all_results
            logger.info(f"Found {len(pdf_files)} PDF files to process: {pdf_files}")
        except Exception as e:
             logger.error(f"Error accessing papers directory {self.papers_dir}: {str(e)}")
             return all_results

        #Process model

        for model_key in models_to_test:
            logger.info(f"Starting evaluation with model: {model_key}")

            try:
                logger.info(f"Loading model: {model_key}")
                model_load_success = self.load_model(model_key)

                if not model_load_success:
                    logger.error(f"Failed to load model {model_key}. Skipping to next model.")
                    all_results[model_key] = {"error": "Model loading failed"}
                    continue
                for paper_file in pdf_files:
                    logger.info(f"Processing {paper_file} with model {model_key}")

                    try:
                        paper_name = paper_file.replace('.pdf', '') #remove extension to get paper name

                        #Get Q/A questions
                        if qa_questions_dict and paper_name in qa_questions_dict:
                            qa_questions = qa_questions_dict[paper_name]
                        
                        #Process paper
                        paper_results = self.process_paper(paper_file, qa_questions)

                        #Store results for this paper
                        model_results[paper_file] = paper_results

                        logger.info(f"Completed processing {paper_file} with model {model_key}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {paper_file} with model {model_key}: {str(e)}")
                        model_results[paper_file] = {"error": str(e)}
                        continue
                #log summary for model
                all_results[model_key] = model_results

                #Log summary for model
                successful_papers = sum(1 for paper_result in model_results.values() 
                                      if "error" not in paper_result)
                logger.info(f"Model {model_key} completed: {successful_papers}/{len(pdf_files)} papers processed successfully")
            except Exception as e:
                logger.error(f"Error with model {model_key}: {str(e)}")   
                all_results[model_key] = {"error": str(e)} 
            
            finally:
                #unload current model
                if hasattr(self, 'current_model') and self.current_device is not None:
                    logger.info(f"Unloading model {model_key}")
                    self.unload_current_model()
        
        #log pipeline completion summary
        successful_models = sum(1 for model_result in all_results.values() 
                               if "error" not in model_result)
        logger.info(f"Pipeline completed: {successful_models}/{len(models_to_test)} models processed successfully")

        return all_results

    def save_results(self, all_results, timestamp=None) -> bool:

        save_success = False
        total_files_saved = 0
        failed_saves = 0

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("Starting to save evaluation results")

        try:
            #Process results for each model

            for model_key, model_results in all_results.items():
                logger.info(f"Saving results for model: {model_key}")

                #Skip if model has errors
                if "error" in model_results and len(model_results) == 1:
                    logger.warning(f"Skipping save for model {model_key} due to errors")
                    continue
                
                model_dir = Path(self.main_dir / "results" / model_key)

                #Initialize  model summary data
                model_summary ={
                    "model_name": model_key,
                    "model_path": self.models[model_key],
                    "timestamp": timestamp,
                    "total_papers": 0,
                    "successful_papers": 0,
                    "papers_processed": []
                }

                #Save individualised paper results
                for paper_file, paper_results in model_results.items():
                    if "error" in paper_results and len(paper_results) == 1:
                        logger.warning(f"Skipping save for {paper_file} due to errors")
                        continue
                    
                    try:

                        #Create safe file names
                        paper_name = paper_file.replace('.pdf', '')
                        safe_paper_name = paper_name.replace(' ', '_').replace('/', '_')

                        #Save summarization results
                        if 'summarization' in paper_results:
                            summary_filename = f"{model_key}_{safe_paper_name}_summary_{timestamp}"
                            summary_filepath = model_dir / summary_filename

                            summary_data = {
                                "model_name": model_key,
                                "paper_name": paper_file,
                                "timestamp": timestamp,
                                "task": "summarization",
                                "results": paper_results["summarization"]
                            }

                            with open(summary_filepath, 'w', encoding='utf-8') as f:
                                json.dump(summary_data, f, indent=2, ensure_ascii=False)
                            
                            logger.debug(f"Saved summary results: {summary_filepath}")
                            total_files_saved += 1

                        #Save QA results
                        if "qa" in paper_results and paper_results["qa"]:
                            qa_filename = f"{model_key}_{safe_paper_name}_qa_{timestamp}.json"
                            qa_filepath = model_dir/ qa_filename

                            qa_data = {
                                "model_name": model_key,
                                "paper_name": paper_file,
                                "timestamp": timestamp,
                                "task": "qa",
                                "results": paper_results["qa"]
                            }

                            with open(qa_filepath, 'w', encoding='utf-8') as f:
                                json.dump(qa_data, f, indent=2, ensure_ascii=False)
                            
                            logger.debug(f"Saved QA results: {qa_filepath}")
                            total_files_saved += 1
                        
                        #Update model summary
                        model_summary["total_papers"] +=1
                        if paper_results.get("summarization", {}).get("success", False) or \
                            any(qa.get("success", False) for qa in paper_results.get("qa", [])):
                            model_summary['successful_papers'] +=1
                        
                        model_summary["papers_processed"].append({
                            "paper_name": paper_file,
                            "summarization_success": paper_results.get("summarization", {}).get("success", False),
                            "qa_questions_count": len(paper_results.get("qa", [])),
                            "qa_successful_count": sum(1 for qa in paper_results.get("qa", []) if qa.get("success", False))
                        })
                        
                    except Exception as e:
                        logger.error(f"Error saving results for {paper_file} with model {model_key}: {str(e)}")
                        failed_saves += 1
                        continue
                #Save model summary
                try:
                    summary_filename = f"{model_key}_model_summary_{timestamp}.json"
                    summary_filepath = model_dir / summary_filename

                    with open(summary_filepath, 'w', encoding='utf-8') as f:
                        json.dump(model_summary, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved model summary: {summary_filepath}")
                    total_files_saved += 1

                except Exception as e:
                    logger.error(f"Error saving model summary for {model_key}: {str(e)}")
                    failed_saves += 1

                logger.info(f"Completed saving results for model {model_key}")

            #Check overall success
            if failed_saves == 0:
                save_success = True
                logger.info(f"Successfully saved all results: {total_files_saved} files saved")
            else:
                logger.warning(f"Partial save success: {total_files_saved} files saved, {failed_saves} failed")
                save_success = False

        except Exception as e:
            logger.error(f"Critical error during save operation: {str(e)}")
            save_success = False
            
        return  save_success  

    

        


    