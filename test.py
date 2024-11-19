from transformers import AutoModelForCausalLM, AutoTokenizer

model = "HuggingFaceH4/starchat-beta"
llm = HuggingFaceHub(repo_id=model ,
                         model_kwargs={"min_length":30,
                                       "max_new_tokens":500, "do_sample":True,
                                       "temperature":0.2, "top_k":50,
                                       "top_p":0.95, "eos_token_id":49155})