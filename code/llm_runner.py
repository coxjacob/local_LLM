# load the large language model file
from llama_cpp import Llama

llm_path = ".\Models\mistral-7b-instruct-v0.1.Q5_K_M.gguf"
LLM = Llama(model_path=llm_path)

# create a text prompt
prompt = "Q: What are the names of the days of the week? A:"

# generate a response (takes several seconds)
output = LLM(prompt)

# display the response
print(output["choices"][0]["text"])