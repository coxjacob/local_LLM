# load the large language model file
from llama_cpp import Llama

#llm_path = "Models\mistral-7b-claude-chat.Q5_K_M.gguf"
llm_path = "Models\dolphin-2.8-mistral-7b-v02-Q8_0.gguf"
#llm_path = ".\Models\mistral-7b-instruct-v0.1.Q5_K_M.gguf"
def generate_m7b_output(query, max_tokens=1000, 
                        model_path=llm_path,
                        temp=1, n_ctx=4096,
                        verbose=False):
    #Ensure max_tokens is an integer
    max_tokens = int(max_tokens)

    #Create a LLama instance with the specified parameters
    llm = Llama(model_path=model_path, temp=temp,
                verbose=verbose)
    
    #Generate output based on the query
    output = llm(query, max_tokens=max_tokens, echo=True)

    #Extract the text from the output
    generated_text = output["choices"][0]['text']

    #Return the text from the output
    return generated_text

if __name__=="__main__":
    print("\n"*6)
    print("Jacob's_Local_LLM".center(30, " ").center(50, "#"))
    print()
    while True:
        prompt = input("(Q to quit) Querry: ")
        if prompt.lower() == 'q': 
            break
        else: 
            print("\n", ".".center(30,'-'), '\n') 
            answer = generate_m7b_output(prompt)
            list_answer = answer.split('\n')
            print(list_answer[-1])
            print("\n", ".".center(30,'-')) 
            print(".".center(30,'-'), '\n\n') 

    print('\n'*2)

    #prompt = "Q: What are the names of the days of the week? A:" 
