import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList)

from reader.xkcd_1000 import read_xkcd_1000
from src.logits_process import StateMachineLogitsProcessor
from src.state_machine import TokenStateMachine
from src.token_list import create_token_lists

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 1: Read the xkcd words file and create the list of token lists
list_of_token_lists = create_token_lists(tokenizer, read_xkcd_1000())
token_state_machine = TokenStateMachine(list_of_token_lists)

# Step 3: Initialize the logits processor with the state machine
state_machine_logits_processor = StateMachineLogitsProcessor(list_of_token_lists)

# Step 4: Load pre-trained GPT-2 model and tokenizer

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Integrate the logits processor into the model's generation process
logits_processor_list = LogitsProcessorList([state_machine_logits_processor])

def chat():
    print("Chatbot: Hello! How can I help you today? (type 'exit' to quit)")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        # User input
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        messages.append({"role": "user", "content": user_input})

        # Tokenize user input
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        # Generate response
        chat_history_ids = model.generate(
            input_ids,
            logits_processor=logits_processor_list,
            max_new_tokens=512,
            num_beams=5,
        )

        # Decode and add the bot's response to messages
        bot_response = tokenizer.decode(chat_history_ids[:, input_ids.size(1):][0], skip_special_tokens=True)
        print(f"Chatbot: {bot_response}")
        messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    chat()
