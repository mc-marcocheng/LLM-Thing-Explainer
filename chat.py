import sys
from threading import Thread

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList, TextIteratorStreamer)

from llm_thing_explainer.reader.xkcd_1000 import read_xkcd_1000
from llm_thing_explainer.src.logits_process import StateMachineLogitsProcessor
from llm_thing_explainer.src.state_machine import TokenStateMachine
from llm_thing_explainer.src.token_list import create_token_lists

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Step 1: Read the xkcd words file and create the list of token lists
list_of_token_lists = create_token_lists(tokenizer, read_xkcd_1000())

# Step 3: Initialize the logits processor with the state machine
state_machine_logits_processor = StateMachineLogitsProcessor(*list_of_token_lists)

# Step 4: Load pre-trained GPT-2 model and tokenizer

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Integrate the logits processor into the model's generation process
logits_processor_list = LogitsProcessorList([state_machine_logits_processor])

def chat():
    print("Chatbot: Hello! How can I help you today? (type 'exit' to quit)")

    messages = [{"role": "system", "content": "You are a helpful assistant that can only talk in very simple words."}]

    while True:
        # User input
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        messages.append({"role": "user", "content": user_input})

        # Tokenize user input
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        # Generate response
        generate_kwargs = dict(
            inputs=input_ids,
            streamer=streamer,
            logits_processor=logits_processor_list,
            max_new_tokens=256,
            # do_sample=False,
            # num_beams=1,
            # temperature=0.0,
            # top_k=30,
            # top_p=30,
            repetition_penalty=1.5,
            # length_penalty=1.0,
            # no_repeat_ngram_size=5,
        )

        # chat_history_ids = model.generate(
        #     input_ids,
        #     streamer=streamer,

        #     pad_token_id=tokenizer.eos_token_id,
        #     max_new_tokens=256,
        #     num_beams=1,
        #     repetition_penalty=1.5,
        # )

        def generate_and_signal_complete():
            model.generate(**generate_kwargs)

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        # Initialize an empty string to store the generated text
        bot_response = ""
        for new_text in streamer:
            bot_response += new_text
            print(new_text, end="")
            sys.stdout.flush()

        # Decode and add the bot's response to messages
        # bot_response = tokenizer.decode(chat_history_ids[:, input_ids.size(1):][0], skip_special_tokens=True)
        print(f"Chatbot: {bot_response}")
        messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    chat()
