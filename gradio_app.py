import gc
from threading import Thread

import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList, TextIteratorStreamer)

from llm_thing_explainer.logits_process import StateMachineLogitsProcessor
from llm_thing_explainer.reader import WORD_READERS
from llm_thing_explainer.token_list import create_token_lists


class LLMLoader:
    def __init__(self):
        self.model_id = None
        self.tokenizer = None
        self.model = None

    def load_model(self, model_id: str, *args):
        if self.model_id == model_id:
            if len(args):
                return model_id, *args
            return model_id
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        if len(args):
            return model_id, *args
        return model_id

    def generate(
            self,
            messages: list[dict],
            reader_name: str = "xkcd 1000 words",
            add_numbers: bool = True,
            max_new_tokens: int = 256,
            num_beams: int = 1,
            temperature: float = 1.,
            top_p: float = 0.95,
            repetition_penalty: float = 1.5,
        ):
        # Tokenize user input
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # Prepare LLM Thing Explainer processor
        list_of_token_lists = create_token_lists(self.tokenizer, WORD_READERS[reader_name](), add_numbers=add_numbers)
        state_machine_logits_processor = StateMachineLogitsProcessor(*list_of_token_lists)
        logits_processor_list = LogitsProcessorList([state_machine_logits_processor])

        # Prepare generation kwargs
        generate_kwargs = dict(
            inputs=input_ids,
            logits_processor=logits_processor_list,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if num_beams == 1:
            # Support text streaming
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs["streamer"] = streamer

            def generate_and_signal_complete():
                self.model.generate(**generate_kwargs)
            t1 = Thread(target=generate_and_signal_complete)
            t1.start()

            # Initialize an empty string to store the generated text
            messages.append({"role": "assistant", "content": ""})
            for new_text in streamer:
                messages[-1]["content"] += new_text
                yield messages

        else:
            # No text streaming for beam search
            chat_history_ids = self.model.generate(
                **generate_kwargs,
            )
            bot_response = self.tokenizer.decode(chat_history_ids[0, input_ids.size(1):], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": bot_response})
            yield messages

def add_user_input(messages: list[dict], user_input: str) -> list[dict]:
    if not messages:
        messages = [{"role": "system", "content": "You are a helpful assistant that can only talk concisely in very simple words."}]
    messages.append({"role": "user", "content": user_input})
    return messages, ""


llm_loader = LLMLoader()

with gr.Blocks(title="LLM Thing Explainer") as demo:
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(label="Thing Explainer", type="messages", height="85vh", autoscroll=True)
        user_input = gr.Textbox(container=False, placeholder="Type your message here...", show_label=False)

    with gr.Tab("Settings"):
        model_dropdown = gr.Textbox(label="Model", value="meta-llama/Llama-3.1-8B-Instruct")
        load_model_button = gr.Button(value="Load Model")
        load_model_button.click(
            fn=llm_loader.load_model,
            inputs=[model_dropdown],
            outputs=[model_dropdown],
        )

        with gr.Row():
            reader_name = gr.Dropdown(label="Word List", choices=list(WORD_READERS.keys()), value="xkcd 1000 words")
            add_numbers_checkbox = gr.Checkbox(label="Add numbers", value=True, scale=0)
        max_new_tokens_slider = gr.Slider(
            label="Max New Tokens",
            minimum=32,
            maximum=2048,
            step=8,
            value=256,
        )
        num_beams_slider = gr.Slider(
            label="Beam Size",
            minimum=1,
            maximum=8,
            step=1,
            value=5,
        )
        temperature_slider = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=1.0,
        )
        top_p_slider = gr.Slider(
            label="Top P",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.95,
        )
        repetition_penalty_slider = gr.Slider(
            label="Repetition Penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.1,
            value=1.5,
        )

    user_input.submit(
        add_user_input,
        inputs=[chatbot, user_input],
        outputs=[chatbot, user_input],
        show_progress=False,
    ).then(
        llm_loader.load_model,
        inputs=[model_dropdown, chatbot],
        outputs=[model_dropdown, chatbot],
    ).then(
        llm_loader.generate,
        inputs=[chatbot, reader_name, add_numbers_checkbox, max_new_tokens_slider, num_beams_slider, temperature_slider, top_p_slider, repetition_penalty_slider],
        outputs=[chatbot],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
