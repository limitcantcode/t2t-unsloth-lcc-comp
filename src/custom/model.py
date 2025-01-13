from unsloth import FastLanguageModel
import logging
from transformers import TextIteratorStreamer

class UnslothAIModel():
    def __init__(self, model_name, load_in_4_bit):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = load_in_4_bit
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, sys_prompt, user_prompt):
        messages=[
            { "role": "system", "content": sys_prompt},
            { "role": "user", "content": user_prompt }
        ]

        logging.debug(f"Sending messages: {messages}")
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        stream = TextIteratorStreamer(self.tokenizer, skip_prompt = True)
        _ = self.model.generate(input_ids=inputs, streamer=stream, max_new_tokens=128,
                        use_cache=True, temperature=1.5, min_p=0.1)

        logging.debug(f"Streaming results")
        full_response = ""
        for chunk in stream:
            end_index = -1
            content_chunk = chunk or ""
            if self.tokenizer.eos_token in content_chunk:
                end_index = chunk.index(self.tokenizer.eos_token)
                content_chunk = content_chunk[:end_index]
            full_response += content_chunk
            yield content_chunk
            if end_index >= 0: break
        logging.debug(f"Finished with full response: {full_response}")
