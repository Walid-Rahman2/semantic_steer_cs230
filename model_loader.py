import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingfaceModel:
    def __init__(self, model_name, cache_dir, max_new_tokens=50):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map={"": 0},
            trust_remote_code=True
        )

    def predict(self, input_data, temperature=0.7, return_latent=False):
        # Tokenize input
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda:0")

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the generated sequence
        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Determine the number of generated tokens
        token_stop_index = len(outputs.sequences[0])
        n_input_token = len(inputs["input_ids"][0])
        n_generated = token_stop_index - n_input_token

        # Extract hidden states
        hidden = outputs.decoder_hidden_states if "decoder_hidden_states" in outputs.keys() else outputs.hidden_states
        if len(hidden) == 1:
            last_input = hidden[0]
        elif (n_generated - 1) >= len(hidden):
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Get the last token embedding
        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Optionally extract latent embeddings
        if return_latent:
            if len(hidden) == 1:
                sec_last_input = hidden[0]
            elif (n_generated - 2) >= len(hidden):
                sec_last_input = hidden[-2]
            else:
                sec_last_input = hidden[n_generated - 2]
            sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
            last_tok_bef_gen_input = hidden[0]
            last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()
        else:
            sec_last_token_embedding = None
            last_tok_bef_gen_embedding = None

        hidden_states = (last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding)

        return full_answer, hidden_states
