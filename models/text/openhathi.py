import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


class HindiTextGenerator:
    def __init__(self, model_name="sarvamai/OpenHathi-7B-Hi-v0.1-Base"):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

    def generate_text(self, prompt, max_length=30):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate text
        generate_ids = self.model.generate(
            inputs.input_ids, max_length=max_length
        )

        # Decode generated IDs
        generated_text = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return generated_text


if __name__ == "__main__":
    # Example usage
    hindi_text_generator = HindiTextGenerator()
    prompt = "मैं एक अच्छा हाथी हूँ"
    generated_text = hindi_text_generator.generate_text(prompt)
    print(f"Generated Text: {generated_text}")
