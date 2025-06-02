

class HuggingFaceModel:
    def __init__(self):
        model_name = "meta-llama/Llama-3.2-1B" 
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt, max_new_tokens=128, stop="Observation:"):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        generated_ids = input_ids.clone()
        output_text = ""

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            new_token_text = self.tokenizer.decode(next_token[0])
            output_text += new_token_text

            if stop and stop in output_text:
                output_text = output_text.split(stop)[0]
                break

            print(new_token_text, end="", flush=True)

        return output_text.strip()