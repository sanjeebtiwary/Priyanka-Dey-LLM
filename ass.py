import torch
from transformers import AutoModelForCausalLM

# Load the Llama 2 model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Generate text
generated_text = model.generate(
    input_ids=torch.tensor([1]).unsqueeze(0),
    max_length=100,
    num_beams=5,
    repetition_penalty=2.0,
    length_penalty=1.0,
    early_stopping=True,
)

# Print the generated text
print(generated_text)
