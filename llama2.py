from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

# Define a function to generate summaries
def generate_summary(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_summary

# Example usage
restaurant_review = "This restaurant has amazing food, and the ambiance is superb. The service is top-notch, and I would highly recommend it."
summary = generate_summary(restaurant_review)
print(summary)
