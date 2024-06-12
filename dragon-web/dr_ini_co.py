
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# logging.get_logger("transformers").setLevel(logging.ERROR)

tokenizer = AutoTokenizer.from_pretrained("/dragonAI",
                                          padding_side='left')
model = AutoModelForCausalLM.from_pretrained("/dragonAI")


def predict(counts,req):
    chat_history_ids = None  # Initialize chat history

    for step in range(counts):
        # Get user input
        user_input = f'{req}'

        # Encode the new user input, add the eos_token and return a tensor in PyTorch
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                                  dim=-1) if step > 0 else new_user_input_ids

        # Generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Pretty print the last output tokens from the bot
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_response
# Example usage:
# predict(5)
