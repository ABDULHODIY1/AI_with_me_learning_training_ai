import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# logging.get_logger("transformers").setLevel(logging.ERROR)
path = "/results"
tokenizer = AutoTokenizer.from_pretrained(f"{path}",padding_side='left')
model = AutoModelForCausalLM.from_pretrained(f"{path}")
while True:
    # Let's chat for 5 lines
    for step in range(1):
        prompt = input(">>> ")
        if prompt == "quit":
            break
        else:
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            # pretty print last ouput tokens from bot
            print("Dragon: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
