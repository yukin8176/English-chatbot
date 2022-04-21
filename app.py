import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, render_template, session
from datetime import timedelta
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)
app.secret_key = "abcdefg"
app.permanent_session_lifetime = timedelta(minutes=3)


@app.route("/")
def home():
    return render_template("index.html")

# Chat


@app.route("/", methods=["POST"])
def get_bot_response():
    global chat_history_ids

    if "step" in session:
        session["step"] += 1
    else:
        session["step"] = 0
    step = session["step"]
    print(step)

    if "msg" in session:
        messages = session["msg"]
        # chat_history_ids = session["chi"]
    else:
        messages = []

    user_input = request.form["user_input"]
    print(user_input)
    messages.append([0,user_input])

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    messages.append([1, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)])

    print(chat_history_ids)
    session["msg"] = messages
    # session["chi"] = chat_history_ids
    return render_template("index.html", messages=messages)


if __name__ == "__main__":
    model_name = "microsoft/DialoGPT-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    os.environ["FLASK_ENV"] = "development"
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
