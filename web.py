from flask import Flask, render_template,request

from model import chatbot_response

app = Flask(__name__,template_folder='template')

@app.route("/")
def home():
    return render_template('Home.html')

@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')

@app.route("/get")
def get_bot_response():
    userinput = request.args.get("text")
    return str(chatbot_response(userinput))

if __name__ == "__main__":
    app.run(debug=True)