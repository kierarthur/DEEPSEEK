from flask import Flask

app = Flask(__name__)

@app.route("/")
def root():
    return "Checklist headshot service is running"
