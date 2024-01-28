from flask import Flask, render_template
from AirWriter import AirWriter
import cv2
import numpy as np


app = Flask(__name__)
app.register_blueprint(AirWriter, url_prefix="")

@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
