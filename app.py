import os
import joblib

from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "model/fake_news_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    news_text = None

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()

        if not news_text:
            prediction = "Please enter some news text."

        elif len(news_text) > 20000:
            prediction = "News text is too long. Maximum 20,000 characters."

        else:
            result = int(model.predict([news_text])[0])

            if result == 0:
                prediction = "Fake News"
            else:
                prediction = "Real News"

            probabilities = model.predict_proba([news_text])[0]
            confidence = round(float(max(probabilities)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text
    )


if __name__ == "__main__":
    app.run(debug=True)
