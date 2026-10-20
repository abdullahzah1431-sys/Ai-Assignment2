from flask import Flask, render_template, request
from predictor import make_prediction

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            result, probability = make_prediction(request.form)
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Prediction could not be completed. Please check the entered values and try again."

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error,
        form_data=request.form,
    )


if __name__ == "__main__":
    app.run(debug=True)
