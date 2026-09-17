from pathlib import Path
import json
import joblib
import pandas as pd
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "insurance_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
FEATURE_ORDER = metadata["feature_order"]


def predict_insurance(age, bmi, children, gender, smoker, region):
    row = {
        "age": int(age),
        "bmi": float(bmi),
        "children": int(children),
        "sex_male": 1 if gender == "Male" else 0,
        "smoker_yes": 1 if smoker == "Yes" else 0,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0,
    }

    input_df = pd.DataFrame([row], columns=FEATURE_ORDER)
    prediction = model.predict(input_df)[0]
    return f"Estimated insurance cost: {prediction:,.2f}"


with gr.Blocks(title="Medical Insurance Cost Predictor") as demo:
    gr.Markdown("# Medical Insurance Cost Predictor")
    gr.Markdown(
        "Enter a customer profile below to get an estimated insurance charge "
        "from the trained Multiple Linear Regression model."
    )

    with gr.Row():
        with gr.Column():
            age = gr.Slider(18, 64, value=30, step=1, label="Age")
            bmi = gr.Slider(15, 55, value=25, step=0.1, label="BMI")
            children = gr.Slider(0, 5, value=0, step=1, label="Number of Children")
            gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
            smoker = gr.Radio(["No", "Yes"], value="No", label="Smoker")
            region = gr.Dropdown(
                ["Northeast", "Northwest", "Southeast", "Southwest"],
                value="Northeast",
                label="Region",
            )
            predict_button = gr.Button("Predict", variant="primary")

        with gr.Column():
            result = gr.Textbox(label="Prediction Result", interactive=False)
            gr.Markdown(
                "This is a learning project built on the Medical Cost Personal Dataset. "
                "The prediction is an estimate, not a real insurance quote."
            )

    predict_button.click(
        fn=predict_insurance,
        inputs=[age, bmi, children, gender, smoker, region],
        outputs=result,
    )

    gr.Examples(
        examples=[
            [25, 22.5, 0, "Female", "No", "Northeast"],
            [50, 35.0, 2, "Male", "Yes", "Southeast"],
        ],
        inputs=[age, bmi, children, gender, smoker, region],
    )

if __name__ == "__main__":
    demo.launch()
