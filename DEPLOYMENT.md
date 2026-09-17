# Deploying the Gradio app

The repository is already set up so the same app can run locally or on Hugging Face Spaces.

## Hugging Face Spaces

1. Create a new Space at Hugging Face.
2. Choose **Gradio** as the SDK.
3. Upload the contents of this repository (or copy the files from GitHub).
4. Make sure `app.py`, `requirements.txt`, and the `model/` folder are present.
5. Wait for the Space to build.
6. Open the public app URL and test at least two customer profiles.

The app loads `model/insurance_model.pkl`, so it does not need to retrain on startup.

For the internship submission, take screenshots of:

- the GitHub repository home page
- the rendered README
- the deployed app
- one working prediction result
- the final repository file structure
