# Deploying the Streamlit app

The deployment version of the project uses Streamlit and loads the trained model directly from the `model/` folder.

## Streamlit Community Cloud

1. Push this version of the project to GitHub.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Create a new app and select this repository.
4. Use the `main` branch and set the app file to `app.py`.
5. Deploy the app and wait for the build to finish.
6. Open the public app link and test a few different customer profiles.

No API keys or secrets are needed for this project.

The app loads `model/insurance_model.pkl`, so it does not retrain every time it starts.

## Quick checks after deployment

Try these two profiles and confirm that the deployed app gives approximately the same results as the notebook tests:

- Age 25, BMI 22.5, 0 children, Female, Non-smoker, Northeast → about **2,283.40**
- Age 50, BMI 35, 2 children, Male, Smoker, Southeast → about **35,675.76**

For the Phase 8 submission, keep screenshots of:

- the GitHub repository
- the rendered README
- the deployed application
- a working prediction
- the final repository structure

Once the app is live, add its public link to the README.
