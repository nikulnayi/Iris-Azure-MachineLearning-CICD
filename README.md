# 🔁 Azure ML CI/CD: Iris Classification Deployment
![CI](https://github.com/your-username/your-repo/actions/workflows/azureml-cicd.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Azure ML](https://img.shields.io/badge/Azure%20ML-Workspace-blue)](https://ml.azure.com/)

This project demonstrates a complete CI/CD workflow using **Azure Machine Learning** and **GitHub Actions** to:

- ✅ Train a Scikit-learn model on the Iris dataset
- ✅ Register the model to Azure ML
- ✅ Deploy it to a managed online endpoint
- ✅ Trigger the pipeline automatically on every `main` branch push

---

## 📁 Project Structure

```
.
├── assets/
│   ├── deployment.yml         # Deployment config (endpoint, model, env)
│   └── environment.yml        # Conda environment for inference
├── model/
│   └── score.py               # Scoring script for predictions
├── outputs/
│   └── model.joblib           # Trained model (auto-generated)
├── src/
│   └── train.py               # Training script
├── .github/workflows/
│   └── azureml-cicd.yml       # GitHub Actions pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 CI/CD Flow

### ✅ Trigger: `git push origin main`

1. **Train**: Runs `train.py` to train a logistic regression model
2. **Register**: Saves and registers the model to Azure ML
3. **Deploy**: Deploys to a live endpoint via `deployment.yml`

---

## 🔬 Model

We use a simple **logistic regression** model trained on the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

```python
LogisticRegression(max_iter=200)
```

---

## 🔮 Inference (score.py)

```python
def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    inputs = np.array(data["data"])
    preds = model.predict(inputs).tolist()
    return { "result": preds }
```

---

## 📦 Deploy Config: `deployment.yml`

```yaml
model: azureml:iris-logreg-model:1
environment: azureml:iris-env:1
instance_type: Standard_DS2_v2
instance_count: 1
code_configuration:
  code: ./model
  scoring_script: score.py
```

---

## 🔐 Secrets Required

Set this GitHub secret:

- `AZURE_CREDENTIALS`: output of `az ad sp create-for-rbac --sdk-auth ...`

---

## 🔎 Test Your Deployed Endpoint

```bash
curl -X POST https://<your-endpoint>.inference.ml.azure.com/score \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-key>" \
  -d @sample.json
```

---

## 📊 Example Output

```json
{
  "result": ["setosa", "versicolor"]
}
```

---

## 💡 Credits

Built with:
- Azure Machine Learning
- GitHub Actions
- Scikit-learn

---

## 📌 Next Steps

- [ ] Add monitoring and logging
- [ ] Implement batch inference
- [ ] Deploy a more complex model

---

## 🧠 Want Help?

DM me on [LinkedIn](https://www.linkedin.com/in/nikul-nayi/) or [open an issue](https://github.com)!
