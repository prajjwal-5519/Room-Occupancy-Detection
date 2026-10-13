[README.md](https://github.com/user-attachments/files/33391332/README.md)
# Room Occupancy Detection - Naive Bayes (Streamlit app)

Web app for the sensor-based occupancy classifier: Gaussian NB and Kernel NB (KDE), 80/20 split,
accuracy, confusion matrix, Type I/II error, precision, recall, F1 and heat maps, plus a live
prediction form.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app (entry point) |
| `nb_models.py` | Kernel Naive Bayes class |
| `requirements.txt` | Python dependencies |
| `occupancy_dataset.csv` | *(recommended)* the CSV your Colab notebook creates |

If `occupancy_dataset.csv` is missing, the app downloads the data from UCI at start-up.
Adding the CSV to the repo makes start-up faster and removes that dependency.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a **public GitHub repository** and upload `app.py`, `nb_models.py`, `requirements.txt`
   (and `occupancy_dataset.csv` from Colab: Files panel -> right-click -> Download).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app** -> pick your repository, branch `main`, main file path `app.py`.
4. Click **Deploy**. After a few minutes you get a public URL like
   `https://<your-app-name>.streamlit.app` to put in your report.

Alternative: create a Hugging Face Space with the Streamlit SDK and upload the same files.
