# SPACE IET · Clasificador de Exoplanetas (Flask)

- Auth (registro/login)
- Modo **Novato**: RF fijo (n_estimators=1000, random_state=10)
- Modo **Experto**: seleccionar modelo + hiperparámetros (listas restringidas)
- Dataset por usuario: guardar muestra → reentrenar NASA+usuario → borrar

## Local
```bash
python -m venv venv
venv\Scripts\activate  # Windows | source venv/bin/activate (macOS/Linux)
pip install -r requirements.txt
python run.py
# http://localhost:5000 → regístrate e ingresa
```

## Flow
1) Novice by default (switch up).  
2) Predicts with 11 features.  
3) In results: **Save sample** (added to your dataset); **Retrain with my data**; **Delete my data** (in Novice/Expert).

Artefactos: `instance/models/{user_id}/{key}/`. Cache NASA: `instance/cache/nasa_merged.parquet`.

## A web platform for connecting and exploring ideas, hosted on Render.
https://cosmos-connect-hcok.onrender.com
