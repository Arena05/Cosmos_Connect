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

## Flujo
1) Novato por defecto (switch arriba).  
2) Predice con 11 features.  
3) En resultados: **Guardar muestra** (se añade a tu dataset); **Reentrenar con mis datos**; **Eliminar mis datos** (en Novato/Experto).

Artefactos: `instance/models/{user_id}/{key}/`. Cache NASA: `instance/cache/nasa_merged.parquet`.
