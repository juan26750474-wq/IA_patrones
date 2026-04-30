import pandas as pd
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import warnings

# Silenciar avisos de nombres de columnas
warnings.filterwarnings("ignore", category=UserWarning)

# 1. CONEXIÓN SEGURA
user = os.getenv('DB_USER')
password = os.getenv('DB_PASS')
host = "horti.space"
db = "u135472660_precio"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")

# 2. EXTRACCIÓN DE DATOS
query = """
SELECT Producto, Tipo, Precio, F_Inicio,
       (SELECT AVG(t2.Precio) FROM datos3 t2 WHERE t2.Producto = t1.Producto AND t2.Tipo = t1.Tipo AND WEEK(t2.F_Inicio, 1) = WEEK(t1.F_Inicio, 1)) as media_sem,
       (SELECT t3.Precio FROM datos3 t3 WHERE t3.Producto = t1.Producto AND t3.Tipo = t1.Tipo AND t3.F_Inicio < t1.F_Inicio ORDER BY t3.F_Inicio DESC LIMIT 1) as precio_ant
FROM datos3 t1
WHERE F_Inicio = (SELECT MAX(F_Inicio) FROM datos3)
AND Tipo NOT IN ('CÓNICO', 'CONICO', 'DULCE CÓNICO', 'LAMUYO ROJO', 'CALIFORNIA VERDE')
"""

df = pd.read_sql(query, engine)

# Si la base de datos devuelve algo vacío para evitar errores
if df.empty:
    print("No hay datos para analizar hoy.")
else:
    # 3. PREPARACIÓN DE LAS "FEATURES"
    df['diff_hist'] = abs(df['Precio'] - df['media_sem'])
    df['var_sem'] = abs(df['Precio'] - df['precio_ant']) / df['precio_ant'].replace(0, np.nan)
    df.fillna(0, inplace=True)

    # Dimensiones para la IA
    X = df[['Precio', 'diff_hist', 'var_sem']]

    # 4. EJECUCIÓN DE ISOLATION FOREST
    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    
    # --- ESTA ES LA LÍNEA QUE FALTABA: ENTRENAR EL MODELO ---
    model.fit(X) 
    
    df['score_ia'] = model.decision_function(X)
    df['es_anomalia'] = model.predict(X)

    # 5. GUARDAR RESULTADOS EN TABLA NUEVA
    # Solo guardamos lo que la IA considera raro (-1)
    anomalias = df[df['es_anomalia'] == -1]
    anomalias.to_sql('alertas_ia', engine, if_exists='replace', index=False)

    print(f"Análisis Palantir completado con éxito. {len(anomalias)} anomalías detectadas.")
