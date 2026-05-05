import pandas as pd
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. CONEXIÓN
user = os.getenv('DB_USER')
password = os.getenv('DB_PASS')
host = "srv1016.hstgr.io"
db = "u135472660_precio"
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")

# 2. QUERY PRINCIPAL — semana actual
query_actual = """
SELECT Producto, Tipo, Precio, F_Inicio,
       (SELECT AVG(t2.Precio) FROM datos3 t2 
        WHERE t2.Producto = t1.Producto AND t2.Tipo = t1.Tipo 
        AND WEEK(t2.F_Inicio, 1) = WEEK(t1.F_Inicio, 1)) as media_sem,
       (SELECT t3.Precio FROM datos3 t3 
        WHERE t3.Producto = t1.Producto AND t3.Tipo = t1.Tipo 
        AND t3.F_Inicio < t1.F_Inicio 
        ORDER BY t3.F_Inicio DESC LIMIT 1) as precio_ant,
       (SELECT t3.F_Inicio FROM datos3 t3 
        WHERE t3.Producto = t1.Producto AND t3.Tipo = t1.Tipo 
        AND t3.F_Inicio < t1.F_Inicio 
        ORDER BY t3.F_Inicio DESC LIMIT 1) as fecha_ant,
       (SELECT t4.Precio FROM datos3 t4 
        WHERE t4.Producto = t1.Producto AND t4.Tipo = t1.Tipo 
        AND t4.F_Inicio < t1.F_Inicio 
        ORDER BY t4.F_Inicio DESC LIMIT 1 OFFSET 1) as precio_2sem
FROM datos3 t1
WHERE F_Inicio = (SELECT MAX(F_Inicio) FROM datos3)
AND Tipo NOT IN ('CÓNICO', 'CONICO', 'DULCE CÓNICO', 'LAMUYO ROJO', 'CALIFORNIA VERDE')
"""

# 3. QUERY HISTÓRICA — para volatilidad y percentil
query_historico = """
SELECT Producto, Tipo,
       STDDEV(Precio) as std_hist,
       AVG(Precio)    as media_hist,
       MIN(Precio)    as min_hist,
       MAX(Precio)    as max_hist
FROM datos3
WHERE Tipo NOT IN ('CÓNICO', 'CONICO', 'DULCE CÓNICO', 'LAMUYO ROJO', 'CALIFORNIA VERDE')
GROUP BY Producto, Tipo
"""

df         = pd.read_sql(query_actual,    engine)
df_histori = pd.read_sql(query_historico, engine)

if not df.empty:

    # DEDUPLICACIÓN
    df = df.sort_values('F_Inicio', ascending=False)
    df = df.drop_duplicates(subset=['Producto', 'Tipo'], keep='first')

    # Unir histórico
    df = df.merge(df_histori, on=['Producto', 'Tipo'], how='left')

    # --- FILTRO ANTIFALSOS POSITIVOS ---
    df['F_Inicio']  = pd.to_datetime(df['F_Inicio'])
    df['fecha_ant'] = pd.to_datetime(df['fecha_ant'])
    df['dias_desde_ultimo'] = (df['F_Inicio'] - df['fecha_ant']).dt.days
    df.loc[df['dias_desde_ultimo'] > 15, 'precio_ant']  = df['Precio']
    df.loc[df['dias_desde_ultimo'] > 15, 'precio_2sem'] = df['Precio']

    # 4. VARIABLES — CON SIGNO para preservar dirección del movimiento

    # Desviación vs media HISTÓRICA (no semanal) — con signo: + = caro, - = barato
    df['diff_hist'] = (df['Precio'] - df['media_hist']) / df['media_hist'].replace(0, np.nan)

    # Variación semanal — con signo: + = subida, - = bajada
    df['var_sem'] = (df['Precio'] - df['precio_ant']) / df['precio_ant'].replace(0, np.nan)

    # Variación 2 semanas — con signo: + = subida, - = bajada
    df['var_2sem'] = (df['Precio'] - df['precio_2sem']) / df['precio_2sem'].replace(0, np.nan)

    # Volatilidad histórica: std / media (siempre positivo, mide dispersión)
    df['volatilidad_hist'] = df['std_hist'] / df['media_hist'].replace(0, np.nan)

    # Percentil histórico: 0 = mínimo histórico, 1 = máximo histórico
    df['percentil_precio'] = (df['Precio'] - df['min_hist']) / (df['max_hist'] - df['min_hist']).replace(0, np.nan)

    df.fillna(0, inplace=True)

    # 5. MODELO IA — usa valores absolutos para que la magnitud sea la señal,
    #    no la dirección (el IF detecta rareza, no tendencia)
    X = df[['Precio',
            'diff_hist',       # el IF ve la magnitud: usamos abs aquí solo para el modelo
            'var_sem',
            'var_2sem',
            'volatilidad_hist',
            'percentil_precio']].copy()
    X['diff_hist'] = X['diff_hist'].abs()
    X['var_sem']   = X['var_sem'].abs()
    X['var_2sem']  = X['var_2sem'].abs()

    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    model.fit(X)

    df['score_ia']    = model.decision_function(X)
    df['es_anomalia'] = model.predict(X)

    # 6. GUARDAR — guardamos las columnas CON SIGNO para que PHP interprete dirección
    columnas_guardar = [
        'Producto', 'Tipo', 'Precio', 'F_Inicio',
        'score_ia', 'es_anomalia',
        'var_sem', 'var_2sem', 'diff_hist',   # con signo
        'volatilidad_hist', 'percentil_precio'
    ]
    anomalias = df[df['es_anomalia'] == -1][columnas_guardar]
    anomalias.to_sql('alertas_ia', engine, if_exists='replace', index=False)

    print(f"Análisis completado. {len(anomalias)} anomalías detectadas.")
