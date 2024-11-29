from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)

# Configuración de archivos y generación para entrenamiento y predicción
train_files = ["train_files/Gen 17_SD2020.xlsx", "train_files/Gen 18_SD2021.xlsx", "train_files/Gen 19_SD2022.xlsx"]
train_generations = ["gen17", "gen18", "gen19"]
predict_file_path = "neural-network/GEN 19_SD2022_Prueba.xlsx"  # Archivo de predicción
predict_generation = "gen19"  # Generación para predicción

# Mapeos específicos por generación
column_mapping = {
    "gen17": {
        "PROGRAMA EDUCATIVO": "PROGRAMA EDUCATIVO",
        "PYMES": "ADMINISTRACION Y GESTION",
    },
    "gen18": {
        "PROGRAMA EDUCATIVO": "PROGRAMA EDUCATIVO",
        "PYMES": "ADMINISTRACION Y GESTION",
    },
    "gen19": {
        "Programa Educativo": "PROGRAMA EDUCATIVO",
        "PYMES": "ADMINISTRACION Y GESTION",
    },
    "gen21": {
        "Carrera": "PROGRAMA EDUCATIVO",
        "PYMES": "ADMINISTRACION Y GESTION",
    }
}

# Cargar y procesar los datos
def load_and_process_data(file_path, generation):
    data = pd.read_excel(file_path)

    # Renombrar columnas según el mapeo de la generación
    if generation in column_mapping:
        data.rename(columns={k: v for k, v in column_mapping[generation].items() if k in data.columns}, inplace=True)
    
    if "PROGRAMA EDUCATIVO" in data.columns:
        data['PROGRAMA EDUCATIVO'] = data['PROGRAMA EDUCATIVO'].replace(column_mapping[generation])

    label_encoder_PROGRAMAEDUCATIVO = LabelEncoder()
    if "PROGRAMA EDUCATIVO" in data.columns:
        data['PROGRAMA EDUCATIVO'] = label_encoder_PROGRAMAEDUCATIVO.fit_transform(data['PROGRAMA EDUCATIVO'])
    
    data.replace('#N/D', -1, inplace=True)

    categorical_mappings = {
        "Estado civil": {"Soltero(a)": 0, "Casado(a)": 1, "Divorciado(a)": 2, "Union Libre": 3},
        "Hijos": {"No": 0, "Si": 1},
        "Problema o padecimiento": {"No": 0, "Si": 1},
        "Trabaja": {"No": 0, "Si": 1},
        "UPQ Primera opción": {"No": 0, "Si": 1}
    }
    for col, mapping in categorical_mappings.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
    
    numeric_columns = ["Razonamiento matemático", "Razonamiento abstracto", "R. Verbal", "R. Espacial", "Informática",
                       "Cálculo", "Comunicación", "Mecánico", "Servicio", "Finanzas", "Ingreso mensual de padre",
                       "Ingreso mensual de madre", "Ingresos mensuales familiares", 'Minutos de trayecto a la universidad']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    range_defined_features = ['Razonamiento matemático', 'Razonamiento abstracto', 'R. Verbal', 
                              'R. Espacial', 'Informática', 'Cálculo', 'Comunicación', 'Mecánico', 
                              'Servicio', 'Finanzas']
    no_range_features = ['Ingreso mensual de padre', 'Ingreso mensual de madre', 
                         'Ingresos mensuales familiares', 'Minutos de trayecto a la universidad']
    
    data[range_defined_features] = data[range_defined_features] / 100

    min_max_scaler = MinMaxScaler()
    data[no_range_features] = min_max_scaler.fit_transform(data[no_range_features])

    if {'BAJA 1ER. CUATRI', 'BAJA 2DO. CUATRI', 'BAJA 3ER. CUATRI'}.issubset(data.columns):
        data['BAJA'] = data[['BAJA 1ER. CUATRI', 'BAJA 2DO. CUATRI', 'BAJA 3ER. CUATRI']].apply(
            lambda x: 1 if any(val in ['BT', 'BV', 'BAC'] for val in x) else 0, axis=1)
    else:
        data['BAJA'] = 0

    return data

# Cargar datos de entrenamiento
def load_and_combine_training_data(files, generations):
    combined_data = pd.DataFrame()
    for file_path, generation in zip(files, generations):
        data = load_and_process_data(file_path, generation)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

train_data = load_and_combine_training_data(train_files, train_generations)
features = ['PROGRAMA EDUCATIVO', 'Razonamiento matemático', 'Razonamiento abstracto', 'R. Verbal', 
            'R. Espacial', 'Informática', 'Cálculo', 'Comunicación', 'Mecánico', 'Servicio', 
            'Finanzas', 'Ingreso mensual de padre', 'Ingreso mensual de madre', 'Ingresos mensuales familiares', 
            'Minutos de trayecto a la universidad', 'Estado civil', 'Hijos', 'Problema o padecimiento', 
            'Trabaja', 'UPQ Primera opción']
X_train = train_data[features]
y_train = train_data['BAJA']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# Modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))
model.save('model/lstm_model.h5')

# Endpoint para predecir
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        predict_data = load_and_process_data(file_path, predict_generation)
        X_predict = predict_data[features]
        X_predict_scaled = scaler.transform(X_predict)
        X_predict_reshaped = np.reshape(X_predict_scaled, (X_predict_scaled.shape[0], X_predict_scaled.shape[1], 1))

        predictions = model.predict(X_predict_reshaped)
        predictions_percentage = predictions * 100

        predict_data['Probabilidad Baja (%)'] = predictions_percentage
        output_file = f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        predict_data.to_excel(output_file, index=False)
        return jsonify({"message": f"Predicciones guardadas en {output_file}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
