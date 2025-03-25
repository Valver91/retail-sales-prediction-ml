# 🛍️ Predicción de Ventas para Tiendas Minoristas con Machine Learning

Este proyecto presenta una solución basada en Machine Learning para predecir las ventas diarias de una cadena de tiendas minoristas. Se utiliza un modelo de regresión construido con redes neuronales profundas (MLP) entrenado con datos históricos.

---

## 📊 Objetivo

Predecir las ventas diarias de tiendas físicas a partir de variables como:
- Número de clientes
- Promociones activas
- Temporada del año
- Estado de apertura
- Otras características operativas

---

## 🧠 Modelo Utilizado

- 📌 Red neuronal profunda (5 capas ocultas, 350 neuronas c/u)
- 🔧 Activación: ReLU
- 🧪 Salida: Neurona única con activación lineal (regresión)
- ⚙️ Optimización: Adam
- 🎯 Pérdida: MSE (Error Cuadrático Medio)

---

## 📁 Dataset

> **Fuente:** Dataset simulado para entrenamiento con +1 millón de registros.  

---

## 📈 Resultados

| Métrica                 | Valor     |
|------------------------|-----------|
| MAE Base (promedio)    | 2886.89   |
| MAE Final (modelo ML)  | **609.63** |
| MSE Final              | 825,782.31 |
| Promedio real de ventas| 5773.13   |

✅ Mejora de más del **78%** en precisión respecto a una predicción trivial usando el promedio.

---

## 🛠️ Tecnologías

- Python
- Pandas, NumPy
- Keras (TensorFlow backend)
- Scikit-learn
- Matplotlib

---
