# Apuntes de Ciencia de Datos

Cuaderno de apuntes del curso de Ciencia de Datos (UNAB, sexto semestre). Reúne teoría, ejemplos numéricos y notebooks prácticos en Python con TensorFlow y Keras, desde el perceptrón hasta modelos generativos.

## Contenido

| Corte | Carpeta | Temas |
|-------|---------|-------|
| Primer corte | [Apuntes-Ciencia-de-Datos](Apuntes-Ciencia-de-Datos) | Perceptrón, redes densas, introducción a CNN, taller con datos de pacientes |
| Segundo corte | [Apuntes ciencia de datos segundo corte](Apuntes%20ciencia%20de%20datos%20segundo%20corte) | CNN, transferencia de aprendizaje, YOLO, autoencoders, señales ECG, GAN y Stable Diffusion |

---

## Primer corte

Introducción práctica y teórica al aprendizaje automático y a las redes neuronales, con enfoque en problemas reales de clasificación, regresión y análisis de datos.

### Temas

- Fundamentos del aprendizaje automático y redes neuronales.
- Perceptrón: pesos, sesgos, activación, errores y ajuste de parámetros.
- Entrenamiento mediante descenso del gradiente y actualización de pesos.
- Funciones de activación: ReLU, sigmoid y softmax.
- Redes neuronales densas para clasificación y regresión.
- Modelos secuenciales y funcionales en TensorFlow y Keras.
- Preparación de datos, partición train/validation/test y métricas de evaluación.
- Introducción a redes convolucionales y transferencia de aprendizaje.

### Estructura

- **Perceptrón:** lógica de redes neuronales simples y su entrenamiento.
- **Redes densas:** redes multicapa, clasificación no lineal y arquitectura.
- **CNN y visión computacional:** procesamiento de imágenes y aprendizaje transferido.
- **Taller:** proyecto final con un modelo sobre datos reales de pacientes, preprocesamiento y evaluación.

---

## Segundo corte

El segundo corte pasa de las redes densas a la visión por computadora y los modelos generativos. Empieza por cómo funciona una convolución, sigue con clasificación, detección y segmentación de imágenes con modelos preentrenados, y termina con autoencoders, GAN y modelos de difusión.

### 1. Redes neuronales convolucionales (CNN)

| Notebook | Resumen |
|----------|---------|
| Cuaderno 9 – Convolución | Convolución "a mano": filtros Sobel 3x3 para detectar bordes, magnitud del gradiente y max pooling 2x2 sobre una imagen. |
| Cuaderno 7 – CNN | Qué es una CNN y sus componentes (convolución, pooling, capas densas). Utilidades de Keras para visión: carga de imágenes, aumentación de datos, normalización y modelos preentrenados. |
| Cuaderno 8 A2 – CNN con MNIST | Construcción, entrenamiento, evaluación y guardado (arquitectura y pesos) de una CNN que reconoce dígitos escritos a mano. |
| Cuaderno 9 – Taller de clasificación de imágenes | Clasificador de flores con TensorFlow y Keras: organizar un dataset propio, aumentación de datos y control del sobreajuste. |

### 2. Transferencia de aprendizaje

| Notebook | Resumen |
|----------|---------|
| Cuaderno 8 – Transferencia de aprendizaje | Diferencia entre transferencia de aprendizaje, *fine-tuning* y modelos en cascada. Tabla de modelos preentrenados populares y cómo reutilizarlos en Keras. |
| Cuaderno 9D – Detección de puntos clave | Detección de puntos clave (*keypoints*) en imágenes de perros con el dataset StanfordExtra, usando un modelo preentrenado como base. |

### 3. Detección y segmentación de objetos con YOLO

| Notebook | Resumen |
|----------|---------|
| Cuaderno 9 – Detección de objetos con YOLOv8 / Detección de objetos con YOLOv11 | Conceptos de detección (clasificación + localización), modelos y datasets comunes, formatos de etiquetado de *bounding boxes* e introducción a YOLO. Taller de transferencia de aprendizaje con YOLO. |
| Cuaderno 9A – Matrículas de vehículos | Pipeline completo sobre video: detección de vehículos con YOLO, detección de matrículas y lectura del texto con OCR. |
| Cuaderno 9B – Detección de EPP | Entrenamiento de un detector personalizado con YOLO (Ultralytics) para elementos de protección personal: dataset, `data.yaml` y métricas en TensorBoard. |
| Cuaderno 9C – Segmentación de instancias | Segmentación de instancias con YOLO11 y un dataset anotado con polígonos en Roboflow. |

### 4. Autoencoders y señales ECG

| Notebook | Resumen |
|----------|---------|
| Cuaderno 14 – Introducción a los autoencoders | Codificador, decodificador y espacio latente. Ejemplos: autoencoder básico y eliminación de ruido en imágenes. |
| Cuaderno 15A – Procesamiento de ECG | Lectura de un ECG real (base LTAFDB de PhysioNet) con `wfdb`, cálculo de la frecuencia cardíaca y detección de arritmias. |
| Cuaderno 15B – Autoencoder para ECG | Uso de un autoencoder para distinguir señales ECG normales de anómalas. |
| Cuaderno 15C – Autoencoder sobre LTAFDB | Segmentación de latidos del registro 23 de LTAFDB y detección de anomalías cardíacas a partir del error de reconstrucción. |
| Cuaderno 15D – Limpieza de imágenes | Autoencoder entrenado para limpiar imágenes con ruido. |

### 5. Modelos generativos

| Notebook | Resumen |
|----------|---------|
| Cuaderno 19 – GAN en Keras | Modelos generativos frente a discriminativos. GAN que genera dígitos MNIST: generador, discriminador, funciones de pérdida y ciclo de entrenamiento. |
| Taller GAN Panini FIFA 2026 | DCGAN entrenada con imágenes de futbolistas, con *label smoothing* y aumentación de datos, para crear una tarjeta coleccionable a partir de una foto propia. |
| Taller Stable Diffusion | Generación de imágenes a partir de texto con Hugging Face Diffusers: cómo funcionan los modelos de difusión, el pipeline y sus parámetros. |

### Herramientas usadas

Python, TensorFlow, Keras, Ultralytics (YOLOv8 / YOLO11), Roboflow, OpenCV, OCR, `wfdb` (PhysioNet), Hugging Face Diffusers y Google Colab con GPU.

---

## Objetivo del curso

Que el estudiante entienda cómo funcionan las redes neuronales, desde el perceptrón hasta los modelos generativos, y sea capaz de aplicarlas a problemas reales de ciencia de datos: imágenes, video y señales biomédicas.
