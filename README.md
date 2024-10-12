# TP Final Vision por computadora II - Transferencia de estilo a pinturas al óleo

Este repositorio contiene el trabajo práctico final de la materia Visión por Vision por computadora II de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). El objetivo es implementar un modelo de aprendizaje profundo que sea capaz de transformar fotografías al óleo, manteniendo el contenido original y logrando una estilización coherente.

## Transferencia de estilo: contenido a pinturas

![painting_1](saved_images/painting_1.png)

![painting_2](saved_images/painting_2.png)

![painting_3](saved_images/painting_3.png)

![painting_4](saved_images/painting_4.png)

En promedio el SSIM obtenido fue de 0.6179 para la transferencia de estilo de contenido a pinturas para la iteración final.

## Transferencia de estilo: contenidos a pinturas

![content_1](saved_images/content_1.png)

![content_2](saved_images/content_2.png)

![content_3](saved_images/content_3.png)

![content_4](saved_images/content_4.png)

En promedio el SSIM obtenido fue de 0.6270 para la transferencia de estilo de pinturas a contenido para la iteración final.

## Consistencia de ciclo

![cycle_consistency_1](saved_images/cycle_consistency_1.png)

![cycle_consistency_2](saved_images/cycle_consistency_2.png)

![cycle_consistency_3](saved_images/cycle_consistency_3.png)

![cycle_consistency_4](saved_images/cycle_consistency_4.png)

Como se puede ver, aunque existe una pérdida de información en la transformación, el modelo es capaz de mantener la coherencia en la transformación de las imágenes.

# Instrucciones de uso

## Requisitos
Para poder ejecutar el código es necesario tener instalado Python 3.7 o superior. Se recomienda utilizar un entorno virtual para instalar las dependencias del proyecto. Además, se recomienda tener una GPU con CUDA configurado para el entrenamiento del modelo.

## Instalación
Para instalar las dependencias del proyecto, ejecutar el siguiente comando:

```bash
pip install -r requirements.txt
```

## Organización
El proyecto está organizado de la siguiente manera:
- [Preparación de datos](data_preparation_notebook.ipynb): Jupyter notebook para la preparación de los datos.
- [Entrenamiento](training_notebook.ipynb): Jupyter notebook para el entrenamiento del modelo.