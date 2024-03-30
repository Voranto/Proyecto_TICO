# Proyecto_TICO
Hecho por: Luca Siegel
Título: Reconocimiento de dígitos dibujados haciendo uso del algoritmo KNN

Funcionamiento: Usando el algoritmo knn(k-nearest neighbors) se puede intentar adivinar utilizando la distancia euclidiana para determinar la variación de color de los pixeles a que numero se parece el numero dibujado. El algoritmo funciona "comparando" el valor que tiene uno de los píxeles de nuestro dibujo(su color), con el resto de los pixeles en ese punto de la base de datos MNIST. Esta es una base de datos de acceso público con max de 60000 numeros dibujados a mano, recopilados y convertidos en una imagen de 28x28 pixeles  Tiene muchas dificultades con ciertos números, como el 9 o el 6, ya que se parecen mucho a otros números, como por ejemplo, el 9 se parece mucho a un 4 al dibujarse. 

#Ejecución del programa. Está requerido tener instalado la última versión del módulo de pygame, además de incluir la path de los archivos de MNIST en las variables 
"TEST DIR" y "DATA_DIR". La variable "number_comparisons" determina la precisión y velocidad del programa, siendo inversamente proporcionales 
(60000) es el valor máximo, y con tal cantidad, el programa tardará entre 10 y 15 segundos en procesarse.


# INSTALACION
