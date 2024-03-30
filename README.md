# Reconocimiento de caracteres 0-9 con el algoritmo KNN
Hecho por: Luca Siegel


Título: Reconocimiento de dígitos dibujados haciendo uso del algoritmo KNN


Funcionamiento: Usando el algoritmo [k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) se puede intentar adivinar utilizando la distancia euclidiana para determinar la variación de color de los pixeles a que numero se parece el numero dibujado. El algoritmo funciona "comparando" el valor que tiene uno de los píxeles de nuestro dibujo(su color), con el resto de los pixeles en ese punto de la base de datos MNIST. Esta es una base de datos de acceso público con max de 60000 numeros dibujados a mano, recopilados y convertidos en una imagen de 28x28 pixeles. La precisión del algoritmo varía mucho del número y de la forma de dibujar este. Con los números 0,1,2 y 4 tiene mucha facilidad, ya que hay pocos números que se parezcan a estos. Sin embargo, la diferencia entre un 3 y un 9 son unos pocos píxeles, así que tiene muchas dificultades y hay que dibujar los números teniendo en cuenta esto.



# INSTALACION

Está requerido tener instalado la última versión del módulo de pygame, además de incluir la path de los archivos de MNIST en las variables 
"TEST DIR" y "DATA_DIR". La variable "number_comparisons" determina la precisión y velocidad del programa, siendo inversamente proporcionales 
(60000) es el valor máximo, y con tal cantidad, el programa tardará entre 10 y 15 segundos en procesarse.


# EJECUCIÓn
