# Reconocimiento de caracteres 0-9 con el algoritmo KNN
Hecho por: Luca Siegel


Título: Reconocimiento de dígitos dibujados haciendo uso del algoritmo KNN


Funcionamiento: Usando el algoritmo [k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) se puede intentar adivinar utilizando la distancia euclidiana para determinar la variación de color de los pixeles a que numero se parece el numero dibujado. El algoritmo funciona "comparando" el valor que tiene uno de los píxeles de nuestro dibujo(su color), con el resto de los pixeles en ese punto de la base de datos [MNIST](https://en.wikipedia.org/wiki/MNIST_database). Esta es una base de datos de acceso público con max de 60000 numeros dibujados a mano, recopilados y convertidos en una imagen de 28x28 pixeles. La precisión del algoritmo varía mucho del número y de la forma de dibujar este. Con los números 0,1,2 y 4 tiene mucha facilidad, ya que hay pocos números que se parezcan a estos. Sin embargo, la diferencia entre un 3 y un 9 son unos pocos píxeles, así que tiene muchas dificultades y hay que dibujar los números teniendo en cuenta esto.

# CONSEJOS PARA AUMENTAR LA EFICIENCIA
- Está recomendado no usar una brush size menor de 2 para el dibujo general, ya que las imagenes de comparación tienen un grosor similar a la brush size 2 o 3
- Es aconsejable que caracter dibujado no ocupe la enteridad del grid, sino que se asemeje a los ejemplos de dibujos lo máximo posible

# EJEMPLOS DE DIBUJOS


# INSTALACION

El programa está hecho en Python, por lo que es necesario la última versión de [Python](https://www.python.org/downloads/)


Para la parte del canvas y la zona de dibujo, este programa usa la libraría de [pygame](https://pypi.org/project/pygame/), así que también es necesario instalarla: Esto se puede hacer en Windows con la "Command Prompt", pero es necesario el sistema de manejo de paquetes [PIP](https://pip.pypa.io/en/stable/installation/).


Para instalar todo lo anterior basta con pegar los siguientes comandos en la Command Prompt de Windows y tener Python ya instalado

```
curl https://bootst/rap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install pygame
```

Ya están todos los requerimientos instalados!

# EJECUCIÓN

Descargar el .zip y extraerlo en la carpetda de "Downloads" (si se desea cambiar la carpeta, será nacesario cambiar la path de debajo)

Copiar la path de la carpeta. Será algo similar a C:\Users\USUARIO_AQUI\Downloads\OCR

Abrir la consola de comandos (Windows Command Prompt)

```
cd C:\Users\USUARIO_AQUI\Downloads\OCR
python3 main.py
```

Dibujar un número siguiendo las guías anteriores

Disfrutar!

