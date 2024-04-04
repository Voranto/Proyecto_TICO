#Proyecto Final de TICO. Hecho por: Luca Siegel Moreno
#Funcionamiento: Usando el algoritmo knn(k-nearest neighbors) se puede intentar adivinar utilizando la distancia euclidiana para determinar la variación
#de color de los pixeles a que numero se parece el numero dibujado. Tiene muchas dificultades con ciertos números, como el 9 o el 6, ya que se parecen mucho
#a otros números, como por ejemplo, el 9 se parece mucho a un 4 al dibujarse. 
#
#Ejecución del programa. Está requerido tener instalado la última versión del módulo de pygame, además de incluir la path de los archivos de MNIST en las variables 
#"TEST DIR" y "DATA_DIR". La variable "number_comparisons" determina la precisión y velocidad del programa, siendo inversamente proporcionales 
#(60000) es el valor máximo, y con tal cantidad, el programa tardará entre 10 y 15 segundos en procesarse.
import time
import pygame
from sys import exit
import math
import random
number_comparisons = 60000


#Para comparar la imagen dibujada con el algoritmo de knn, utilizamos la base de datos de MNIST, con una colección de 60000 imagenes de números 
#dibujados a mano, con su etiqueta apropiada.
TEST_DATA_archivo = "t10k-images.idx3-ubyte"
TEST_LABELS_archivo =  "t10k-labels.idx1-ubyte"
TRAIN_DATA_archivo =  "train-images.idx3-ubyte"
TRAIN_LABELS_archivo =  "train-labels.idx1-ubyte"



#Inicializamos Pygame, que es lo que usaremos para la interfaz
pygame.init()

#Inicializamos tambien el sound engine de pygame, para los sonidos
pygame.mixer.init()


#Unas variables clave para altura y anchura de la pantalla
width = 850
height = 784

#esta variable puede parecer redundante, ya que es igual que la anchura, pero me permite poder variar las dimensiones de la pantalla 
#sin arruinar todo el canvas y funciones de dibujo
grid_size = 784

#renderizar la pantalla
screen = pygame.display.set_mode((width,height))



#Título
pygame.display.set_caption("Proyecto Final Tico. Luca Siegel Moreno 2ºBH")



#reloj/framerate
clock = pygame.time.Clock()




#Esta función es la encargada de leer las etiquetas de los números de la base de datos MNIST, ya que la base de datos no está en forma de imagenes, sino
#que su formato es de idx1-ubyte, que consiste de una serie de bytes que podemos traducir a su etiqueta correspondiente
def leer_etiquetas(archivo,n_max_labels = None):
    
    #variable que guarda todas las etiquetas
    labels = [] 
    
    
    with open(archivo, "rb") as f: #abrir el fichero archivo como f, y leerlo en binario ("rb")
        
        #numero inutil (representa algo que no necesitamos)
        _  =f.read(4)
        
        
        #los siguientes 12 bytes representan el numero de imagenes, el numero de filas y de columnas
        n_labels = bytes_to_int(f.read(4))
        #n_max_labels representa el numero de labels que queremos leer (predeterminado 60000)
        if n_max_labels:
            n_labels = n_max_labels
            
            
        #leemos el contenido de 1 byte en 1byte
        for _ in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels


#la variable count es meramente decorativa para saber el numero de comprobaciones ejecutadas cada vez
count = 0


#esta función realiza lo mismo que lo anterior, pero con mayor complejidad, ya que en vez de leerse simples etiquetas(números), estamos interpretando imagenes
#asi que la eficiencia es muy baja, con una complejidad temporal de O(n^3), ya que tenemos 3 "for" loops, para iterar a traves de cada imagen,
#fila y columna
def leer_imagenes(archivo,max_num_imgs = None):
    global count
    
    #matriz 3D que guarda todas las imagenes
    images = [] 
    with open(archivo, "rb") as f: #abrir el fichero  como f, y leerlo en binario ("rb")
        _  =f.read(4) #numero inutil (representa algo que no necesitamos)
        
        
        #los siguientes 12 bytes representan el numero de imagenes, el numero de filas y de columnas
        num_imgs = bytes_to_int(f.read(4))
        
        
        #max_num_imgs representa el numero de imagenes que queremos leer (predeterminado 60000)
        if max_num_imgs:
            num_imgs = max_num_imgs
            
        #el numero de filas y columnas son los proximos 8 bytes
        n_filas = bytes_to_int(f.read(4))
        n_columnas = bytes_to_int(f.read(4))
        for img_idx in range(num_imgs):
            image = []#variable que guarda la imagen actual
            for _1 in range(n_filas):
                row = []#variable que guarda la columna actual
                for _ in range(n_columnas):
                    count += 1
                    pixel = f.read(1) #leemos el pixel actual de 8 bits y lo apendizamos a la row
                    row.append(pixel)
                image.append(row)#metemos la row en la image
            images.append(image)#metemos la image en el conjunto de images
    #devolvemos la array 3D
    return images




#ya que los valores de los colores de la training data esta en binario, utilizamos esta funcion para pasar de bytes a int, y usamos una protección para
#comprobar que el input no es un int, y que si sea un int, solo se devuelva en forma de número
def bytes_to_int(byte_data):
    if type(byte_data) == int:
        return byte_data
    else:
        return int.from_bytes(byte_data,"big")



#Esta función es encargada de transformar la matriz 2D que hemos dibujado en nuestra pantalla de pygame y pasarlo a una lista unidimensional, para mayor facilidad 
#al procesar por el algoritmo knn
def pasar_lista_unidimensional(X):
    lista = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            lista.append(X[i][j])
    return [lista]



#Ya que el formato de el dibujo nuestro y el de la base de datos MNIST es ligeramente distinto(tridimensional), tengo estas próximas dos funciones para traspasar la
# matriz de la base de datos MNIST a una matriz unidimensional, para una vez mas facilitar los cálculos de las distancias euclidianas
def pasar_lista_unidimensional_MNIST(X):
    return [aplanar_lista(sample) for sample in X]
def aplanar_lista(l):
    return [pixel for sublist in l for pixel in sublist]



#Fórmula simple de la distancia euclidiana entre dos pixeles, "x_i", y "y_i", que surgen de el comando zip(x,y), que es una función que recoge dos iterables 
#(listas) y devuelve sus valores de uno en uno de manera simultanea, en forma de tuplas 
def dist(x,y):
    distancias = []
    #distancias euclidianas con sqrt((a1-b1)^2 + (a2-b2)^2...etc)
    for x_i,y_i in zip(x,y):
        distancias.append((bytes_to_int(x_i) - bytes_to_int(y_i)) **2)
    return sum(distancias)**0.5


#Aqui estamos comparando la imagen dibujada con TODAS las imagenes en X_train, que son las imagenes de la base de datos
def distancia_entre_samples(X_train,test_sample):
    return [dist(train_sample,test_sample) for train_sample in X_train] #por todas las imagenes, calculamos su distancia arriba



#El algoritmo knn devuelve los k numeros mas parecidos al dibujado, esta funcion simplemente recoge el valor del más frecuente
def most_frequent_element(lista):
    return max(lista, key= lista.count)




#Este es el algoritmo principal que usamos para las comparaciones
def knn(X_train,y_train,X_test, k = 3):
    
    #Y_pred es nuestra predicción del número ( ponemos -1 como placeholder)
    y_pred = -1 
    

    print("Aplicando el algoritmo knn (llevará un rato)...")
    
    #iteramos a traves de X_test ( que es la matriz 2D de nuestro dibujo), con la funcion "enumerate", que devuelve el valor en ese punto de la lista"test sample"
    #junto con su índice "test_sample_idx"
    for test_sample_idx,test_sample in enumerate(X_test):
        
        #Calculamos la distancia 
        training_distancias = distancia_entre_samples(X_train,test_sample) #queremos conseguir las distancias a todos los puntos
        print(training_distancias)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(enumerate(training_distancias), key = lambda x: x[1]) ]#ordenamos las distancias de menor a mayor
        
        
        candidates = [bytes_to_int(y_train[idx]) for idx in sorted_distance_indices[:k]] # k mejores candidatos con menor distancia
        print("K-NEAREST NEIGHBORS WERE: ", candidates)
        
        #Escogemos el mas frecuente entre todos los candidatos
        top_candidate = most_frequent_element(candidates)
        
        
        y_pred = top_candidate #apuntamos a predicción
    return y_pred


#Función principal, se encarga de toda la preparación previa antes de usar el algoritmo
def main():
    global X_test
    
    #comenzamos a trackear el tiempo
    start_time = time.time()
    print("Leyendo los archivos de entrenamiento...")
    
    
    #"X" es igual a las imagenes y "y" es el label asignado dentro la database MNIST
    #La variable "number_comparisons" es la asignada antes para saber con cuantas imagenes comparamos nuestro dibujo
    X_train = leer_imagenes(TRAIN_DATA_archivo,number_comparisons)
    y_train = leer_etiquetas(TRAIN_LABELS_archivo,number_comparisons)

    
    print("Convirtiendo el dibujo a una grid bidimensional...")
    
    #queremos pasar la matriz de valores de nuestro dibujo a una matriz unidimensional
    X_train = pasar_lista_unidimensional_MNIST(X_train) 
    
    #Activamos el algoritmo knn con los datos procesados anteriormente
    y_pred = knn(X_train,y_train,X_test,5)
    
    #Terminamos de medir el tiempo
    end_time = time.time()
    
    
    #Devolvemos resultado
    print("NÚMERO DE COMPARACIONES: ", count)
    print("TIEMPO QUE TARDÓ EN PROCESAR:", end_time-start_time)
    print("EL NÚMERO QUE ACABAS DE ESCRIBIR(posiblemente): " ,y_pred)
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#A partir de aquí termina la parte del algoritmo en sí, y comienza la parte de renderizados y dibujos, utilizando el módulo de pygame


#Iconos de la goma y el lápiz para permitir borrar y dibujar en el canvas
pencil = pygame.image.load("sprites/pencil.png").convert_alpha()
#Utilizamos la funcion de scale para darle a los los sprites ( imagenes), el tamaño deseado
pencil = pygame.transform.scale(pencil, (30,30))


eraser = pygame.image.load("sprites/eraser.png").convert_alpha()
eraser = pygame.transform.scale(eraser,(30,30))
#usamos la funcion rotate para rotar la imagen de la coma para que quepa en el marco
eraser = pygame.transform.rotate(eraser,45)


marco = pygame.image.load("sprites/marco.png").convert_alpha()
marco = pygame.transform.scale(marco,(60,60))


clearcanvas = pygame.image.load("sprites/clearcanvas.png").convert_alpha()
clearcanvas = pygame.transform.scale(clearcanvas,(45,45))


marco_process = pygame.image.load("sprites/marco_process.png").convert_alpha()
marco_process = pygame.transform.rotate(marco_process,90)
marco_process = pygame.transform.scale(marco_process,(70,200))


background = pygame.image.load("sprites/background.png").convert_alpha()



#sonidos de los botones
boton1_sound = pygame.mixer.Sound("sounds/boton1.mp3")
boton2_sound = pygame.mixer.Sound("sounds/boton2.mp3")
boton3_sound = pygame.mixer.Sound("sounds/boton3.mp3")

#al dibujar, queremos un sonido corto que se asemeje de manera relativa al dibujo, por lo cual tenemos 3 sonidos distintos que escogemos aletoriamente
drawing_sound = [pygame.mixer.Sound("sounds/drawing_sound.mp3"),pygame.mixer.Sound("sounds/drawing_sound_2.mp3"),pygame.mixer.Sound("sounds/drawing_sound_3.mp3")]

#bajamos el volumen a todos los sonidos con la funcion set_volume a un 10% de su sonido original
for sound in drawing_sound:
    sound.set_volume(0.1)



#la bola que se mueve al ajustar el brush size
brush_size_ball = pygame.image.load("sprites/brush_size_ball.png")
brush_size_ball = pygame.transform.scale(brush_size_ball,(20,20))


#Para los elementos móviles, escogemos una variable predeterminada para poder moverlos y cambiar sus valores fácilmente
marco_pos = [790,5]
brush_size_pos = [807,375,20,20]

#Voy a asignar la variable "brush_size" con varios parametros arbitrarios. 1 representa el mas pequeño, con un brush size de 1x1, luego el valor 2
#representa una cruz de tamaño 3x3, que va a ser lo predeterminado, y para terminar estara el brush size 3, que representa un 3x3
brush_size = 2



#Inicializamos la variable image_array que contendrá la matriz de 28x28 que contendrá el color de cada pixel de nuestro dibujo
image_array = []  

#Inicializamos dos variables para dos fonts de distinto tamaño, "font" y "font_small"
font = pygame.font.Font(None, 50) 
font_small = pygame.font.Font(None, 25) 


#Dibujamos unas lineas para que el usuario tenga una mejor idea de donde dibujar. Usamos dos bucles para iterar a traves del eje "x" y del eje "y".
for i in range(0,grid_size+28,28):
    pygame.draw.line(screen,"white",(0,i),(784,i))



#Mismo bucle en el eje X
#Aprovechamos uno de estos loops para llenar la matriz "image_array", con todo 0s, que es el valor predeterminado(corresponde al color negro)
for j in range(0,grid_size,28):
    image_array.append([0]*28)
    pygame.draw.line(screen,"white",(j,0),(j,784))
    
    

#Funcion encargada de dibujar o borrar el pixel, recibiendo de input las coordenadas del ratón en ese momento
#Para facilitar el dibujar, el grosor del pincel predeterminado(brush size = 2) es de una cruz de 3x3, así que necesitamos 5 dibujos cada vez que
# hacemos click, pero esto puede darle problemas a la matriz, ya que si es de tamaño 28, y estas con el raton en las casilla 28, se produce un error ya 
# que el pincel intentadibujar en la casilla 29 ( que no existe). Para evitar esto hacemos unos checks individuales para verificar que nos encontramos dentro 
# de estos parámetros. Para poder usar la misma funcion para ambos dibujar y borrar, usamos como input, ademas de las coordenadas, el nombre del color
#(blanco para dibujar, negro para borrar), y el color (255 para dibujar, 0 para borrar)
def drawerase(x,y,color_name,color_value):
    pygame.draw.rect(screen,color_name,(x-x%28,y-y%28,28,28)) 
    image_array[math.trunc(y/28)][math.trunc(x/28)] = color_value
    #si la brush size es mayor que 1, implica que tenemos que dibujar más pixeles ademas de ese único ( la cruz de 3x3)
    if brush_size > 1:
        
        #pixel de la derecha
        if math.trunc(x/28)+1 < 28: 
            pygame.draw.rect(screen,color_name,(x-x%28+28,y-y%28,28,28)) 
            image_array[math.trunc(y/28)][math.trunc(x/28)+1] = color_value
        
        #pixel izquierda
        if math.trunc(x/28)-1 >= 0:  
            pygame.draw.rect(screen,color_name,(x-x%28-28,y-y%28,28,28))
            image_array[math.trunc(y/28)][math.trunc(x/28)-1] = color_value
        
        #pixel debajo
        if math.trunc(y/28)+1 < 28: 
            pygame.draw.rect(screen,color_name,(x-x%28,y-y%28+28,28,28)) 
            image_array[math.trunc(y/28)+1][math.trunc(x/28)] = color_value
        
        
        #pixel encima
        if math.trunc(y/28)-1 >= 0: 
            pygame.draw.rect(screen,color_name,(x-x%28,y-y%28-28,28,28)) 
            image_array[math.trunc(y/28)-1][math.trunc(x/28)] = color_value
            
        #si la brush size es mayor que 2, es decir, 3, hay que dibujar el cuadrado de 3x3 entero
        if brush_size > 2:
            
            #pixel abajo derecha
            if math.trunc(x/28)+1 < 28 and math.trunc(y/28)+1 < 28: 
                pygame.draw.rect(screen,color_name,(x-x%28+28,y-y%28+28,28,28)) 
                image_array[math.trunc(y/28)+1][math.trunc(x/28)+1] = color_value
            
            #pixel arriba izquierda
            if math.trunc(x/28)-1 >= 0 and math.trunc(y/28)-1 >= 0:  
                pygame.draw.rect(screen,color_name,(x-x%28-28,y-y%28-28,28,28))
                image_array[math.trunc(y/28)-1][math.trunc(x/28)-1] = color_value
            
            #pixel abajo izquierda
            if math.trunc(y/28)+1 < 28 and math.trunc(x/28)-1 >= 0: 
                pygame.draw.rect(screen,color_name,(x-x%28-28,y-y%28+28,28,28)) 
                image_array[math.trunc(y/28)+1][math.trunc(x/28)-1] = color_value
            
            
            #pixel arriba derecha
            if math.trunc(y/28)-1 >= 0 and math.trunc(x/28)+1 < 28: 
                pygame.draw.rect(screen,color_name,(x-x%28+28,y-y%28-28,28,28)) 
                image_array[math.trunc(y/28)-1][math.trunc(x/28)+1] = color_value





#Funcion para vaciar el canvas al completo
def clearcanvasfunction():
    global image_array
    #limpiamos el canvas entero dibujando un cuadrado de las coordenadas de la grid entera que lo ocupe tood
    pygame.draw.rect(screen, "black",(0,0,784,784))
    
    #vaciamos la array con los valores de los colores de los pixeles
    image_array = []
    
    #Rehacemos las grid lines, ya que se han borrado con el cuadrado dibujado previamente
    for i in range(0,grid_size+28,28):
        pygame.draw.line(screen,"white",(0,i),(784,i))
    for j in range(0,grid_size,28):
        #recompletamos la image_array
        image_array.append([0]*28)
        pygame.draw.line(screen,"white",(j,0),(j,784))
        



#antes de procesar la imagen, quiero que los bordes de el dibujo se vuelvan grises, para tratar de mejorar la eficiencia del programa,
#ya que de tal manera, se volverá mas similar a la base de datos
def grayscale(array):
    
    #iteramos a traves de todas las columnas
    for i in range(len(array[0])):
        #iteramos a traves de todas las filas
        for j in range(len(array)):
            
    
            #si el valor es distinto de 0, es que está coloreado, y le tendriamos que asignar un valor mas grisaceo (200), para 
            #que sea mas similar a la base de datos MNIST            
            if array[i][j] != 0:
                
                #si el pixel coloreado se encuentra en el borde de la grid, es parte del borde del dibujo necesariamente
                if i == 0 or j == 0 or i == len(array)-1 or j == len(array)-1:
                    array[i][j] = 200
                else:
                    
                    #sino, comprobamos si hay alguno de los pixeles en sus alrededores que no esté dibujado, es decir, 
                    #que su color sea 0
                    if array[i+1][j] == 0 or array[i-1][j] == 0 or array[i][j+1] == 0 or array[i+1][j-1] == 0:
                        array[i][j] = 200
                        
    #devolvemos la array inicial
    return array


  
#Esta variable "boton" sirve para que el dibujo no tenga que estar hecho con clicks individuales, sino que mientras se mantenga el click apretado se mantenga
#dibujando . Es una variable de estado
button_pressed = False


#Esta variable determina entre dibujar y borrar
drawing_mode = "draw"

#asignamos a variables todos los textos
process_drawing_text = font.render("Process", True, (0, 0, 0))
process_drawing_text = pygame.transform.rotate(process_drawing_text,-90)
#le asignamos un "rect" a el boton de procesar el drawing, para que se puedan detectar colisiones entre el raton y este boton. Los valores dentro de un rect son
#[x,y,anchura,altura]
process_drawing_rect = [800,600,50,150]
brush_size_text_1 = font_small.render("Brush", True, (0, 0, 0))
brush_size_text_2 = font_small.render("Size", True, (0, 0, 0))
clear_text = font_small.render("Clear",True,(0,0,0))

#para que no se ejecute el sonido de dibujar cada vez que tengas el boton apretado, que sería 60 veces por segundo (la framerate), hay un cooldown
#puesto para que solo se ejecute una vez cada 1 segundo aproximadamente, solo si se tiene el boton del raton apretado
cooldown_draw_sound = 0    



#Este es el bucle principal que gestiona todas las comprobaciones sobre si hay click en el canvas o en el botón, y para que se pueda cerrar el juego
while True:

    #creamos un rectangulo invisible alrededor del ratón para verificar colisiones con distintos botones. Hacemos que se actualize su posicion cada vez
    #que se ejecute el bucle
    mouse_rect = pygame.Rect(pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1],20,20)
    
    #Creamos el rectangulo del marco de los botones para comprobar colision y click con el raton
    eraser_rect = pygame.Rect(795,75,50,50)
    pencil_rect = pygame.Rect(800,20,50,50)
    clearcanvas_rect = pygame.Rect(795,180,50,50)
    
    
    #para que se pueda cerrar el programa y detectar eventos como botones del raton
    for event in pygame.event.get():
        
        #Si le damos al boton de cerrar, que se cierre
        if event.type == pygame.QUIT:
            pygame.QUIT()
            exit()
        
        
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            #Si apretamos el boton de click "hacia abajo", que la variable de estado se active
            button_pressed = True
            
            #recogemos la posicion del raton para comprobaciones
            pos = pygame.mouse.get_pos()
            
            
            #Si el boton procesor colisiona con el rectangulo del raton *mientras* hay un click del raton, ejecutar el algoritmo KNN
            if pygame.Rect.colliderect(mouse_rect,process_drawing_rect):
                
                #convertir la matriz en otra matriz con bordes mas grises para optimizar el algoritmo
                image_array = grayscale(image_array)
                
                #Cogemos la variable "image_array", que si recordamos, era la que gestionaba nuestra zona de dibujo, y la convertimos a una lista
                #unidimensional y la igualamos a la variable "X_test", que es la que usabamos en el algoritmo knn
                X_test = pasar_lista_unidimensional(image_array)
                print("Cargando...")
                
                
                #Ejecutamos el sonido de hacer click al boton
                boton2_sound.play()
                
                
                #ejecutar el algoritmo principal
                main()
                
            
            
            #si colisiona con el lapiz, ejecutar lo siguiente
            elif pygame.Rect.colliderect(mouse_rect,pencil_rect):
                
                #actualizar la posicion del marco a la zona donde esta el lapiz
                marco_pos = [790,5]
                
                #si ya estabamos en modo dibujar, no queremos que se ejecute el sonido, seria redundante
                if drawing_mode != "draw":
                    boton1_sound.play()
                    
                #cambiar el modo a dibujar
                drawing_mode = "draw"
            
            
            #exactamente el mismo codigo que con el lapiz, pero con coordenadas y sonidos distintos
            elif pygame.Rect.colliderect(mouse_rect,eraser_rect):
                marco_pos = [790,65]
                if drawing_mode != "erase":
                    boton1_sound.play()
                drawing_mode = "erase"
            
            
            #si la colision es con el boton de "Clear Canvas", que se ejecute el sonido de clear canvas
            elif pygame.Rect.colliderect(mouse_rect,clearcanvas_rect):
                boton3_sound.play()
                
                #que se ejecute la funcion de vaciar el canvas
                clearcanvasfunction()
                
                
            #si hacemos click con la bolita de cambiar el brush size
            elif pygame.Rect.colliderect(mouse_rect,brush_size_pos):
                
                #queremos guardar el drawing_mode anterior, porque en cuanto se levante el boton del raton queremos que 
                #se vuelva al modo anterior
                previous_mode = drawing_mode
                #cambiarmos el drawing_mode
                drawing_mode ="brush_size"
                
            
        
        
        #Checkear si se ha levantado el boton
        elif event.type == pygame.MOUSEBUTTONUP:
            if drawing_mode == "erase":
                
                
                #Al usar el boton de "borrar", lo que verdaderamente ocurre es que se dibuja una casilla negra en ese lugar, por lo que 
                # se borran las grid lines. Para ambos optimizar el codigo y que visualmente sea más agradable, tengo establecido que solo se
                #dibujen estas grid lines una vez se haya levantado el click derecho 
                for i in range(0,grid_size+28,28):
                    pygame.draw.line(screen,"white",(0,i),(784,i))
                for j in range(0,grid_size+28,28):
                    pygame.draw.line(screen,"white",(j,0),(j,784))
                    
            #si el modo era "brush_size", hay que restablecer el drawing_mode al anterior antes de activar el brush_size
            elif drawing_mode == "brush_size":
                drawing_mode = previous_mode
            
            #desactivar la variable de estado de tener el boton apretado
            button_pressed = False
            
            
    #Si estamos manteniendo el click del ratón, que se dibuje
    if drawing_mode == "draw" and button_pressed == True:
        #Recogemos las coordenadas del ratón
        pos = pygame.mouse.get_pos()
        
        
        
        #Si el eje y es menor de 756, implica que estamos dentro de la zona de dibujo y no dentro de la zona del botón, asi que activamos la función
        #de dibujar en esas coordenadas
        if pos[1] < 784 and pos[0] < 784:
            drawerase(pos[0],pos[1],"white",255)
            
            #si el cooldown del sonido de dibujo es 0, activamos uno de los sonidos aleatoriamente
            if cooldown_draw_sound == 0:
                drawing_sound[random.randint(0,2)].play()
                
                #restablecemos el contador a 60, para que tarde 60 ciclos(1 segundo), en poder volverse a ejecutar
                cooldown_draw_sound = 60
                
    #exactamente el mismo codigo pero para borrar,  con color negro y valor del color "0"
    elif drawing_mode == "erase" and button_pressed == True:
        pos = pygame.mouse.get_pos()
        if pos[1] < 784 and pos[0] < 784:
            drawerase(pos[0],pos[1],"black",0)
            
            
    #Si estamos en el modo de brush size y tenemos el click apretado, queremos que se mueva la bolita del brush size pero solo en el eje "y", y solo dentro de ciertos
    #parametros arbitrarios
    elif drawing_mode == "brush_size" and button_pressed == True:
        #cogemos posicion del raton
        pos = pygame.mouse.get_pos()
        
        #si estamos en esos parametros, que se mueva en el eje y con las mismas coordenadas del raton
        if 300 < pos[1] < 430:
            brush_size_pos[1] = pos[1]
            
            #valores arbitrarios para establecer cuando es cada brush_size
            if brush_size_pos[1] < 340:
                brush_size = 3
            elif 330 <= brush_size_pos[1] <= 390:
                brush_size = 2
            elif brush_size_pos[1] > 380:
                brush_size = 1
                
        #si las coordenadas se salen de los parametros establecidos, queremos que no los superen, sino que se queden en el maximo
        elif pos[1] >= 430:
            brush_size_pos[1] = 430
            brush_size = 1
        else:
            brush_size_pos[1] = 300
            brush_size = 3
    
    
    #al final de cada bucle le restamos 1 al contador de sonido de dibujar para ir restableciendo el contador
    if cooldown_draw_sound > 0:
        cooldown_draw_sound -= 1    
        
        
    
    #vaciar la iteracion anterior del la GUI
    screen.blit(background,(784,0)) 
     

    #renderizado de los textos
    screen.blit(marco_process,(788,570))
    screen.blit(process_drawing_text,process_drawing_rect)
    screen.blit(brush_size_text_1, (790,460))
    screen.blit(brush_size_text_2, (795,480))
    
    #como el texto de brush_size_3 es el numero, el cual varía, tenemos que actualizarlo en cada iteracion para comprobar que no ha cambiado
    brush_size_text_3 = font.render(str(brush_size), True, (0, 0, 0))
    screen.blit(brush_size_text_3,(807,500))
    screen.blit(clear_text,(793,233))
    
    
    #renderizado de el brush size tool
    pygame.draw.line(screen,"black",(815,300),(815,450),4)
    screen.blit(brush_size_ball,brush_size_pos)
    
    #renderizar marcos de los iconos
    screen.blit(marco,marco_pos)
    
    
    #renderizar los iconos de la goma y el lápiz
    screen.blit(pencil,pencil_rect)
    screen.blit(eraser,eraser_rect)
    screen.blit(clearcanvas,clearcanvas_rect)
    
    
    #Función encargada de actualizar el display de pygame, para que en vez de renderizar cosas únicas y estáticas, poder tener un canvas dinámico
    pygame.display.update()
    
    
    #framerate
    clock.tick(240)
    
    
