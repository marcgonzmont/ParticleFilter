# Particle Filter

Para ejecutar el algoritmo implementado se ha de pasar como argumento el directorio donde se tengan las secuencias de prueba, ya sean secuencias de imágenes o vídeos. El programa extraerá los ficheros o los directorios de manera automática. Ej:

`main.py -p <sequences_path>`

Para configurar el algoritmo es necesario abrir el código fuente del script _main.py_. Las variables y sus efectos son las siguientes:

- seq_idx: índice de la secuencia o vídeo a utilizar (por orden alfabético natural
- video: bool que indica si se va a utilizar fichero de vídeo
- N: número de partículas
- show: permite ver la evaluación en el frame actual. También se puede aplicar a las funciones _backSubtrMOG()_, _initialization()_
- debug: permite ver el seguimiento y la población total en el frame actual
- history: controla el ratio de aprendizaje del algoritmo de sustracción de fondo
- count_down: número de frames que se permiten sin detectar el objeto antes de reinicializar la población