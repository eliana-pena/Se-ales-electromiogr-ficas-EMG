# Problema-de-Coctel
**Fecha:** 8999997777 de septiembre de 2024  
**Autor:** Juan Andres Torres, Julián Rodríguez y Eliana Peña

# Descripción
En este laboratorio, titulado "Señales ", el objetivo principal es 

# Tabla de Contenidos
Lista de secciones del informe con enlaces a cada una (opcional pero útil).
1. [Introducción](#introducción)
2. [Metodología](#metodología)
3. [Desarrollo](#desarrollo)
4. [Resultados](#resultados)
5. [Discusión](#discusión)
6. [Conclusión](#conclusión)
7. [Referencias](#referencias)
8. [Anexos](#anexos)
   
# Introducción
El procesamiento de señales de audio es clave en muchas aplicaciones tecnológicas, como el reconocimiento de voz y la mejora de grabaciones. En este laboratorio, se trabajó con grabaciones de voces y sonidos ambientales capturadas con tres teléfonos móviles, con el objetivo de aplicar técnicas de Análisis de Componentes Independientes (ICA) para separar las fuentes de sonido y aislar al menos una voz.

Las grabaciones, realizadas a 44 kHz, presentaron desafíos técnicos como desfases entre señales debido a las diferencias en los micrófonos y tiempos de captura. Se espera mejorar la separación de señales y obtener una buena relación señal-ruido (SNR), evaluando la calidad de la separación obtenida.

# Metodología

El siguiente laboratorio presenta el análisis en frecuencias de las señales de voz, esto por medio del problema presentado por la fiesta de coctel, el problema plantea una situación social donde se colocaron varios micrófonos. Sin embargo, se solicita escuchar la voz de uno de los participantes, a pesar de que los receptores captaron varias fuentes de sonido. En este caso, se colocaron 3 micrófonos que detectarán al mismo tiempo la voz de 3 personas diferentes (los emisores), hablando a la vez como es evidenciado en la la figura 1. 

![coc2](https://github.com/user-attachments/assets/7445aaac-2087-4dfc-8297-4995274543e2)

Figura 1: Es un diagrama que demuestra la posicion de los emisores y receptores de sonido.

**Configuración del sistema**
La grabación se realizó en una sala insonorizada, con el objetivo de minimizar las interferencias y reflejos de sonido. Se emplearon tres micrófonos de tipo mono, provenientes de diferentes teléfonos móviles, colocados de manera paralela a las fuentes de sonido. Cada fuente de sonido estaba representada por una persona hablando, y la disposición espacial fue la siguiente:

- Persona 3 se encontraba frente al Micrófono 3 a una distancia de 1,16 cm.
- Persona 1 estaba al lado de Persona 2 a 1,16 cm de distancia, y frente a ella, a 1,16 cm, se encontraba el Micrófono 2.
- Persona 2 estaba situada al lado de Persona 1, con una distancia de 1,16 cm, y frente a ella, también a 1,16 cm, estaba el Micrófono 1.
Tanto los micrófonos como las fuentes de sonido fueron alineados de forma paralela, siguiendo las indicaciones del docente, sin tener en cuenta criterios especiales para evitar ecos o reflexiones, debido al acondicionamiento insonorizado de la sala. La orientación de los micrófonos fue directamente hacia las fuentes de sonido, asegurando una captura clara y directa de las voces.

**Captura de la señal**
Para esta grabación, el sistema de adquisición se utlizo la aplicación movil Recforge II, estuvo conformado por tres micrófonos mono provenientes de teléfonos móviles diferentes. Los micrófonos capturaron las señales de audio de manera individual, sin realizar un ajuste previo en las características técnicas de cada dispositivo, lo que resulta en posibles variaciones entre las respuestas de los micrófonos.

Se configuró la captura de audio con una frecuencia de muestreo de 44KHz, garantizando que se cumpla con el teorema de Nyquist y se capture adecuadamente el rango de frecuencias de las voces humanas. El tiempo de captura fue de aproximademente de 5 segundos, lo que permitió obtener una muestra significativa para el análisis posterior. Aunque el nivel de cuantificación no se ajustó entre los diferentes teléfonos, se espera que el rango dinámico de cada dispositivo haya sido suficiente para evitar saturación o pérdida de información en las señales.

El SNR (relación señal-ruido) inicial de cada grabación fue calculado para verificar la calidad de las señales capturadas. Esta relación es crítica, ya que un SNR bajo podría complicar la separación de fuentes en las etapas posteriores. 

En el presente laboratorio se usaron las siguientes librerias:

- **NumPy**: `import numpy as np`  
- **SoundFile**: `import soundfile as sf`  
- **scikit-learn (FastICA)**: `from sklearn.decomposition import FastICA`  
  Implementa el algoritmo FastICA para el Análisis de Componentes Independientes, una técnica utilizada para separar señales mezcladas.
- **Matplotlib**: `import matplotlib.pyplot as plt`  
- **SciPy (FFT)**: `import scipy.fftpack as fft`  
  Proporciona funciones para calcular la Transformada Rápida de Fourier, útil para analizar los componentes de frecuencia de las señales.
- **Librosa**: `import librosa` y `import librosa.display`  
  Un paquete de Python para el análisis de música y audio. Proporciona bloques de construcción para crear sistemas de recuperación de información musical.
- **SciPy (Estadística)**: `from scipy.stats import pearsonr`  
  Se utiliza para calcular los coeficientes de correlación de Pearson.
- **SciPy (IO)**: `from scipy.io import wavfile`  
- **SciPy (Procesamiento de Señales)**: `from scipy.signal import butter, lfilter`  
  Proporciona funciones para diseñar y aplicar filtros digitales.

# Desarrollo
A continuacion se presenta el desarrollo de objetivo del laboratorio por medio de herramientas de python: 

### Carga de los audios
Primero, para cargar los archivos existentes, se creó una carpeta que contenía tanto el programa como los audios correspondientes y para cargarlos al codigo se utilizó la librería  `Soundfile`.  
Las variables dato se utilizan para almacenar los audios en un arreglo, y Fs nos indica la frecuencia de muestreo de estos.
```python
#Audios

dato1, Fs = sn.read('audiomass-output (2).wav')
dato2, Fs2 = sn.read('audiomass-output (3).wav')
dato3, Fs3 = sn.read('audiomass-output (1).wav')

#Ruidos

ruido1, Fs4 = sn.read('Ruido2.wav')
ruido2, Fs4 = sn.read('Ruido 1 celular 1-corte.wav')
ruido3, Fs4 = sn.read('r1celular1.wav')
```
### Calculo del SNR
Primero se verifico por medio de un condicional que todas las frecuencias de muestreo sean las mismas.
Para calcular la relación señal-ruido (SNR), se definió una función llamada `calcular_snr`, en la cual se evaluaron los audios junto con sus respectivos ruidos.
```python
# Verificar que las frecuencias de muestreo sean iguales
if not (Fs == Fs2 == Fs3 == Fs4 == Fs5):
    raise ValueError("Las frecuencias de muestreo no son iguales. No se puede continuar.")

# Calcular SNR
def calcular_snr(senal, ruido):
    potencia_senal = np.mean(senal ** 2)
    potencia_ruido = np.mean(ruido ** 2)
    snr = 10 * np.log10(potencia_senal / potencia_ruido)
    return snr

# SNR para cada micrófono
snr1 = calcular_snr(dato1, ruido2)
print(f"La relación señal a ruido (SNR) del micrófono 1 es: {snr1:.2f} dB")

snr2 = calcular_snr(dato2, ruido2)
print(f"La relación señal a ruido (SNR) del micrófono 2 es: {snr2:.2f} dB")

snr3 = calcular_snr(dato3, ruido2)
print(f"La relación señal a ruido (SNR) del micrófono 3 es: {snr3:.2f} dB")
```
### Análisis temporal y espectral
Para el análisis temporal, se definió la función `analisis_temporal_espectral`. En esta función, `senal` representa la señal de audio que se va a analizar, y `fs` es la frecuencia de muestreo de la señal expresada en Hz.
```python
def analisis_temporal_espectral(senal, fs):
    tiempo = np.arange(len(senal)) / fs
    plt.figure(figsize=(14, 5))
```
Graficamos el análisis temporal de la señal usando la libreria `Matplotlib`.

```python
    plt.subplot(1, 2, 1)
    plt.plot(tiempo, senal)
    plt.title(f'Análisis Temporal - {titulo}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
```
Para el análisis espectral de frecuencias, se utilizó la Transformada Rápida de Fourier (FFT). Para calcularla se usaron las funciónes `fftfreq` y `fft` de la librería `scipy.fftpack`, asi como la funcion `abs` de `numpy`.

```python
    freqs = fft.fftfreq(len(senal), 1/fs)
    fft_senal = np.abs(fft.fft(senal))
    plt.subplot(1, 2, 2)
    plt.plot(freqs[:len(freqs)//2], fft_senal[:len(freqs)//2])
    plt.title(f'Análisis Espectral - {titulo}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')

    plt.tight_layout()
    plt.show()

analisis_temporal_espectral(dato1, Fs, "Micrófono 1")
analisis_temporal_espectral(dato2, Fs, "Micrófono 2")
analisis_temporal_espectral(dato3, Fs, "Micrófono 3")
```
### Separacion de fuentes-ICA
La separación de fuentes se realizó utilizando el método FastICA (Análisis de Componentes Independientes), con el objetivo de extraer las señales de cada persona hablante a partir de las grabaciones mezcladas obtenidas por los micrófonos. 
Puesto que una señal tenia mayor longitud con respecto a las otras, se opto por escalonarlas a la misma longitud usando `min_length`.

```python
min_length = min(len(dato1), len(dato2), len(dato3))
dato1 = dato1[:min_length]
dato2 = dato2[:min_length]
dato3 = dato3[:min_length]
```

 Se Combinaron las tres señales en una sola matriz.
```python
X = np.c_[dato1, dato2, dato3]
````
Mediante FastICA de `sklearn.decomposition`, se creó un objeto ICA para separar tres componentes independientes. La función `ica.fit_transform(X)` ajusta el modelo ICA a la matriz de señales X y transforma X en señales separadas, que se almacenan en la matriz `S_`. Esta matriz `S_` tiene el objetivo de convertir las señales separadas a un rango de amplitud adecuado para el formato de audio de 16 bits.
```python 
ica = FastICA(n_components=3, random_state=0)
S_ = ica.fit_transform(X)
S_ = (S_ / np.max(np.abs(S_))) * 32767
````
A traves de la libreria `scipy.io` se generaron los audios separados en al carpeta de origen.
```python
wavfile.write("separadita1.wav", Fs, S_[:, 0].astype(np.int16))
wavfile.write("separadita2.wav", Fs2, S_[:, 1].astype(np.int16))
wavfile.write("separadita3.wav", Fs2, S_[:, 2].astype(np.int16))
````
Se graficaron las señales mezcladas y separadas.
```python
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(X)
plt.title("Señales mezcladas")
plt.subplot(3, 1, 2)
plt.plot(S_)
plt.title("Señales separadas")
plt.show()
```
### Comprobación de separación de fuentes
Para llevar esto a cabo se utilizaron dos señales, una de referencia y una señal separada. En este caso se presentaron los espectros de frecuencia de ambas señales para observar si se presentan diferencias usando la libreria `Librosa`.
```python

plt.figure(figsize=(10, 6))
librosa.display.specshow(S_referencia_db, sr=Fs, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma - Señal de Referencia')
plt.show()

# Espectrograma de una de las señales separadas
S_separada = librosa.stft(S_.T[0])  # Ejemplo con la primera señal separada
S_separada_db = librosa.amplitude_to_db(np.abs(S_separada), ref=np.max)

plt.figure(figsize=(10, 6))
librosa.display.specshow(S_separada_db, sr=Fs, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma - Señal Separada 1')
plt.show()

```
Calculamos el coeficiente de correlación de Pearson para medir la similitud entre la señal de referencia y una de las señales separadas obtenidas por ICA a traves de la libreria `scipy.stats`. Para que esto se llevara a cabo fue necesario calcular las señales en las dimensiones adecuadas.
```python
# Obtén la longitud mínima entre las dos señales
min_length = min(len(S_referencia.flatten()), len(S_.T[2].flatten()))

# Recorta la señal de referencia a la longitud mínima
S_referencia_flat = S_referencia.flatten()[:min_length]
S_T0_flat = S_.T[2].flatten()

# Convierte las señales a sus partes reales
S_referencia_real = S_referencia_flat.real
S_T0_real = S_T0_flat.real

# Calcula la correlación
correlacion = pearsonr(S_referencia_real, S_T0_real)
print(f"Coeficiente de correlación: {correlacion[0]:.4f}")

```
### Filtro Pasa-banda
Se le aplico un filtro pasa-banda a la señal resultante del metodo de separacion de fuentes ICA. Se implementaron dos funciones: `butter_bandpass` la cual genera un filtro pasa banda Butterworth y calcula los coeficientes `a` y `b`. La otra funcion es `bandpass_filter` que aplica el filtro pasa banda a una señal de entrada utilizando los coeficientes calculados previamente. Por otro lado, las variables `lowcut` y `highcut` corresponden a las frecuencias de corte inferior y superior para el filtro pasa banda.

```python
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#definimos los dominios de frecuencia 
lowcut =3000
highcut =3200

#aplicamos el filtro
filtered_audio = bandpass_filter(S_[:, 1].astype(np.int16), lowcut, highcut, Fs, order=6)

# Guardamos la señal filtrada
filtered_audio_path = 'filtered_audio.wav'
wavfile.write(filtered_audio_path, Fs, filtered_audio.astype(np.int16))
```

# Resultados
El presente análisis se centra en la evaluación de los resultados obtenidos del código de separación de señales mediante la técnica FastICA y su comparación con la calidad esperada en base a los indicadores de SNR y correlación.

### 1. **Relación Señal a Ruido (SNR)**

Para evaluar la calidad de las señales capturadas, se calculó la SNR de las señales provenientes de tres micrófonos:

- Micrófono 1: 34.88 dB
- Micrófono 2: 26.54 dB
- Micrófono 3: 49.77 dB

<p align="center">
   <img src="https://github.com/user-attachments/assets/776bd5bc-5715-4d09-b250-9f06a54bd333" alt="Descripción de la imagen" width="500">  
</p>

<p><em>"La imagen muestra las señales capturadas por los tres micrófonos en el dominio de la frecuencia, superpuestas en un único gráfico. Cada señal está representada una señal de sonido oscilatoria que permite observar las diferencias en la amplitud y el comportamiento temporal de las capturas realizadas por cada micrófono."</em></p>

Las diferencias en la amplitud de las señales son indicativas de variaciones en la calidad de la captura, probablemente debidas a la presencia de ruido. Visualmente, se puede observar que la señal del micrófono 3 tiene una mayor amplitud y menos fluctuaciones, lo que coincide con el mayor valor de SNR reportado (49.77 dB), indicando que capturó una señal más limpia. En cambio, las señales de los micrófonos 1 y 2 presentan más irregularidades, lo que refleja la presencia de más ruido en sus capturas, especialmente en el caso del micrófono 2, que tiene el valor de SNR más bajo (26.54 dB).

Estos resultados muestran una variación considerable en la SNR de cada micrófono, lo que sugiere diferencias en la calidad de captura. El micrófono 3 ofrece la mayor SNR (49.77 dB), lo que indica que la señal capturada por este dispositivo contiene menos ruido en comparación con los otros. En contraste, el micrófono 2 tiene una SNR más baja (26.54 dB), lo que sugiere una mayor presencia de ruido. Estas diferencias pueden estar asociadas a factores como la ubicación de los micrófonos o la sensibilidad de cada dispositivo.

### 2. **Calidad de las Señales Separadas**

Se aplicó la técnica FastICA para separar las señales mezcladas provenientes de los micrófonos. Los valores de SNR de las señales separadas fueron los siguientes:

- Señal Separada 1: 76.25 dB
- Señal Separada 2: 76.25 dB
- Señal Separada 3: 76.25 dB

<p align="center">
   <img src="https://github.com/user-attachments/assets/0cf28a9a-8180-47a3-88cc-35e125cd2555" alt="Descripción de la imagen" width="400"> 
</p>
<p><em>"La imagen muestra las gráficas de las tres señales de voz separadas después de aplicar la técnica FastICA. Cada gráfica representa una señal de sonido oscilatoria, capturada en el dominio temporal."</em></p>

Se observa que las señales separadas son más coherentes y limpias en comparación con las señales mezcladas. Aunque se ha logrado una notable reducción de ruido, algunas oscilaciones irregulares aún son visibles, lo que indica que la separación no fue completamente perfecta y persisten componentes de otras fuentes sonoras. Estas irregularidades pueden deberse a la superposición residual de otras voces o ruidos. 

El aumento significativo en la SNR de las señales separadas sugiere una mejora notable en la calidad de las señales obtenidas tras la aplicación de FastICA. La técnica logró reducir el ruido presente en las señales capturadas inicialmente, obteniendo señales separadas con una SNR mucho mayor. Sin embargo, aunque los valores de SNR indican una buena separación en términos de reducción de ruido, es importante considerar otros factores, como la correlación con la señal de referencia, para determinar la fidelidad de la separación.
### 4. **Audios de las señales separadas**
Por medio del metodo de separacion de fuentes ICA se obtuvo el siguiente resultado

https://github.com/user-attachments/assets/558bcb84-2fbc-4c0b-bf1d-ea2e5286f976

Como se puede percibir en el audio a pesar de eliminar la voz de la tercera persona, aun se alcanza a identificar la voz de dos personas en el audio, por lo cual se recurrio al filtrado de esta señal por medio de un filtro pasa banda 

[https://github.com/user-attachments/assets/3ceb9d88-5861-462b-8fce-bc2600ff7b65](https://github.com/user-attachments/assets/69891a74-0337-4938-97d9-57983da6a231)

En este audio se percibe con mayor claridad la voz de la participante número 1 en comparación con la del participante número 2, sin embargo aun se alcanza a escuchar esta voz.

### 4. **Análisis Espectral y Temporal**

Los gráficos temporales y espectrales de las señales capturadas y separadas mostraron diferencias notables. El análisis espectral demostró que las señales separadas presentan una mayor coherencia en el rango de frecuencias esperado para las señales deseadas, lo que sugiere una mejora en la calidad general tras la separación.

![image](https://github.com/user-attachments/assets/a8d5ab78-9ffd-4241-8810-9eb65675e0ac)
![image](https://github.com/user-attachments/assets/efc7710f-e26f-4a4d-920f-6ce6b0d5738e)
![image](https://github.com/user-attachments/assets/7dec6c79-0c2a-4a4b-822f-6fb6758d6567)

<p><em>""La imagen presenta el análisis temporal y espectral de las señales capturadas por cada uno de los tres micrófonos."</em></p>

En la parte izquierda de la imagen, se muestran las gráficas del análisis temporal, donde las señales se representan en función del tiempo. Estas gráficas permiten observar la amplitud y variaciones de las señales capturadas, destacando diferencias notables entre los micrófonos. Se observa que el micrófono 3 presenta una señal más estable y con menos fluctuaciones, lo que sugiere una mejor calidad en la captura, en concordancia con su SNR más alto (49.77 dB).

En la parte derecha de la imagen, se muestra el análisis espectral de las mismas señales, donde se representa la amplitud en función de las frecuencias. Estas gráficas espectrales revelan la distribución de la energía de las señales a través de las distintas frecuencias, destacando las componentes dominantes en cada señal. El micrófono 3, nuevamente, muestra una distribución de frecuencias más limpia, mientras que los micrófonos 1 y 2 presentan una mayor dispersión de energía en frecuencias no deseadas, lo que indica la presencia de más ruido.

![image](https://github.com/user-attachments/assets/4dc1a9fa-0f83-4e54-bc07-fe268ca9c99e) <p><em>"Imagen de los espectrogramas de las señales separadas."</em></p>

<p align="center">
   <img src="https://github.com/user-attachments/assets/f2c059fd-1a98-43a4-bdff-118a8b529e5e" alt="Descripción de la imagen" width="400"> 
</p>
<p><em>"Imagen del espectrograma de la voz de referencia."</em></p>

Al observar los espectrogramas de las señales separadas y la señal de referencia. Si bien las señales separadas mostraron una reducción de ruido, todavía se observa cierta contaminación de otras señales no deseadas en las frecuencias más bajas, lo que indica que la separación no fue completamente efectiva en aislar la voz de interés.

La comparación visual de los espectrogramas mostró que, aunque la señal de voz principal se encuentra presente en una de las señales separadas, aún se observan componentes de las otras voces en las señales separadas.

El espectrograma de la voz de referencia muestra una mayor concentración de energía en ciertas bandas de frecuencia, mientras que las señales separadas presentan una dispersión más amplia de energía, indicando que la separación no fue completamente exitosa. Este fenómeno sugiere que las características del ruido y el solapamiento de las voces dificultaron el proceso de separación.

### 5. **Correlación con la Voz de Referencia**

Para evaluar cuantitativamente la similitud entre la voz de referencia y una de las señales separadas, se calculó el coeficiente de correlación de Pearson, obteniendo un valor de -0.0020. Este resultado está lejos de los valores esperados cercanos a 1 (o -1 en caso de una relación inversa), lo que indica que la separación no fue efectiva para aislar la voz de referencia y  que todavía contiene componentes de otras fuentes.
### 6. **Discusión de los Resultados**

Los resultados muestran que, aunque se logró una mejora en la SNR de las señales separadas, la separación completa de la señal deseada no fue lograda. Esto se refleja tanto en la baja correlación como en la presencia de contaminación de otras señales en el análisis espectral. La razón de esto podría estar relacionada con las condiciones de captura de las señales, tales como la disposición de los micrófonos, la calidad de los mismos, o incluso la naturaleza de las señales originales.

La comparación cualitativa entre los espectrogramas de la señal de referencia y las señales separadas muestra que, aunque se logró una mejora significativa en la calidad de las señales separadas, no se consiguió una separación perfecta. Es posible que ajustes adicionales en el procesamiento de señales o mejoras en la configuración del sistema (mejores micrófonos o una disposición más óptima) puedan llevar a mejores resultados en futuras iteraciones del experimento.

**Conclusión:** La técnica FastICA permitió una mejora en la SNR de las señales, pero no fue suficiente para obtener una separación completa de la señal deseada.


# Discusión
En el siguiente apartado se evidenciará la discusión de los resultados, esto por medio de la interpretación de las gráficas evidenciadas en el índice anterior.

Con respecto a cada uno de los SNR, pudimos evidenciar que la adquisición de la señal de ellos fue la adecuada, esto debido al uso de una habitación insonorizada del exterior, lo que al ser valores positivos mayores a diez esta fue una toma idónea como se evidencia en cada uno de los casos, lo que repercute en poca interferencia para tomar las muestras. Aunque el aspecto de la calidad de pistas de audio, pudo variar por el desfase de las señales correspondientes al factor humano, debido a que las señales tuvieron que ser relativamente similares. Sin embargo, hay una de ellas, que cuenta con periodos de tiempo que afectaron el resultado final.

Otro valor a considerar que el filtrado logro eliminar la voz de la 3ra persona, por lo que el filtro realizado por ICA, se muestra como una herramienta que puede eliminar frecuencias bajas como en el caso de la persona 3. Sin embargo, la falta de una separación completa de las voces podría estar relacionada con las características de los micrófonos o con las propias características de las voces. Esto se evidencia en el hecho de que, tras el filtrado, aún se podían escuchar algunas de las voces, aunque una más que la otra. Esto indica que las voces de la persona 2 y la persona 1 probablemente tienen espectros de frecuencia similares. 
# Conclusión

Los dos factores que afectaron los resultados en la separación de las fuentes fueron el uso de diferentes micrófonos y el factor humano al momento de capturar las señales.Por lo cual, se concluye que el uso de micrófonos del mismo tipo y la implementación de un software especializado en el procesamiento de audios provenientes de diferentes fuentes son mejoras que podrían reducir el error en este tipo de prácticas de laboratorio.

En la presente practica del laboratorio se pudo evidenciar como la posición relativa de los micrófonos puede afectar a la toma de datos, esto es debido a que la intensidad captada por el receptor en este caso el micrófono, al estar más cerca de la fuente de sonido, captara con mayor intensidad las vibraciones generadas en el aire, logrando opacar las que no estén en la misma distancia o condición, esto se puede evidenciar cuando se opaca la voz 3 en el micrófono 1, al ser la voz más alejada, esa su vez la que menos presencia tiene en la pista de audio.


# Referencias
Citas de libros, artículos o recursos en línea utilizados.
