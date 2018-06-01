# encoding: utf-8

#-------1------НАСТРОЙКА
# настройка Python
import numpy as np #numpy для работы с многомерными массивами и матрицами
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# установка параметров отображения по умолчанию
plt.rcParams['figure.figsize'] = (10, 10)        # большие изображения
plt.rcParams['image.interpolation'] = 'nearest'  # отключить интерполяцию
plt.rcParams['image.cmap'] = 'gray'  # выводить изображение в оттенках серого, вместо цветного

#Добавление системных модулей
import sys
import os
import time

import caffe # Добавление модуля caffe (в данном случае путь к caffe уже указан в sys.path)

# chdir to this script's directory

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

mout = open('mout.txt', 'a')

caffe_root = '/home/caffe/'# Задание корневой директории Caffe

# Проверка и загрузка тренированной модели CaffeNet
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system(caffe_root+'scripts/download_model_binary.py '+caffe_root+'models/bvlc_reference_caffenet')


#-------2------ЗАГРУЗКА СЕТИ
#--------------НАСТРОЙКА ПРЕОБРАЗОВАТЕЛЯ ВХОДНОГО ИЗОБРАЖЕНИЯ

#установка режима происходит ДО создания сети или решения
caffe.set_mode_cpu()
#caffe.set_device(gpu_id)
#caffe.set_mode_gpu()


model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,      # определение структуры модели сети
                model_weights,  # содержит веса тренированной модели
                caffe.TEST)     # использовать тестовом режиме (например, не выполнять маскирование)


# Загружаем средние значения ImageNet изображения для последующей нормализации
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # средние значения пикселей BGR
print 'Средние значения палитры BGR:', zip('BGR', mu)


# Создание преобразователя для входных данных 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # Изменение формы преставления изображения (C_H_W вместо H_W_C)
transformer.set_mean('data', mu)            # Вычитание набора средних значений из каждого канала для нормализации
transformer.set_raw_scale('data', 255)      # Установка масштаба исходных изображений [0, 255] вместо [0, 1]
transformer.set_channel_swap('data', (2,1,0))  # Перестановка цветовых каналов (RGB в BGR)


#-------3------КЛАССИФИКАЦИЯ ТЕСТОВОГО ИЗОБРАЖЕНИЯ
#--------------СРАВНЕНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ КЛАССИФИКАЦИИ В РЕЖИМАХ CPU И GPU

# Для демонстрации мы установим размер пакета изображений до 50
# что также подойдет для классификации одного изображения.

# Установка размера входных данных
# (при пропуске шага значения устанавливаются в соответствии с моделью сети)
# layer "data" input_param { shape: { dim: 10 dim: 3 dim: 227 dim: 227 } }
# Также мы можем изменить размер позже, например, для изменения размера пакета)
net.blobs['data'].reshape(50,        # размер пакета изображений
                          3,         # 3-канальные (BGR) изображения
                          227, 227)  # размер изображения 227x227

# Загрузка RGB изображения в форме H_W_C
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
# Преобразование изображения для формата Caffe 
transformed_image = transformer.preprocess('data', image)
# Сохранить изображение
plt.imshow(image)
plt.savefig("image.png")

# Копирование изображения в память выделенную для сети 
net.blobs['data'].data[...] = transformed_image

# ВЫПОЛНЕНИЕ КЛАССИФИКАЦИИ
# Сравним время выполнения с помощью CPU и GPU
print '\nКлассификация [CPU]...'
t1 = time.time()
output = net.forward()
t2 = time.time()
print t2-t1,'sec'
mout.write('CPU classification: ' + str(t2-t1) + 'sec\n')

print '\nКлассификация [GPU]...'
caffe.set_device(0)
caffe.set_mode_gpu()
t3 = time.time()
net.forward()
t4 = time.time()
print t4-t3,'sec'
mout.write('GPU classification: ' + str(t4-t3) + 'sec\n')

print '\nCPU/GPU time = ',(t2-t1)/(t4-t3)
mout.write('CPU/GPU time = ' + str((t2-t1)/(t4-t3)))

mout.close()


# Получение из выходного слоя 'prob' вектора вероятностей совпадения классов для загруженного ранее изображения.
output_prob = output['prob'][0]
# Индекс output_prob - номер класса изображения.
# Значение output_prob - значение вероятности номера класса.

# Загрузка файла с описанием классов
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt' # номер строки - номера класса (нумерация с 0)
labels = np.loadtxt(labels_file, str, delimiter='\t')


# Вывод номера класса с наибольшим значением вероятности
print 'Класс изображения №',output_prob.argmax(),' Описание:', labels[output_prob.argmax()]

# Вывод 5 наибольших вероятностей совпадений
# Сортировка вероятностей выходного вектора
top_inds = output_prob.argsort()[::-1][:5] # Получаем первые 5 наибольших совпадений

print 'ТОП-5 совпадений'
for prob,label in zip(output_prob[top_inds], labels[top_inds]):
	print '-вероятность: ',prob,' класс: ',label


#-------4------ИЗУЧЕНИЕ ПРОМЕЖУТОЧНЫХ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ
#--------------ВЫВОД ФОРМЫ И ПАРАМЕТРОВ СЛОЕВ

# Форму каждого слоя можно посмотреть следующим образом.
# Обычно форма имеет вид(batch_size, channel_dim, height, width).
for layer_name, blob in net.blobs.iteritems():
	print layer_name + '\t' + str(blob.data.shape)

# Форму параметров можно посмотреть следующим образом.
# Обычно форма имеет вид:
# -для весов (output_channels, input_channels, filter_height, filter_width)
# -для смещений (output_channels,)
for layer_name, param in net.params.iteritems():
	print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)


#-------5------Вспомогательная функция визуализации фильтров слоев

def vis_square(data,str):
	"""Берем массив с формой (n, height, width) или (n, height, width, 3) и отображаем каждый фильтр (height, width) в сетке размером приблизительно sqrt(n) by sqrt(n)"""
	
	# Нормализация данных для отображения
	data = (data - data.min()) / (data.max() - data.min())
	
	# сделать количество фильтров квадратным
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = (((0, n ** 2 - data.shape[0]),
	(0, 1), (0, 1))                 # Добавление разделителя между фильтрами
	+ ((0, 0),) * (data.ndim - 3))
	data = np.pad(data, padding, mode='constant', constant_values=1)
	
	# Плитка фильтров на изображении
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	plt.imshow(data)	
	plt.axis('off')
	plt.savefig(str)


# Отобразим фильтры первого сверточного слоя 'conv1' 
# Параметры представляют собой список [веса, смещения]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1), "conv1.png")

# Изображение на выходе после прохода сверточного слоя 'conv1'
# Отобразим изображения после обработки первыми 36-мя фильтрами
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat,"conv1_image.png")

# Изображение на выходе после слоя объединения 'pool5'
feat = net.blobs['pool5'].data[0]
vis_square(feat,"pool5.png")

# Полносвязный слой 'fc6'
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig("fc6.png")

# Вывод распределения вероятностей выходного слоя 'prob'
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.savefig("prob.png")


#-------6------КЛАССИФИКАЦИЯ СОБСТВЕННОГО ИЗОБРАЖЕНИЯ

# Загрузка изображения
#my_image_url = '...'
my_image_url = 'http://murkote.com/wp-content/uploads/2015/06/Canadian_Sphynx1.jpg'
os.system('wget -O image.jpg ' + my_image_url)

# преобразование и загрузка изображения в сеть
image = caffe.io.load_image('image.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# выполнение классификации
net.forward()

# получение выходных вероятностей
output_prob = net.blobs['prob'].data[0]

# Отобразим фильтры первого сверточного слоя 'conv1' 
# Параметры представляют собой список [веса, смещения]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1), "conv1_custom.png")

# Изображение на выходе после прохода сверточного слоя 'conv1'
# Отобразим изображения после обработки первыми 36-мя фильтрами
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat,"conv1_image_custom.png")

# Изображение на выходе после слоя объединения 'pool5'
feat = net.blobs['pool5'].data[0]
vis_square(feat,"pool5_custom.png")

# Полносвязный слой 'fc6'
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig("fc6_custom.png")

# Вывод распределения вероятностей выходного слоя 'prob'
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.savefig("prob_custom.png")

# Вывод 5 наибольших вероятностей совпадений
# Сортировка вероятностей выходного вектора
top_inds = output_prob.argsort()[::-1][:5] # Получаем первые 5 наибольших совпадений

print 'ТОП-5 совпадений'
for prob,label in zip(output_prob[top_inds], labels[top_inds]):
	print '-вероятность: ',prob,' класс: ',label

plt.imshow(image)
plt.savefig("custom_image.png")

raw_input("Выход")
                                  
