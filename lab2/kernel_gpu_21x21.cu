
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>


using std::cin;
using std::cout;


/*******************************************************************************************************************************************************
*
* DEVICE
*
*******************************************************************************************************************************************************/


namespace Device
{
	// Количество потоков на измерение, которое мы будем использовать.
	const int nThreads = 32;

	// Исходное изображение.
	double *inImage = nullptr;
	// Выходное изображение.
	double *outImage = nullptr;
	// Веса фильтров (размерности фильтров должны быть одинаковыми.
	double *filters = nullptr;
	// Результат работы фильтров
	double *featureMaps = nullptr;

	// Вспомогательная структура со всеми размерами (в штуках, а не в байтах).
	struct Size
	{
		std::size_t inImageSize;
		std::size_t outImageWidth;
		std::size_t outImageHeight;
		std::size_t outImageSize;
		std::size_t kernelWeightsSize;
		std::size_t featureMapsize;
	} size;

	// Освобождает память.
	static void freeMemory()
	{
		cudaFree(inImage);
		cudaFree(outImage);
		cudaFree(filters);
		cudaFree(featureMaps);
	}

	/*
	* @param width - ширина изображения.
	* @param height - высота изображения.
	* @param stride - сдвиг окна фильтра.
	* @param filterLineSize - размер строки в матрице одного фильтра. Должен быть нечетным.
	* @param filtersCount - количество фильтров.
	* @note Подразумевается, что размеры всех фильтров одинаковы.
	*/
	static int allocateMemory(std::size_t width, std::size_t height, std::size_t stride, std::size_t filterLineSize, std::size_t filtersCount)
	{
		// Для контроля ошибок выделения памяти.
		auto cudaError = cudaSuccess;

		// Выделяем память.
		// Память выделяется в БАЙТАХ, поэтому даже для int или float нужно домножать на sizeof (то есть на количество байт, которое занимает переменная типа).

		auto kernelWeightsSize = filtersCount * filterLineSize * filterLineSize;
		cudaError = cudaMalloc(&filters, kernelWeightsSize * sizeof(*filters));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		auto inImageSize = width * height * 3;
		cudaError = cudaMalloc(&inImage, inImageSize * sizeof(*inImage));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		auto outImageWidth  = ((width - filterLineSize) / stride + 1);
		auto outImageHeight = ((height - filterLineSize) / stride + 1);
		auto outImageSize   = outImageWidth * outImageHeight * 3;

		cudaError = cudaMalloc(&outImage, outImageSize * sizeof(*outImage));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		auto featureMapsize = filtersCount * outImageSize;
		cudaError = cudaMalloc(&featureMaps, featureMapsize * sizeof(*featureMaps));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		// Заполняем выделенную память нулями.

		cudaError = cudaMemset(filters, 0, kernelWeightsSize * sizeof(*filters));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		cudaError = cudaMemset(inImage, 0, inImageSize * sizeof(*inImage));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		cudaError = cudaMemset(outImage, 0, outImageSize * sizeof(*outImage));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		cudaError = cudaMemset(featureMaps, 0, featureMapsize * sizeof(*featureMaps));

		if (cudaSuccess != cudaError) {
			freeMemory();
			return -1;
		}

		// Заполняем структуру с размерами элементов, объявленную выше.

		size.kernelWeightsSize = kernelWeightsSize;
		size.inImageSize       = inImageSize;
		size.outImageWidth     = outImageWidth;
		size.outImageHeight    = outImageHeight;
		size.outImageSize      = outImageSize;
		size.featureMapsize    = featureMapsize;

		return 0;
	}
}
/* Функция ядра, являющаяся имитацией сверточного слоя нейросети.
 * 
 * Эта функция работает следующим образом
 * 1. на вход подается изображение inImage. Это изображение имеет структуру трехмерного массива [[[r, g, b], [r, g, b], ..., ], [[r, g, b], [r, g, b], ..., ], ... ]
 * здесь [r, g, b] - это пиксель (так как он состоит из трех цветов)
 * [[r, g, b], [r, g, b], ..., ] - это строчка пикселов
 * [[[r, g, b], [r, g, b], ..., ], [[r, g, b], [r, g, b], ..., ], ... ] - контейнер таких строчек.
 *
 * У изображения есть ширина и высота. Массив имеет размерность [высота [ширина [3]]]. Об этом важно помнить!!!
 *
 * Если развернуть это изображение в одномерный массив, то его размерность будет высота Х ширина Х 3, тогда
 * [[[r, g, b], [r, g, b], ..., ], [[r, g, b], [r, g, b], ..., ], ... ] превратится в r, g, b, r, g, b, r, g, b, ...
 *
 * ЗАМЕЧАНИЕ
 * Мы будем говорить о ПИКСЕЛЕ (или ТОЧКЕ) и о ЦВЕТЕ. Пиксель - это [r, g, b] (то есть иногда для удобства мы будем представлять, как будто мы работаем не с трехмерным массивом, а с двухмерным).
 * Цвет - это конкретное значение (от 0 до 255) конкретного цвета конкретного пикселя. Когда мы будем говорить о цвете, никакой абстракции уже не останется.
 *
 * 2. на вход подаются фильтры. Это массив вида [ [[1, 1, 1,], [1, 1, 1], [1, 1, 1]], ... ]. В данном случае передается массив матриц-фильтров 3Х3
 *
 * ЗАМЕЧАНИЯ
 * фильтры обязательно должны быть квадратными
 * размерность обязательно нечетная
 * все фильтры имеют одинаковую размерность
 * 
 * Фильтры - это матрицы, которые позволяют находить на изображении специальные особенности, подобно тому, как работает зрительная кора головного мозга. 
 * Например, это могут быть прямые линии, линии под наклоном, а могут быть и сложные паттерны типа человеческого лица.
 * Каждый фильтр после применения к изображению формирует новое изображение меньшей размерности. На этом изображении (в зависимости от того, насколько правильно подобраны веса)
 * четко отображаются признаки, описанные выше. Все эти изображения формируют карту призаков или featureMaps. Эти изображения мы будем сохранять в одноименную переменную. 
 * Таким образом, featureMaps будет по сути массивом изображений в виде [[[[r, g, b], [r, g, b], ..., ], [[r, g, b], [r, g, b], ..., ], ... ], ...]
 * Сколько фильтов, столько и изображений, причем идут они соответственно номерам, то есть первый фильтр формирует первое изображение в массиве, второй - второе и т.д.
 *
 * В тот же самый момент, мы возьмем все карты признаков, попиксельно сложим, разделим на количество фильтров, и результат сохраним в outImage. По способу хренения он идентичен inImage (однако меньшего размера).
 * 
 *
 * @param inImage      - входное изображение для обработки.
 * @param width        - ширина изображения.
 * @param height       - высота изображения.
 * @param filters      - веса фильтров.
 * @param filtersCount - количество фильтров.
 * @param stride       - смещение фильтра на следующем шаге.
 * @param outImage     - выходное изображение.
 * @param featureMaps  - карты признаков.
 * @param outWidth     - ширина выходного изображения.
 * @param outHeight    - высота выходного изображения.
 */

__global__ void gpuCNN(
	const double *inImage,
	std::size_t width,
	std::size_t height,
	double *filters,
	std::size_t filtersCount,
	std::size_t filterLineSize,
	std::size_t stride,
	double *outImage, 
	double *featureMaps,
	std::size_t outWidth,
	std::size_t outHeight)
{
	auto halfLineSize = filterLineSize / 2;

	stride = (0 == stride) ? 1 : stride;

	auto outPixelX = threadIdx.x + blockIdx.x * blockDim.x;
	auto outPixelY = threadIdx.y + blockIdx.y * blockDim.y;

	if (outPixelX < outWidth && outPixelY < outHeight) 
	{
		auto pixelX = outPixelX * stride + halfLineSize;
		auto pixelY = outPixelY * stride + halfLineSize;

		// Функция ядра на GPU
	for(std::size_t colorIdx = 0; colorIdx < 3; ++colorIdx)
	{
    		double outImageSum = 0;
     		auto outColorPos = outPixelY*outWidth*3 + outPixelX*3 + colorIdx;
    		for(std::size_t filterIdx = 0; filterIdx < filtersCount; ++filterIdx)
    		{
        		double currentFilterSum = 0;

        		for(std::size_t i = 0; i < filterLineSize; ++i) 
        		{
            			for(std::size_t j = 0; j < filterLineSize; ++j)
            			{
                 			auto convPixelX = pixelX + j - halfLineSize;
                 			auto convPixelY = pixelY + i - halfLineSize;
                 			auto colorPos = convPixelY*width*3 + convPixelX*3 + colorIdx;
                 			auto weightPos = filterIdx*filterLineSize*filterLineSize + i*filterLineSize + j;
                 			currentFilterSum += inImage[colorPos] * filters[weightPos];
           			}
       			}
       			
			outImageSum += currentFilterSum;

       			featureMaps[filterIdx*outWidth*outHeight*3 + outColorPos] = currentFilterSum;
    		}

		outImage[outColorPos] = outImageSum / (float)filtersCount;
	}

		
	}
}


/*******************************************************************************************************************************************************
*
* DEVICE
*
*******************************************************************************************************************************************************/


void cpuCNN(
	const std::vector<double> &inImage,
	std::size_t width,
	std::size_t height,
	const std::vector<double> &filters,
	std::size_t filtersCount,
	std::size_t filterLineSize,
	std::size_t stride,
	std::vector<double> &outImage,
	std::vector<double> &featureMaps,
	std::size_t outWidth,
	std::size_t outHeight
) 
{
	/*
	static const auto halfLineSize = filterLineSize / 2;
	stride = (0 == stride) ? 1 : stride;

        auto outPixelX = threadIdx.x + blockIdx.x * blockDim.x;
        auto outPixelY = threadIdx.y + blockIdx.y * blockDim.y;

        if (outPixelX < outWidth && outPixelY < outHeight)
	{
                auto pixelX = outPixelX * stride + halfLineSize;                        auto pixelY = outPixelY * stride + halfLineSize;       
	
		
	}

 Аналогичная функция на CPU.
	*/	
}


/*******************************************************************************************************************************************************
*
* MAIN
*
*******************************************************************************************************************************************************/

namespace CliArgs
{
	// Количество аргументов, которое ожидает программа (имя программы среди них с индексом 0, поэтому количество на 1 больше).
	static const int N_ARGS      = 5;
	// Путь к входному изображению.
	static const int IN_FILE_POS = 1;
	// Размерность файла - ширина.
	static const int IMG_WIDTH   = 2;
	// Размерность файла - высота.
	static const int IMG_HEIGHT  = 3;
	// Страйд
	static const int STRIDE      = 4;
}

/*******************************************************************************************************************************************************
*
* MAIN
*
*******************************************************************************************************************************************************/


// ВАЖНО
// Дабы окончательно не усложнять программу, проверки вводимых данных делаются по минимуму.


int main(int argc, char *argv[])
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	setlocale(0, "russian");
#endif

	// Проверяем количество аргументов.
	if (CliArgs::N_ARGS != argc) {
		cout << "Неверное количество аргументов." << std::endl;
		getchar();
		return 1;
	}

	// Извлекаем имена файлов.
	auto imageFilePath = argv[CliArgs::IN_FILE_POS];

	// Извлекаем размерность картинки.
	auto imageWidth  = atoi(argv[CliArgs::IMG_WIDTH]);
	auto imageHeight = atoi(argv[CliArgs::IMG_HEIGHT]);

	// Извлекаем страйд.
	auto stride = atoi(argv[CliArgs::STRIDE]);

	auto imageSize = imageWidth * imageHeight * 3;

	// Читаем данные из файла с изображением.

	std::ifstream ifs(imageFilePath, std::ios_base::in);

	if (!ifs.is_open()) {
		cout << "Невозможно открыть файл " << imageFilePath << std::endl;
		getchar();
		return 1;
	}

	std::cout << "Начато чтение из файла..." << std::endl;

	std::vector<double> imageData(imageSize);

	for (std::size_t i = 0; i < imageSize; ++i)
		ifs >> imageData[i];

	ifs.close();

	std::cout << "Чтение закончено" << std::endl;

	// Заполняем фильтры.

	auto filterLineSize = 21;
	auto filtersCount   = 4;

	std::vector<double> filters = {
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,		

		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		
		2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,

		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
		-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5
	};
	
	// Выделяем память на устройстве.

	if (0 != Device::allocateMemory(imageWidth, imageHeight, stride, filterLineSize, filtersCount)) {
		cout << "Ошибка выделения памяти на графической карте" << std::endl;
		getchar();
		return 1;
	}

	cout << "Закончено выделение памяти на устройстве" << std::endl;

	// Копируем данные HOST -> GPU.

	auto cudaError = cudaSuccess;

	cudaError = cudaMemcpy(Device::inImage, imageData.data(), imageSize * sizeof(imageData[0]), cudaMemcpyHostToDevice);

	if (cudaSuccess != cudaError) {
		cout << "Ошибка при копировании результата на устройство: " << cudaError << std::endl;
		getchar();
		Device::freeMemory();
		return 1;
	}

	cudaError = cudaMemcpy(Device::filters, filters.data(), filters.size() * sizeof(filters[0]), cudaMemcpyHostToDevice);

	if (cudaSuccess != cudaError) {
		cout << "Ошибка при копировании результата на устройство: " << cudaError << std::endl;
		getchar();
		Device::freeMemory();
		return 1;
	}

	cout << "Закончено копирование данных на устройство" << std::endl;

	// Расчет на CPU.
/*
	std::vector<double> cpuOutImage(Device::size.outImageSize, 0.0);
	std::vector<double> cpuFeatureMaps(Device::size.featureMapsize, 0.0);

	auto cpuBeginTime = std::chrono::steady_clock::now();
	cpuCNN(
		imageData,
		imageWidth,
		imageHeight,
		filters,
		filtersCount,
		filterLineSize,
		stride,
		cpuOutImage,
		cpuFeatureMaps,
		Device::size.outImageWidth,
		Device::size.outImageHeight
	);
	auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cpuBeginTime).count();

	std::ofstream cpuImgOfs("cpu_out_image.txt", std::ios_base::out | std::ios_base::trunc);

	if (cpuImgOfs.is_open())
		for (std::size_t i = 0; i < Device::size.outImageSize; ++i)
			cpuImgOfs << static_cast<unsigned int>(cpuOutImage[i]) % 255 << " ";

	cpuImgOfs.close();

	cout << "Запись изображения в файл закончена..." << std::endl;

	cout << "Начата запись карты признаков в файл..." << std::endl;

	std::ofstream cpuFmOfs("cpu_out_features.txt", std::ios_base::out | std::ios_base::trunc);

	if (cpuFmOfs.is_open())
		for (std::size_t i = 0; i < Device::size.featureMapsize; ++i)
			cpuFmOfs << static_cast<unsigned int>(cpuFeatureMaps[i]) % 255 << " ";

	cpuFmOfs.close();

	cout << "Запись карты признаков в файл закончена..." << std::endl;
*/
	// Расчет на GPU.
	
	dim3 threads(Device::nThreads, Device::nThreads);

	auto nBlocksX = Device::size.outImageWidth / threads.x;
	nBlocksX     += (0 == Device::size.outImageWidth % threads.x) ? 0 : 1;
	auto nBlocksY = Device::size.outImageHeight / threads.y;
	nBlocksY     += (0 == Device::size.outImageHeight % threads.y) ? 0 : 1;

	dim3 blocks(nBlocksX, nBlocksY);

	// Запуск функции ядра.
	auto gpuBeginTime = std::chrono::steady_clock::now();
	gpuCNN <<< blocks, threads >>> (
		Device::inImage, 
		imageWidth, 
		imageHeight, 
		Device::filters, 
		filtersCount, 
		filterLineSize, 
		stride, 
		Device::outImage, 
		Device::featureMaps, 
		Device::size.outImageWidth, 
		Device::size.outImageHeight
	);
	cudaDeviceSynchronize();
	auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gpuBeginTime).count();

	cout << "Закончен расчет на GPU" << std::endl;

	// Теперь тащим с GPU результат.

	auto outImage    = new double[Device::size.outImageSize];
	auto featureMaps = new double[Device::size.featureMapsize];

	cudaError = cudaMemcpy(outImage, Device::outImage, Device::size.outImageSize * sizeof(*outImage), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaError) {
		cout << "Ошибка при копировании изображения с устройства: " << cudaError << std::endl;
		getchar();
		Device::freeMemory();
		return 1;
	}

	cudaError = cudaMemcpy(featureMaps, Device::featureMaps, Device::size.featureMapsize * sizeof(*featureMaps), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaError) {
		cout << "Ошибка при копировании карт признаков с устройства: " << cudaError << std::endl;
		getchar();
		Device::freeMemory();
		return 1;
	}

	Device::freeMemory();

	cout << "Копирование результата с устройства закончено" << std::endl;

	// Пишем в файлы.

	std::cout << "Начата запись изображения в файл..." << std::endl;

	std::ofstream imgOfs("out_image.txt", std::ios_base::out | std::ios_base::trunc);

	if (imgOfs.is_open())
		for (std::size_t i = 0; i < Device::size.outImageSize; ++i)
			imgOfs << static_cast<unsigned int>(outImage[i]) % 255 << " ";

	imgOfs.close();

	cout << "Запись изображения в файл закончена..." << std::endl;

	cout << "Начата запись карты признаков в файл..." << std::endl;

	std::ofstream fmOfs("out_features.txt", std::ios_base::out | std::ios_base::trunc);

	if (fmOfs.is_open())
		for (std::size_t i = 0; i < Device::size.featureMapsize; ++i)
			fmOfs << static_cast<unsigned int>(featureMaps[i]) % 255 << " ";

	fmOfs.close();

	cout << "Запись карты признаков в файл закончена..." << std::endl;

	delete[] outImage;
	delete[] featureMaps;

	cout << std::endl << std::endl;
	cout << "Полученное изображение имеет параметры " << Device::size.outImageWidth << " X " << Device::size.outImageHeight << std::endl;
	cout << "Карта признаков имеет параметры " << Device::size.outImageWidth << " X " << Device::size.outImageHeight * filtersCount << std::endl << std::endl;

//	cout << "Время на CPU: " << cpuTime << " миллисекунд "<< std::endl;
 cout << "Время на GPU: " << gpuTime << " миллисекунд" << std::endl;
	cout << "Для выхода нажмите Enter." << std::endl;

	getchar();

	return 0;
}

