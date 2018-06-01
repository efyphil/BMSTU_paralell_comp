// NBodyProblem.cpp: определяет точку входа для консольного приложения.

//#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <fstream>

using namespace std;
#define gravity 10 // гравитационная постоянная
#define dt 0.1 // шаг по времени
#define N 800 // количество частиц
#define fmax 1000 // максимальное значение силы
#define Niter 1000 // число итераций


#define Nthr 16 // число потоков

// частица (тело)
struct Particle
{
	double x, y, vx, vy;
};

// сила
struct Force
{
	double x, y;
};


Particle p[N]; // массив n тел
Force f[N]; // массив n сил
double m[N]; // массив n масс

Force tf[N][Nthr]; // массивы n сил для каждого потока


ofstream fout("results.txt"); // файл для записи результатов

						   // инициализация начальных параметров системы
void Init()
{
	for (int i = 0; i < N; i++)
	{
		//----------------------------------------------------------------------
		// эти параметры можно менять
		p[i].x = rand() % 4000 + 10 + (1.0 / rand()); // координата х i-ого тела
		p[i].y = rand() % 4000 + 10 + (1.0 / rand()); // координата y i-ого тела
		p[i].vx = 0; // проекция на ось x верктора начальной скорость i-ого тела
		p[i].vy = 0; // проекция на ось y верктора начальной скорость i-ого тела
		m[i] = 10000; // масса i-ого тела
					 //----------------------------------------------------------------------

		f[i].x = 0; // в начальный момент времени все силы равны 0
		f[i].y = 0;
	}

	for (int i = 0; i < N; i++)
		for (int j = 0; j < Nthr; j++)
		{
			tf[i][j].x = 0;
			tf[i][j].y = 0;
		}

}

// Запись текущих координат в файл
void WriteCoordinates()
{
	for (int i = 0; i < N; i++)
	{
		fout << p[i].x << "_" << p[i].y << "|";
	}
	fout << endl;
}

// непараллельный расчет сил в момент времени t
void CalcForces1()
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			if (i == j) continue;
			double dx = p[j].x - p[i].x, dy = p[j].y - p[i].y,
				r_2 = 1 / (dx * dx + dy * dy),
				r_1 = sqrt(r_2),
				fabs = gravity * m[i] * m[j] * r_2;
			if (fabs > fmax) fabs = fmax;
			f[i].x = f[i].x + fabs * dx * r_1;
			f[i].y = f[i].y + fabs * dy * r_1;
		}
}

// непараллельный расчет сил в момент времени t с одновременным нахождением сил для пары тел
void CalcForces2()
{
	for (int i = 0; i < N - 1; i++)
		for (int j = i + 1; j < N; j++)
		{
			double dx = p[j].x - p[i].x, dy = p[j].y - p[i].y,
				r_2 = 1 / (dx * dx + dy * dy),
				r_1 = sqrt(r_2),
				fabs = gravity * m[i] * m[j] * r_2;
			if (fabs > fmax) fabs = fmax;
			f[i].x += dx = fabs * dx * r_1;
			f[i].y += dy = fabs * dy * r_1;
			f[j].x -= dx;
			f[j].y -= dy;
		}
}

// параллельный расчет сил в момент времени t
void CalcForces1Par()
{
#pragma omp parallel for num_threads(Nthr)
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			if (i == j) continue;
			double dx = p[j].x - p[i].x, dy = p[j].y - p[i].y,
				r_2 = 1 / (dx * dx + dy * dy),
				r_1 = sqrt(r_2),
				fabs = gravity * m[i] * m[j] * r_2;
			if (fabs > fmax) fabs = fmax;
			f[i].x += fabs * dx * r_1;
			f[i].y += fabs * dy * r_1;
		}
}


// параллельный расчет сил в момент времени t с использованием критической секции
void CalcForces2ParA()
{
#pragma omp parallel for num_threads(Nthr)
	for (int i = 0; i < N - 1; i++)
		for (int j = i + 1; j < N; j++)
		{
			double dx = p[j].x - p[i].x, dy = p[j].y - p[i].y,
				r_2 = 1 / (dx * dx + dy * dy),
				r_1 = sqrt(r_2),
				fabs = gravity * m[i] * m[j] * r_2;
			if (fabs > fmax) fabs = fmax;
#pragma omp critical
			{
				f[i].x += dx = fabs * dx * r_1;
				f[i].y += dy = fabs * dy * r_1;
				f[j].x -= dx;
				f[j].y -= dy;
			}
		}
}


// параллельный расчет сил в момент времени t с использованием дополнительных массивов для каждого потока
void CalcForces2ParB()
{
#pragma omp parallel for num_threads(Nthr)
	for (int i = 0; i < N - 1; i++)
	{
		int k = omp_get_thread_num();
		for (int j = i + 1; j < N; j++)
		{
			double dx = p[j].x - p[i].x, dy = p[j].y - p[i].y,
				r_2 = 1 / (dx * dx + dy * dy),
				r_1 = sqrt(r_2),
				fabs = gravity * m[i] * m[j] * r_2;
			if (fabs > fmax) fabs = fmax;
			tf[i][k].x += dx = fabs * dx * r_1;
			tf[i][k].y += dy = fabs * dy * r_1;
			tf[j][k].x -= dx;
			tf[j][k].y -= dy;
		}
	}
#pragma omp parallel for num_threads(Nthr)
	for (int i = 0; i < N; i++)
		for (int j = 0; j < Nthr; j++)
		{
			f[i].x += tf[i][j].x;
			f[i].y += tf[i][j].y;
			tf[i][j].x = 0;
			tf[i][j].y = 0;
		}
}


// параллельный расчет сил в момент времени t с использованием динамической балансировки нагрузки между потоками
int block = 25;
void CalcForces2ParC()
{
#pragma omp parallel for num_threads(Nthr) schedule(dynamic, block) 
	for (int i = 0; i < N - 1; i++)
	{
		int k = omp_get_thread_num();
		for (int j = i + 1; j < N; j++)
		{
			double dx = p[j].x - p[i].x, dy = p[j].y - p[i].y,
				r_2 = 1 / (dx * dx + dy * dy),
				r_1 = sqrt(r_2),
				fabs = gravity * m[i] * m[j] * r_2;
			if (fabs > fmax) fabs = fmax;
			tf[i][k].x += dx = fabs * dx * r_1;
			tf[i][k].y += dy = fabs * dy * r_1;
			tf[j][k].x -= dx;
			tf[j][k].y -= dy;
		}
	}
#pragma omp parallel for num_threads(Nthr) schedule(dynamic, block) 
	for (int i = 0; i < N; i++)
		for (int j = 0; j < Nthr; j++)
		{
			f[i].x += tf[i][j].x;
			f[i].y += tf[i][j].y;
			tf[i][j].x = 0;
			tf[i][j].y = 0;
		}
}


// параллельный расчет сил в момент времени t с ручным распределением итераций между потоками 
void CalcForces2ParD()
{
#pragma omp parallel num_threads(Nthr)
	{
		int k = omp_get_thread_num();
		for (int i = k; i < N - 1; i += Nthr)
		{
			for (int j = i + 1; j < N; j++)
			{
				double dx = p[j].x - p[i].x, dy = p[j].y - p[i].y,
					r_2 = 1 / (dx * dx + dy * dy),
					r_1 = sqrt(r_2),
					fabs = gravity * m[i] * m[j] * r_2;
				if (fabs > fmax) fabs = fmax;
				tf[i][k].x += dx = fabs * dx * r_1;
				tf[i][k].y += dy = fabs * dy * r_1;
				tf[j][k].x -= dx;
				tf[j][k].y -= dy;
			}
		}
	}
#pragma omp parallel for num_threads(Nthr)
	for (int i = 0; i < N; i++)
		for (int j = 0; j < Nthr; j++)
		{
			f[i].x += tf[i][j].x;
			f[i].y += tf[i][j].y;
			tf[i][j].x = 0;
			tf[i][j].y = 0;
		}
}



// непараллельный пересчет координат и обнуленеие массива сил
void MoveParticlesAndFreeForces()
{
	for (int i = 0; i < N; i++)
	{
		double dvx = f[i].x * dt / m[i],
			dvy = f[i].y * dt / m[i];
		p[i].x += (p[i].vx + dvx / 2) * dt;
		p[i].y += (p[i].vy + dvy / 2) * dt;
		p[i].vx += dvx;
		p[i].vy += dvy;
		f[i].x = 0;
		f[i].y = 0;
	}
}

// параллельный пересчет координат и обнуление массива сил
void MoveParticlesAndFreeForcesPar()
{
#pragma omp parallel for num_threads(Nthr)
	for (int i = 0; i < N; i++)
	{
		double dvx = f[i].x * dt / m[i],
			dvy = f[i].y * dt / m[i];
		p[i].x += (p[i].vx + dvx / 2) * dt;
		p[i].y += (p[i].vy + dvy / 2) * dt;
		p[i].vx += dvx;
		p[i].vy += dvy;
		f[i].x = 0;
		f[i].y = 0;
	}
}


int main()
{
	Init();
	double time = omp_get_wtime();
	for (int i = 0; i < Niter; i++)
	{
		//--------------------------------
		//CalcForces1();
		//CalcForces2();
		//CalcForces1Par();
		//CalcForces2ParA();
		//CalcForces2ParB();
		//CalcForces2ParC();
		CalcForces2ParD();

		//MoveParticlesAndFreeForces();
		MoveParticlesAndFreeForcesPar();
		//--------------------------------

		WriteCoordinates();
	}
	time = omp_get_wtime() - time;
	cout << "Time: " << 1000 * time << "ms" << endl << "Press any key to exit...";
	fout.close();
	getchar();

	return 0;
}

