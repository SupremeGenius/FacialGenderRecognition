#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include "opencv2/core/core.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include<climits>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <dirent.h>


using namespace cv;
using namespace std;



Mat binary(Mat img)
{
	Mat dst = img.clone();

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (dst.at<uchar>(i, j) > 127)
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	return dst;
}


int classifyBayes(Mat img, Mat priors, Mat likelihood)
{
	vector<double> probs;
	for (int c = 0; c < 2; c++)
	{
		double p = 0;
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				double val = 0;
				if (img.at<uchar>(i, j) == 255)
				{
					val = likelihood.at<double>(c, 162 * i + j);
				}
				else
				{
					val = 1 - likelihood.at<double>(c, 162 * i + j);
				}


				p += log(val);
			}
		}

		probs.push_back(log(priors.at<double>(c)) + p);
	}

	int maxIndex = 0;
	int maxValue = INT_MIN;
	for (int i = 0; i < probs.size(); i++)
	{
		if (probs.at(i) > maxValue)
		{
			maxValue = probs.at(i);
			maxIndex = i;
		}
	}

	return maxIndex;
}





int main(int argc, char** argv) {

	//	Mat img = imread("lab10Data/prs_res_Perceptron/test01.bmp");
	//	imshow("t", img);
	//	waitKey(0);
	//onlineP();
	//knn();

	char* classes[2] = {"man", "woman"};

	Mat X = Mat(60000, 162*193, CV_8UC1);
	Mat Y = Mat(60000, 1, CV_8UC1);

	const int C = 2; //number of classes
	Mat priors(C, 1, CV_64FC1);

	const int d = 162*193;
	Mat likelihood(C, d, CV_64FC1,Scalar(0));

	char fname[256];
	double inst[10] = { 0.0 };

	int n = 0;

	for (int c = 0; c < C; c++)
	{
		int index = 1;
		while (true)
		{
			sprintf(fname, "dataset/train/%s/%04d.jpg", classes[c], index++);
			Mat img = imread(fname, IMREAD_GRAYSCALE);
			if (img.cols == 0) break;
			//process img
			Mat src = binary(img);
			for (int i = 0; i < src.rows; i++)
			{
				for (int j = 0; j < src.cols; j++)
				{
					X.at<uchar>(n, 162 * i + j) = src.at<uchar>(i, j);
				}
			}

			Y.at<uchar>(n++) = c;
			inst[c]++;
		}
	}

	priors.setTo(0);
	for (int i = 0; i < C; i++)
	{
		priors.at<double>(i,0) = inst[i] / n;
	}

	for (int j = 0; j < 162*193; j++)
	{
		for (int i = 0; i < n; i++)
		{
			if (X.at<uchar>(i, j) == 255)
			{
				likelihood.at<double>(Y.at<uchar>(i), j)++;
			}
		}
	}



	for (int c = 0; c < C; c++)
	{
		for (int j = 0; j < 162*193; j++)
		{
			double val = likelihood.at<double>(c, j) / (priors.at<double>(c)*n);

			if (val < pow(10, -5))
			{
				val = pow(10, -5);
			}
			else if (val > 1 - pow(10, -5))
			{
				val = 1 - pow(10, -5);
			}

			likelihood.at<double>(c, j) = val;
		}
	}


	Mat conf = Mat(C, C, CV_32FC1,Scalar(0));


	int ok;
	for (int c = 0; c < C; c++)
	{
		ok = 0;
		int index = 1;
		while (true){
			sprintf(fname, "dataset/test/%s/%04d.jpg", classes[c], index);
			Mat img = imread(fname, IMREAD_GRAYSCALE);
			if (img.cols == 0) break;
			Mat src = binary(img);

			int pred = classifyBayes(src, priors, likelihood);

			conf.at<float>(pred, c)++;

			index++;
		}

	}

	std::cout << conf << endl;

	float suma = 0.0;
	for (int i = 0; i < C; i++)
		suma += conf.at<float>(i, i);

	float total = 0.0;
	for (int i = 0; i < C; i++)
		for (int j = 0; j < C; j++)
			total += conf.at<float>(i, j);

	printf("Acuratete %f\n", suma/total);

	int i = 1;
	while(true){
		sprintf(fname, "dataset/test/man/%04d.jpg", i++);
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		if(img.cols ==0) break;
		Mat src = binary(img);

		int pred = classifyBayes(src, priors, likelihood);

		printf("clasa: %d, imaginea: %04d\n", pred, i-1);
	}
	return 0;
}
