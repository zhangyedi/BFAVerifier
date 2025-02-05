/*
 *  GPUPoly library
 *  This source file is part of ELINA (ETH LIbrary for Numerical Analysis).
 *  ELINA is Copyright � 2020 Department of Computer Science, ETH Zurich
 *  This software is distributed under GNU Lesser General Public License Version 3.0.
 *  For more information, see the ELINA project website at:
 *  http://elina.ethz.ch
 *
 *  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
 *  EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO ANY WARRANTY
 *  THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS OR BE ERROR-FREE AND ANY
 *  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
 *  TITLE, OR NON-INFRINGEMENT.  IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY
 *  DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT,
 *  SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN
 *  ANY WAY CONNECTED WITH THIS SOFTWARE (WHETHER OR NOT BASED UPON WARRANTY,
 *  CONTRACT, TORT OR OTHERWISE).
 */


 /*!
   \file src/bindings.cu
   \brief Implementation of the API functions.
   \author Fran&ccedil;ois Serre

	Implementation of the API functions defined in include/gpupoly.h. They work as a proxy for the NeuralNetwork class, where the actual algorithm is implemented.
  */


#include <iostream>
#include <vector>
#include "intv.h"
#include "network.h"
#include "layers/parsum.h"
#include "layers/concat.h"
#include "layers/relu.h"
#include "layers/sigmoid.h"
#include "layers/tanh.h"
#include "layers/dense.h"
#include "layers/conv2d.h"
#include "layers/maxpool2d.h"
#include "layers/bias.h"
#include "matrix.h"
#include "vector.h"
#include "gpupoly.h"
#include "config.h"
#include "dpolyr.h"


//! Static structure that contains the neural network that the API functions manipulate.
struct SplashScreen
{
	bool networkLoaded;
	SplashScreen():networkLoaded(false)
	{
		std::cout << "GPUPoly " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH;
#ifdef STRONG_FP_SOUNDNESS
		std::cout << "S";
#else
		std::cout << "W";
#endif
#ifndef NDEBUG
		std::cout << " Debug";
#endif
		std::cout << " (built " << __DATE__ << " " << __TIME__ << ") - Copyright (C) 2020 Department of Computer Science, ETH Zurich." << std::endl;
		std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it and to modify it under the terms of the GNU LGPLv3." << std::endl << std::endl;
	}
} ss;

void clean(NeuralNetwork* nn)
{
	ss.networkLoaded=false;
	delete nn;
}

NeuralNetwork* create(int inputSize)
{
	#ifndef MULTIPLE_NETWORKS
	if (ss.networkLoaded)
	{
		std::cerr << "Error: A network is already loaded in GPU memory. Please clean the previous network before creating a new one." << std::endl;
		std::cerr << "Note:  If you actually intend to have several networks on the GPU at once, please compile GPUPoly with the cmake flag MULTIPLE_NETWORKS." << std::endl;
		std::cerr << "       Otherwise, it probably means that you have forgotten a call to clean." << std::endl;
		return nullptr;
	}
	ss.networkLoaded=true;
	#endif
	return new NeuralNetwork(inputSize);
}


bool test_d(NeuralNetwork* nn, const double* dataDown, const double* dataUp, int expectedLabel, bool soundness)
{
	size_t size = (*nn)[0]->outputSize;
	std::vector<Intv<double>> candidate(size);
	for (size_t i = 0; i < size; i++)
		candidate[i] = Intv<double>(dataDown[i], dataUp[i]);
	return nn->run(Vector<double>(candidate), expectedLabel, soundness);
}

bool test_s(NeuralNetwork* nn, const float* dataDown, const float* dataUp, int expectedLabel, bool soundness)
{
	size_t size = (*nn)[0]->outputSize;
	std::vector<Intv<float>> candidate(size);
	for (size_t i = 0; i < size; i++)
		candidate[i] = Intv<float>(dataDown[i], dataUp[i]);
	return nn->run(Vector<float>(candidate), expectedLabel, soundness);
}


void setLayerBox_s(
	NeuralNetwork* nn,
	const float* dataDown,
	const float* dataUp,
	int layer
)
{
	size_t size = (*nn)[layer]->outputSize;
	std::vector<Intv<float>> candidate(size);
	for (size_t i = 0; i < size; i++)
		candidate[i] = Intv<float>(dataDown[i], dataUp[i]);
	nn->getConcreteBounds<float>(layer) = candidate;
}

void setLayerBox_d(
	NeuralNetwork* nn,
	const double* dataDown,
	const double* dataUp,
	int layer
)
{
	size_t size = (*nn)[layer]->outputSize;
	std::vector<Intv<double>> candidate(size);
	for (size_t i = 0; i < size; i++)
		candidate[i] = Intv<double>(dataDown[i], dataUp[i]);
	nn->getConcreteBounds<double>(layer) = candidate;
}


void relax_s(NeuralNetwork* nn, int layer, bool refineActivationsInput, bool soundness)
{
	(*nn)[layer]->eval(nn->getConcreteBounds<float>(layer), soundness, refineActivationsInput);
}


void relax_d(NeuralNetwork* nn, int layer, bool refineActivationsInput, bool soundness)
{
	(*nn)[layer]->eval(nn->getConcreteBounds<double>(layer), soundness, refineActivationsInput);
}

template <typename T>
void evalAffineExpr(NeuralNetwork* nn, T* dest, int layer, int m, const T* A, const T* b, int backsubstitute, bool soundness)
{
	size_t n = (*nn)[layer]->outputSize;
	std::shared_ptr<Matrix<T>> mA = A ? std::make_shared<Matrix<T>>(m, n, A) : nullptr;
	std::shared_ptr<Vector<T>> vb = b ? std::make_shared<Vector<T>>(m, b) : nullptr;
	Vector<T> res;
	if (mA)
		mA->mvm(res, nn->getConcreteBounds<T>(layer));
	else
		res = nn->getConcreteBounds<T>(layer);
	if (vb)
		Vector<T>::add(res, res, *vb);
	if (backsubstitute==1)
	{
		nn->evaluateAffine<T>(res, AlwaysKeep<T>(), layer, true, soundness, mA, vb);
		nn->evaluateAffine<T>(res, AlwaysKeep<T>(), layer, false, soundness, mA, vb);
	}
	else if (backsubstitute == 2)
	{
		nn->evaluateAffine<T>(res, ContainsZero<T>(), layer, true, soundness, mA, vb);
		nn->evaluateAffine<T>(res, ContainsZero<T>(), layer, false, soundness, mA, vb);
	}
	gpuErrchk(cudaMemcpy(dest, res.data(), res.memSize(), cudaMemcpyDeviceToHost));
}

void evalAffineExpr_s(NeuralNetwork* nn, float* dest, int layer, int m, const float* A, const float* b, int backsubstitute, bool soundness)
{
	 evalAffineExpr(nn, dest, layer, m, A, b, backsubstitute, soundness);
}

void evalAffineExpr_d(NeuralNetwork* nn, double* dest, int layer, int m, const double* A, const double* b, int backsubstitute, bool soundness)
{
	evalAffineExpr(nn, dest, layer, m, A, b, backsubstitute, soundness);
}


template <typename T>
void getSensitivity(NeuralNetwork* nn, T* destA, T* destb, bool up, int layer, int m, const T* A, const T* b, bool soundness)
{
	size_t n = (*nn)[layer]->outputSize;
	std::shared_ptr<Matrix<T>> mA = A ? std::make_shared<Matrix<T>>(m, n, A) : nullptr;
	std::shared_ptr<Vector<T>> vb = b ? std::make_shared<Vector<T>>(m, b) : nullptr;
	nn->getSensitivity<T>(destA, destb, layer, up, soundness, m, mA, vb);
}

void getSensitivity_s(NeuralNetwork* nn, float* destA, float* destb, bool up, int layer, int m, const float* A, const float* b, bool soundness)
{
	getSensitivity(nn, destA, destb, up, layer, m, A, b, soundness);
}

void getSensitivity_d(NeuralNetwork* nn, double* destA, double* destb, bool up, int layer, int m, const double* A, const double* b, bool soundness)
{
	getSensitivity(nn, destA, destb, up, layer, m, A, b, soundness);
}


int getOutputSize(NeuralNetwork* nn, int layer)
{
	return (*nn)[layer]->outputSize;
}
int addReLU(NeuralNetwork* nn, int parent, bool useAreaHeuristic)
{
	size_t inputSize = (*nn)[parent]->outputSize;
	return nn->addLayer(new ReLU(*nn, inputSize, useAreaHeuristic, parent));
}

int addReLUVirtual(NeuralNetwork* nn, size_t outputSize, int parent, bool useAreaHeuristic)
{
	size_t inputSize = (*nn)[parent]->outputSize;
	return nn->addLayer(new ReLU(*nn, inputSize, outputSize, useAreaHeuristic, parent));
}

int addSigmoid(NeuralNetwork* nn, int parent)
{
	size_t inputSize = (*nn)[parent]->outputSize;
	return nn->addLayer(new Sigmoid(*nn, inputSize, parent));
}

int addSigmoidVirtual(NeuralNetwork* nn, size_t outputSize, int parent)
{
	size_t inputSize = (*nn)[parent]->outputSize;
	return nn->addLayer(new Sigmoid(*nn, inputSize, outputSize, parent));
}


int addTanh(NeuralNetwork* nn, int parent)
{
	size_t inputSize = (*nn)[parent]->outputSize;
	return nn->addLayer(new Tanh(*nn, inputSize, parent));
}

int addTanhVirtual(NeuralNetwork* nn, size_t outputSize, int parent)
{
	size_t inputSize = (*nn)[parent]->outputSize;
	return nn->addLayer(new Tanh(*nn, inputSize, outputSize, parent));
}

int addActivationInfo(NeuralNetwork* nn, int layerIndex,const DPolyRInfo* info)
{
	ReLU *relulayer = dynamic_cast<ReLU*>((*nn)[layerIndex]);
	if(relulayer != nullptr){
		relulayer->addInfo(*info);
		return 0;
	}
	Sigmoid *sigmoidlayer = dynamic_cast<Sigmoid*>((*nn)[layerIndex]);
	if(sigmoidlayer != nullptr){
		sigmoidlayer->addInfo(*info);
		return 0;
	}
	Tanh *tanhlayer = dynamic_cast<Tanh*>((*nn)[layerIndex]);
	if(tanhlayer != nullptr){
		tanhlayer->addInfo(*info);
		return 0;
	}
	return -1;
}

int clearActivationInfo(NeuralNetwork* nn, int layerIndex)
{
	ReLU *relulayer = dynamic_cast<ReLU*>((*nn)[layerIndex]);
	if(relulayer != nullptr){
		relulayer->clearInfo();
		return 0;
	}
	Sigmoid *sigmoidlayer = dynamic_cast<Sigmoid*>((*nn)[layerIndex]);
	if(sigmoidlayer != nullptr){
		sigmoidlayer->clearInfo();
		return 0;
	}
	Tanh *tanhlayer = dynamic_cast<Tanh*>((*nn)[layerIndex]);
	if(tanhlayer != nullptr){
		tanhlayer->clearInfo();
		return 0;
	}
	return -1;
}

int getActivationInfoSize(NeuralNetwork* nn, int layerIndex)
{
	ReLU *relulayer = dynamic_cast<ReLU*>((*nn)[layerIndex]);
	if(relulayer != nullptr){
		return relulayer->infoSize();
	}
	Sigmoid *sigmoidlayer = dynamic_cast<Sigmoid*>((*nn)[layerIndex]);
	if(sigmoidlayer != nullptr){
		return sigmoidlayer->infoSize();
	}
	Tanh *tanhlayer = dynamic_cast<Tanh*>((*nn)[layerIndex]);
	if(tanhlayer != nullptr){
		return tanhlayer->infoSize();
	}
	return -1;
}

int addBias_d(
	NeuralNetwork* nn,
	int parent,
	const double* data // Constant vector to be added to the input. Contains as many elements as the input.
)
{
	size_t n = (*nn)[parent]->outputSize;
	std::vector<Intv<double>> candidate(n);
	for (size_t i = 0; i < n; i++)
		candidate[i] = Intv<double>(data[i], data[i]);
	return nn->addLayer(new Bias<double>(*nn, candidate, parent));
}

int addBias_s(
	NeuralNetwork* nn,
	int parent,
	const float* data // Constant vector to be added to the input. Contains as many elements as the input.
)
{
	size_t n = (*nn)[parent]->outputSize;
	std::vector<Intv<float>> candidate(n);
	for (size_t i = 0; i < n; i++)
		candidate[i] = Intv<float>(data[i], data[i]);
	return nn->addLayer(new Bias<float>(*nn, candidate, parent));
}

int changeBiasSingle(NeuralNetwork* nn, int layer, size_t index, double rangeMin, double rangeMax)
{
	Bias<double> *biaslayer = dynamic_cast<Bias<double>*>((*nn)[layer]);
	if(biaslayer != nullptr){
		biaslayer->changeBiasSingle(index, rangeMin, rangeMax);
		return 0;
	}
	Bias<float> *biaslayer_s = dynamic_cast<Bias<float>*>((*nn)[layer]);
	if(biaslayer_s != nullptr){
		biaslayer_s->changeBiasSingle(index, rangeMin, rangeMax);
		return 0;
	}
	return -1;
}

int addLinear_d(
	NeuralNetwork* nn,
	int parent,
	int outputSize, // Size of the output (a.k.a. the number of neurons of the layer)
	const double* data // Content of the matrix A in row major order. A has outputSize rows, and its number of columns equals the parent outputSize.
)
{
	size_t m = outputSize;
	size_t n = (*nn)[parent]->outputSize;
	return nn->addLayer(new Dense<double>(*nn, Matrix<double>(m, n, data), parent));
}

int getLinearRowSize(NeuralNetwork* nn, size_t layer)
{
	if(dynamic_cast<Dense<double>*>((*nn)[layer]) != nullptr){
		return dynamic_cast<Dense<double>*>((*nn)[layer])->getRowsSize();
	}
	return -1;
}
int getLinearColSize(NeuralNetwork* nn, size_t layer)
{
	if(dynamic_cast<Dense<float>*>((*nn)[layer]) != nullptr){
		return dynamic_cast<Dense<float>*>((*nn)[layer])->getColsSize();
	}
	return -1;
}

int changeLinear_single_d(NeuralNetwork* nn, size_t layer, size_t row, size_t col, double value)
{
	if(dynamic_cast<Dense<double>*>((*nn)[layer]) == nullptr){
		return -1;
	}
	dynamic_cast<Dense<double>*>((*nn)[layer])->modifyWeight_single(row, col, value);
	return 0;
}

int changeLinear_single_range(NeuralNetwork* nn, size_t layer, size_t row, size_t col, double lb, double ub)
{
	if(dynamic_cast<Dense<double>*>((*nn)[layer]) != nullptr){
		dynamic_cast<Dense<double>*>((*nn)[layer])->modifyWeight_single_range(row, col, lb, ub);
		return 0;
	}

	if(dynamic_cast<Dense<float>*>((*nn)[layer]) != nullptr){
		dynamic_cast<Dense<float>*>((*nn)[layer])->modifyWeight_single_range(row, col, lb, ub);
		return 0;
	}
	
	return -1;
}

int changeLinear_single_s(NeuralNetwork* nn, size_t layer, size_t row, size_t col, float value)
{
	if(dynamic_cast<Dense<float>*>((*nn)[layer]) == nullptr){
		return -1;
	}
	dynamic_cast<Dense<float>*>((*nn)[layer])->modifyWeight_single(row, col, value);
	return 0;
}

int addLinearInfo(NeuralNetwork* nn, int layerIndex, const DPolyRInfo* info)
{
	Dense<double> *denselayer = dynamic_cast<Dense<double>*>((*nn)[layerIndex]);
	if(denselayer != nullptr){
		denselayer->addInfo(*info);
		return 0;
	}
	Dense<float> *denselayer_s = dynamic_cast<Dense<float>*>((*nn)[layerIndex]);
	if(denselayer_s != nullptr){
		denselayer_s->addInfo(*info);
		return 0;
	}
	return -1;
}

int clearLinearInfo(NeuralNetwork* nn, int layerIndex)
{
	Dense<double> *denselayer = dynamic_cast<Dense<double>*>((*nn)[layerIndex]);
	if(denselayer != nullptr){
		denselayer->clearInfo();
		return 0;
	}
	Dense<float> *denselayer_s = dynamic_cast<Dense<float>*>((*nn)[layerIndex]);
	if(denselayer_s != nullptr){
		denselayer_s->clearInfo();
		return 0;
	}
	return -1;
}

int addLinear_s(
	NeuralNetwork* nn,
	int parent,
	int outputSize, // Size of the output (a.k.a. the number of neurons of the layer)
	const float* data // Content of the matrix A in row major order. A has outputSize rows, and its number of columns equals the parent outputSize.
)
{
	size_t m = outputSize;
	size_t n = (*nn)[parent]->outputSize;
	return nn->addLayer(new Dense<float>(*nn, Matrix<float>(m, n, data), parent));
}


int addConv2D_d(
	NeuralNetwork* nn,
	int parent,
	int filters, // number of filters
	int* kernel_shape, // dimentions of the kernel (expects an array of 2 ints, respectively the number of rows and columns
	int* input_shape, // dimentions of the input (expects an array of 3 ints, respectively the number of rows, columns and channels).
	int* stride_shape, // stride shape (expects an array of 2 ints, respectively the number of rows and columns
	int* padding, // padding (expects an array of 2 ints, respectively the number of pixels to add at the top and bottom, and the number of pixels to add on the left and right)
	const double* data // convolution coefficients (given in row major, then column, then channel and filter minor order). Contains filters*kernel_size_rows*kernel_size_cols*input_shape_channels elements.
)
{
	NeuralNetwork::Layer* conv = new Conv2D<double>(*nn,
		filters,
		kernel_shape[0], kernel_shape[1],
		input_shape[0], input_shape[1], input_shape[2],
		stride_shape[0], stride_shape[1],
		padding[0], padding[1], padding[2], padding[3],
		Matrix<double>(kernel_shape[0] * kernel_shape[1], input_shape[2] * filters, data),
		parent);
	return nn->addLayer(conv);
	/*size_t m = binding[prev].outputSize;
	std::vector<double> bias(m);
	size_t k = 0;
	if (channels_first)
		for (int i = 0; i < filters; i++)
			for (int j = 0; j < m / filters; j++)
				bias[k++] = biasData[i];
	else
		for (int j = 0; j < m / filters; j++)
			for (int i = 0; i < filters; i++)
				bias[k++] = biasData[i];
	return binding.nn->addLayer(new Bias(*binding.nn, std::make_shared<Vector>(bias), prev));*/
}

int addConv2D_s(
	NeuralNetwork* nn,
	int parent,
	int filters, // number of filters
	int* kernel_shape, // dimentions of the kernel (expects an array of 2 ints, respectively the number of rows and columns
	int* input_shape, // dimentions of the input (expects an array of 3 ints, respectively the number of rows, columns and channels).
	int* stride_shape, // stride shape (expects an array of 2 ints, respectively the number of rows and columns
	int* padding, // padding (expects an array of 2 ints, respectively the number of pixels to add at the top and bottom, and the number of pixels to add on the left and right)
	const float* data // convolution coefficients (given in row major, then column, then channel and filter minor order). Contains filters*kernel_size_rows*kernel_size_cols*input_shape_channels elements.
)
{
	NeuralNetwork::Layer* conv = new Conv2D<float>(*nn,
		filters,
		kernel_shape[0], kernel_shape[1],
		input_shape[0], input_shape[1], input_shape[2],
		stride_shape[0], stride_shape[1],
		padding[0], padding[1], padding[2], padding[3],
		Matrix<float>(kernel_shape[0] * kernel_shape[1], input_shape[2] * filters, data),
		parent);
	return nn->addLayer(conv);
	/*size_t m = binding[prev].outputSize;
	std::vector<double> bias(m);
	size_t k = 0;
	if (channels_first)
		for (int i = 0; i < filters; i++)
			for (int j = 0; j < m / filters; j++)
				bias[k++] = biasData[i];
	else
		for (int j = 0; j < m / filters; j++)
			for (int i = 0; i < filters; i++)
				bias[k++] = biasData[i];
	return binding.nn->addLayer(new Bias(*binding.nn, std::make_shared<Vector>(bias), prev));*/
}

int addMaxPool2D(
	NeuralNetwork* nn,
	int parent,
	int* pool_shape, // pool shape (expects an array of 2 ints, respectively the number of rows and columns
	int* input_shape, // dimentions of the input(expects an array of 3 ints, respectively the number of rows, columns and channels).
	int* stride_shape, // stride shape (expects an array of 2 ints, respectively the number of rows and columns
	int* padding // padding (expects an array of 2 ints, respectively the number of pixels to add at the top and bottom, and the number of pixels to add on the left and right)
)
{
	NeuralNetwork::Layer* conv = new MaxPool2D(
		*nn,
		pool_shape[0], pool_shape[1],
		input_shape[0], input_shape[1], input_shape[2],
		stride_shape[0], stride_shape[1],
		padding[0], padding[1], padding[2], padding[3],
		parent);
	return nn->addLayer(conv);
}
int addParSum(
	NeuralNetwork* nn,
	int parent1,
	int parent2
)
{
	size_t inputSize = (*nn)[parent1]->outputSize;
	return nn->addLayer(new ParSum(*nn, inputSize, parent1, parent2));
}

int addConcat(
	NeuralNetwork* nn,
	int parent1,
	int parent2
)
{
	return nn->addLayer(new Concat(*nn, (*nn)[parent1]->outputSize, parent1, (*nn)[parent2]->outputSize, parent2));
}


int enableIntegrateTestSetter(NeuralNetwork* nn, int enable)
{
	nn->enableIntegrateTestSetter(enable);
	return 0;
}

int enableRecordConcreteBounds(NeuralNetwork* nn, char mode)
{
	NeuralNetwork::RecordStatus status;
	switch(mode){
		case 'n':
			status = NeuralNetwork::RecordStatus::NO_RECORD;
			break;
		case 'o':
			status = NeuralNetwork::RecordStatus::RECORD_AND_OVERRIDE;
			break;
		case 'u':
			status = NeuralNetwork::RecordStatus::RECORD_AND_UNION;
			break;
		case 'i':
			status = NeuralNetwork::RecordStatus::RECORD_AND_UNION_IF_SAFE;
			break;
		default:
			return -1;
	}
	nn->enableRecordConcreteBoundsSetter(status);
	return 0;
}

char getRecoredConcreteBoundsMode(NeuralNetwork* nn)
{
	NeuralNetwork::RecordStatus status = nn->enableRecordConcreteBoundsGetter();
	if (status == NeuralNetwork::RecordStatus::NO_RECORD)
		return 'n';
	if (status == NeuralNetwork::RecordStatus::RECORD_AND_OVERRIDE)
		return 'o';
	if (status == NeuralNetwork::RecordStatus::RECORD_AND_UNION)
		return 'u';
	if (status == NeuralNetwork::RecordStatus::RECORD_AND_UNION_IF_SAFE)
		return 'i';
	std::cout << "Error. Unknown recode mode in. (bindings.cu);" << std::endl;
	assert(0);
}

void printConcreteBounds_s(NeuralNetwork* nn, int layer, const char* prelude)
{
	nn->printConcreteBounds<float>(layer, prelude);
}

void printConcreteBounds_d(NeuralNetwork* nn, int layer, const char* prelude)
{
	nn->printConcreteBounds<double>(layer, prelude);
}


void printRecordedBounds_s(NeuralNetwork* nn, int layer, const char* prelude)
{
	nn->printRecordedBounds<float>(layer, prelude);
}

void printRecordedBounds_d(NeuralNetwork* nn, int layer, const char* prelude)
{
	nn->printRecordedBounds<double>(layer, prelude);
}