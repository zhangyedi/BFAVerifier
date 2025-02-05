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
   \file src/layers/dense.cu
   \brief Dense linear layer implementation.
   \author Fran&ccedil;ois Serre

   Implementation of class Dense, a neural network layer that performs a linear transform (a.k.a. a matrix-vector multiplication) defined in src/dense.h.
*/


#include "dense.h"


// Small lambda to compute slope k and intercept b
    // for the line passing through two points (x1, y1) and (x2, y2).
// auto getLine = [] __device__ __host__
// (T x1, T y1, T x2, T y2)
// {
// T dx = x2 - x1;
// // Since xl != xu, dx should not be 0 here.
// T k = (y2 - y1) / dx;
// T b = y1 - k * x1;
// return std::make_pair(k, b);
// };

template<typename T>
struct LineParams
{
    T k;
    T b;
};

template<typename T>
struct TwoLineParams
{
    LineParams<T> lineLower;
    LineParams<T> lineUpper;
};

template<typename T>
__device__ __host__
inline LineParams<T> getLine(T x1, T y1, T x2, T y2)
{
    T dx = x2 - x1;
    // Since xl != xu, dx should not be 0 here.
    T k = (y2 - y1) / dx;
    T b = y1 - k * x1;
    return {k, b};
}



//    Compute the two-line abstract that bounds 
//             k1 * x + b1 <= wx <= k2 * x + b2 
//     where w and x are intervals that might cross the origin
// return (k1, b1), (k2, b2), in TwoLineParams
template<typename T>
__device__ __host__
inline TwoLineParams<T>
weighted_input_relax(T wl, T wu, T xl, T xu)
{
    // Assert the preconditions (wl <= wu and xl <= xu).
    // In device code, failing this assert may lead to undefined behavior or
    // program termination during debugging. 
    // For release builds, asserts may be compiled out.
    // assert(wl <= wu && xl <= xu);

    // If xl == xu, handle special-case logic:
    if (xl == xu)
    {
        if (xl >= 0.0)
        {
            // Return (wl, 0), (wu, 0)
            return {{wl, 0.0}, {wu, 0.0}};
        }
        else if (xu <= 0.0)
        {
            // Return (wu, 0), (wl, 0)
            return {{wu, 0.0}, {wl, 0.0}};
        }
    }

    // Define points A, B, C, D:
    //   A = (xl, max(xl*wl, xl*wu))
    //   B = (xu, max(xu*wl, xu*wu))
    //   C = (xl, min(xl*wl, xl*wu))
    //   D = (xu, min(xu*wl, xu*wu))
    // A --- B
    // |     |
    // C --- D
    T Ax = xl;
    T Ay = fmax(xl * wl, xl * wu);

    T Bx = xu;
    T By = fmax(xu * wl, xu * wu);

    T Cx = xl;
    T Cy = fmin(xl * wl, xl * wu);

    T Dx = xu;
    T Dy = fmin(xu * wl, xu * wu);


    // Compute lines:
    //   (k1, b1) for C -> D
    //   (k2, b2) for A -> B
    auto lineCD = getLine(Cx, Cy, Dx, Dy);
    auto lineAB = getLine(Ax, Ay, Bx, By);

    // Return them in order: (k1, b1), (k2, b2)
    return {lineCD, lineAB};
}

template <typename T>
__global__ void applyInfoEval(Intv<T>* dest, const Intv<T>* input, const DPolyRInfo* infos, size_t infoSize)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < infoSize){
		DPolyRInfo info = infos[i];
		if(info.attackType == DPolyRInfo::AttackType::BITFLIP_RANGE_ABSTRACT){
			int weightIndex = info.preNeuronIndex; 
			int neuronIndex = info.curVirtualNeuronIndex; 
			// reuse, actually the curVirtualNeuronIndex stores the neuron index
			T rangeMin = info.rangeMin;
			T rangeMax = info.rangeMax;
			Intv<T> input_intv = input[weightIndex];
			Intv<T> weight_intv = Intv<T>(rangeMin, rangeMax);
			Intv<T> output_intv = input_intv * weight_intv;
			dest[neuronIndex] += output_intv;
		}
	}
}

template <typename TA, typename Tdest, typename TB, bool upper, bool hasA>
__global__ void applyInfoBacksubstitute(
	Intv<Tdest>* nA, const Intv<TA>* A, const int* rows_mapping, Tdest* nb, const Intv<TB>* input, const DPolyRInfo* infos, size_t infoSize, 
	size_t m, size_t nA_n, size_t A_n)
{
	size_t infoIndex = blockIdx.y; // info index
	size_t rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowIndex < m){
		DPolyRInfo info = infos[infoIndex];
		if(info.attackType == DPolyRInfo::AttackType::BITFLIP_RANGE_ABSTRACT){
			int weightIndex = info.preNeuronIndex; 
			int neuronIndex = info.curVirtualNeuronIndex; 
			// reuse, actually the curVirtualNeuronIndex stores the neuron index
			Tdest rangeMin = info.rangeMin;
			Tdest rangeMax = info.rangeMax;
			TA val; 
			if (hasA){
				val = Intv<TA>::access_dr<upper>(A[rowIndex * A_n + neuronIndex]);
			}else{
				val = (rows_mapping[rowIndex] == neuronIndex);
			}

			auto abstractLine = weighted_input_relax<Tdest>(rangeMin, rangeMax, 
				Intv<TB>::access_dr<false>(input[weightIndex]),
				Intv<TB>::access_dr<true>(input[weightIndex]));

			Tdest k1 = abstractLine.lineLower.k;
			Tdest b1 = abstractLine.lineLower.b;
			Tdest k2 = abstractLine.lineUpper.k;
			Tdest b2 = abstractLine.lineUpper.b;

			if (upper){				
				nA[rowIndex * nA_n + weightIndex] += 
					val * (val >= 0 ? k2 : k1);
				nb[rowIndex] += 
					val * (val >= 0 ? b2 : b1);
			}else{
				nA[rowIndex * nA_n + weightIndex] += 
					val * (val >= 0 ? k1 : k2);
				nb[rowIndex] +=
					val * (val >= 0 ? b1 : b2);
			}
		}
	}
}

template <typename T>
Dense<T>::Dense(NeuralNetwork& nn, const Matrix<T>& A, const int parent) :NeuralNetwork::Layer(nn, A.m()), parent(parent), A(A) {}

template <typename T>
void Dense<T>::eval(Vector<double>& dest, bool sound, bool precise)
{
	A.mvm(dest, nn.template getConcreteBounds<double>(parent)); // a simple matrix-vector multiplication
	if(infoPool.size() && dest.interval()){
		// must be an interval vector once we inject attack information.
		applyInfoEval<double> <<< (infoPool.size() + 255) / 256, 256 >>> 
			(dest, nn.template getConcreteBounds<double>(parent), infoPool.data(), infoPool.size());
	}
}
template <typename T>
void Dense<T>::eval(Vector<float>& dest, bool sound, bool precise)
{
	A.mvm(dest, nn.template getConcreteBounds<float>(parent)); // a simple matrix-vector multiplication
	if(infoPool.size() && dest.interval()){
		// must be an interval vector once we inject attack information.
		applyInfoEval<float> <<< (infoPool.size() + 255) / 256, 256 >>> 
			(dest, nn.template getConcreteBounds<float>(parent), infoPool.data(), infoPool.size());
	}
}

template <typename T>
void Dense<T>::backSubstitute(typename AffineExpr<double>::Queue& queue, const AffineExpr<double>& expr) const
{
	std::shared_ptr<Matrix<double>> nA;
	if (expr.A)
	{
		nA = std::make_shared<Matrix<double>>();
		expr.A->mmm(A, *nA, expr.sound);
		if(infoPool.size()){
			dim3 grid((expr.m + 256 - 1) / 256, infoPool.size());
			auto temp_b = expr.b ? *expr.b : Vector<double>(expr.m, false);
			if (!expr.b)
				temp_b.zeroFill();
			auto nb = std::make_shared<Vector<double>>(temp_b);
			dim3 block(256);
			if (expr.up)
				applyInfoBacksubstitute<double, double, T, true, true>
					<<< grid, block >>>(*nA, *expr.A, expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.A->pitch());
			else
				applyInfoBacksubstitute<double, double, T, false, true>
					<<< grid, block >>>(*nA, *expr.A, expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.A->pitch());
			// printf("dense.cu: 243 (parent layer=%d) (expr.up=%d) \n", parent, expr.up);
			// printf("nA->n() = %d expr.n = %d\n", nA->n(), expr.n);
			// nA->print();
			// nb->print();
			// infoPool.print();
			queue.emplace(expr.m, A.n(), parent, expr.up, expr.rows, nA, nb, ConvShape(), expr.sound);
			return;
		}
	}
	else{
		nA = std::make_shared<Matrix<double>>(A.template selectRows<double>(expr.m, expr.rows, true));
		if(infoPool.size()){
			dim3 grid((expr.m + 256 - 1) / 256, infoPool.size());
			dim3 block(256);
			auto temp_b = expr.b ? *expr.b : Vector<double>(expr.m, false);
			if (!expr.b)
				temp_b.zeroFill();
			auto nb = std::make_shared<Vector<double>>(temp_b);
			if (expr.up)
				applyInfoBacksubstitute<double, double, T, true, false>
					<<< grid, block >>>(*nA, nullptr, expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.n);
			else
				applyInfoBacksubstitute<double, double, T, false, false>
					<<< grid, block >>>(*nA, nullptr, expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.n);
			// printf("dense.cu: 264 (parent layer=%d) (expr.up=%d) \n", parent, expr.up);
			// printf("nA->n() = %d expr.n = %d\n", nA->n(), expr.n);
			// printf("sizeof(Intv<Tdest>) = %d\n", sizeof(Intv<double>));
			// printf("sizeof(Tdest) = %d\n", sizeof(double));
			// nA->print();
			// nb->print();
			// printf("nb before: (%d)\n", expr.b ? 1 : 0);
			// temp_b.print();
			// infoPool.print();
			// printf("A.n()=%d nA->n()=%d\n", A.n(), nA->n());
			// printf("expr.rows:\n");
			// std::vector<int> rows_host(expr.m);
			// cudaMemcpy(rows_host.data(), expr.rows, expr.m*sizeof(int), cudaMemcpyDeviceToHost);
			// for(int i = 0; i < expr.m; ++i)
			// 	printf("%d ", rows_host[i]);
			// printf("\n");
			queue.emplace(expr.m, A.n(), parent, expr.up, expr.rows, nA, nb, ConvShape(), expr.sound);
			return;
		}
	}
	gpuChkKer();
	queue.emplace(expr.m, A.n(), parent, expr.up, expr.rows, nA, expr.b, ConvShape(), expr.sound);
}
template <typename T>
void Dense<T>::backSubstitute(typename AffineExpr<float>::Queue& queue, const AffineExpr<float>& expr) const
{
	std::shared_ptr<Matrix<float>> nA;
	if (expr.A)
	{
		nA = std::make_shared<Matrix<float>>();
		expr.A->mmm(A, *nA, expr.sound);
		if(infoPool.size()){
			dim3 grid((expr.m + 256 - 1) / 256, infoPool.size());
			dim3 block(256);
			auto temp_b = expr.b ? *expr.b : Vector<float>(expr.m, false);
			if (!expr.b)
				temp_b.zeroFill();
			auto nb = std::make_shared<Vector<float>>(temp_b);
			if (expr.up)
				applyInfoBacksubstitute<float, float, T, true, true>
					<<< grid, block >>>(*nA, *expr.A, expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.A->n());
			else
				applyInfoBacksubstitute<float, float, T, false, true>
					<<< grid, block >>>(*nA, *expr.A, expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.A->n());
			queue.emplace(expr.m, A.n(), parent, expr.up, expr.rows, nA, nb, ConvShape(), expr.sound);
			return;
		}
	}
	else{
		nA = std::make_shared<Matrix<float>>(A.template selectRows<float>(expr.m, expr.rows, expr.sound && std::is_same<T, double>()));
		if(infoPool.size()){
			dim3 grid((expr.m + 256 - 1) / 256, infoPool.size());
			dim3 block(256);
			auto temp_b = expr.b ? *expr.b : Vector<float>(expr.m, false);
			if (!expr.b)
				temp_b.zeroFill();
			auto nb = std::make_shared<Vector<float>>(temp_b);
			if (expr.up)
				applyInfoBacksubstitute<float, float, T, true, false>
					<<< grid, block >>>(*nA, nullptr , expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.n);
			else
				applyInfoBacksubstitute<float, float, T, false, false>
					<<< grid, block >>>(*nA, nullptr, expr.rows, *nb, nn.template getConcreteBounds<T>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.n);
			queue.emplace(expr.m, A.n(), parent, expr.up, expr.rows, nA, nb, ConvShape(), expr.sound);
			return;
		}
	}
	// if(infoPool.size()){
	// 	auto& dest = *nA;
	// 	assert(dest.interval());
	// 	// std::cout << "infoPool.size() = " << infoPool.size() << std::endl;
	// 	//must be an interval vector once we inject attack information.
	// 	applyInfo<float> <<< (infoPool.size() + 255) / 256, 256 >>> (infoPool.data(), infoPool.size(), dest, A.n());
	// }
	queue.emplace(expr.m, A.n(), parent, expr.up, expr.rows, nA, expr.b, ConvShape(), expr.sound);
}

#ifdef STRONG_FP_SOUNDNESS
template <> Dense<double>::Dense(NeuralNetwork& nn, const Matrix<double>& A, const int parent) :NeuralNetwork::Layer(nn, A.m()), parent(parent), A(A),Af(std::make_shared<Matrix<float>>(A,false)) {}
template <> void Dense<double>::eval(Vector<float>& dest, bool sound, bool precise)
{
	Af->mvm(dest, nn.template getConcreteBounds<float>(parent)); // a simple matrix-vector multiplication
	if(infoPool.size() && dest.interval()){
		// must be an interval vector once we inject attack information.
		applyInfoEval<float> <<< (infoPool.size() + 255) / 256, 256 >>> 
			(dest, nn.template getConcreteBounds<float>(parent), infoPool.data(), infoPool.size());
	}
}
template <>
void Dense<double>::backSubstitute(typename AffineExpr<float>::Queue& queue, const AffineExpr<float>& expr) const
{
	std::shared_ptr<Matrix<float>> nA;
	if (expr.A)
	{
		nA = std::make_shared<Matrix<float>>();
		expr.A->mmm(*Af, *nA, expr.sound);
		if(infoPool.size()){
			dim3 grid((expr.m + 256 - 1) / 256, infoPool.size());
			dim3 block(256);
			auto nb = std::make_shared<Vector<float>>(expr.b ? *expr.b : Vector<float>(expr.m, false));
			if (expr.up)
				applyInfoBacksubstitute<float, float, double, true, true>
					<<< grid, block >>>(*nA, *expr.A, expr.rows, *nb,  nn.template getConcreteBounds<double>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.A->pitch());
			else
				applyInfoBacksubstitute<float, float, double, false, true>
					<<< grid, block >>>(*nA, *expr.A, expr.rows, *nb, nn.template getConcreteBounds<double>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.A->pitch());
			queue.emplace(expr.m, Af->n(), parent, expr.up, expr.rows, nA, nb, ConvShape(), expr.sound);
			return;
		}
	}
	else{
		nA = std::make_shared<Matrix<float>>(Af->template selectRows<float>(expr.m, expr.rows, expr.sound));
		if(infoPool.size()){
			dim3 grid((expr.m + 256 - 1) / 256, infoPool.size());
			dim3 block(256);
			auto nb = std::make_shared<Vector<float>>(expr.b ? *expr.b : Vector<float>(expr.m, false));
			if (expr.up)
				applyInfoBacksubstitute<float, float, double, true, false>
					<<< grid, block >>>(*nA, nullptr, expr.rows, *nb, nn.template getConcreteBounds<double>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.n);
			else
				applyInfoBacksubstitute<float, float, double, false, false>
					<<< grid, block >>>(*nA, nullptr, expr.rows, *nb, nn.template getConcreteBounds<double>(parent), infoPool.data(), infoPool.size(), 
						expr.m, nA->pitch(), expr.n);
			queue.emplace(expr.m, Af->n(), parent, expr.up, expr.rows, nA, nb, ConvShape(), expr.sound);
			return;
		}
	}
	queue.emplace(expr.m, Af->n(), parent, expr.up, expr.rows, nA, expr.b, ConvShape(), expr.sound);
}
#endif

template<typename T>
void Dense<T>::modifyWeight_single(size_t row, size_t col, T value){
	assert(row < A.m());
	assert(col < A.n());
	A.modify_single(row, col, value);
	Af -> modify_single(row, col, value);
}

template<typename T>
void Dense<T>::modifyWeight_column(size_t col, T value){
	assert(col < A.n());
	A.modify_column(col, value);
	Af -> modify_column(col, value);
}

template<typename T>
void Dense<T>::modifyWeight_single_range(size_t row, size_t col, T lb, T ub){
	assert(row < A.m());
	assert(col < A.n());
	if(!A.interval()){
		A.convertToIntv();
		Af -> convertToIntv();
	}
	A.modify_single_range(row, col, lb, ub);
	Af -> modify_single_range(row, col, lb, ub);
}
template<typename T> 
void Dense<T>::convertToIntv(){
	A.convertToIntv();
	Af -> convertToIntv();
}


template class Dense<double>;
template class Dense<float>;
