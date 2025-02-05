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
   \file src/layers/relu.cu
   \brief Implementation of the methods of ReLU layer.
   \author Fran&ccedil;ois Serre

   Implementation of the methods of ReLU layer.
  */
#include "relu.h"
#include "../intv.h"
#include "../filters.h"
#include <type_traits>
#include <cuda_runtime.h>


template<> inline Vector<double>& ReLU::activCst<double>() { return activCstD; }
template<> inline Vector<double>& ReLU::activFac<double>() { return activFacD; }
template<> inline Vector<float>& ReLU::activCst<float>() { return activCstS; }
template<> inline Vector<float>& ReLU::activFac<float>() { return activFacS; }
template<> inline const Vector<double>& ReLU::activCst<double>()const { return activCstD; }
template<> inline const Vector<double>& ReLU::activFac<double>() const { return activFacD; }
template<> inline const Vector<float>& ReLU::activCst<float>() const { return activCstS; }
template<> inline const Vector<float>& ReLU::activFac<float>() const { return activFacS; }
void ReLU::eval(Vector<double>& dest, bool sound, bool precise) {eval<double>(dest, sound,precise);}
void ReLU::eval(Vector<float>& dest, bool sound, bool precise) {eval<float>(dest, sound,precise);}
void ReLU::backSubstitute(typename AffineExpr<double>::Queue& queue, const AffineExpr<double>& expr) const {backSubstitute<double>(queue, expr);}
void ReLU::backSubstitute(typename AffineExpr<float>::Queue& queue, const AffineExpr<float>& expr) const {backSubstitute<float>(queue, expr);}


template <typename T, bool useAreaHeuristic>
__global__ void evalReLU(Intv<T>* dest, Intv<T>* activCst, Intv<T>* activFac, const Intv<T>* inputs, const size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	Intv<T> cur = inputs[idx];

	if (cur.high <= 0) // we're on the lower part of the ReLU
	{
		activCst[idx] = T(0);
		activFac[idx] = T(0);
		cur = T(0);
	}
	else if (cur.low >= 0) // we're on the upper part of the ReLU
	{
		activCst[idx] = T(0);
		activFac[idx] = T(1);
	}
	else // we're in between
	{
		T lambda1 = useAreaHeuristic ? (cur.high > -cur.low) : 0;
		activFac[idx].low = lambda1;
		activCst[idx].low = 0;
		
		T lambda2 = cur.high / (cur.high - cur.low);
		activFac[idx].high = lambda2;
		activCst[idx].high = Intv<T>::template mul_dr<true>(-cur.low, lambda2);

		cur.low = lambda1 * cur.low;
	}
	dest[idx] = cur;
}

template<typename T, bool useAreaHeuristic>
__global__ void applyInfoDevice(
	Intv<T>* dest, Intv<T>* activCst, Intv<T>* activFac, const Intv<T>* inputs, const size_t LayerSize,
	const DPolyRInfo * infoPool, const size_t infoSize)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= infoSize)
		return;
	const DPolyRInfo& info = infoPool[idx];
	if (info.attackType == DPolyRInfo::AttackType::NOATTACK){
		//this bit is not attacked. copy the normal value.
		dest[info.curVirtualNeuronIndex] = dest[info.preNeuronIndex];
		activCst[info.curVirtualNeuronIndex] = activCst[info.preNeuronIndex];
		activFac[info.curVirtualNeuronIndex] = activFac[info.preNeuronIndex];
		return;
	}
	if (info.attackType == DPolyRInfo::AttackType::GLICTHCING){
		//this bit is attacked by the glictching attack.
		dest[info.curVirtualNeuronIndex] = inputs[info.preNeuronIndex];
		activCst[info.curVirtualNeuronIndex] = T(0);
		activFac[info.curVirtualNeuronIndex] = T(1);
		return;
	}
	//else, the AttackType is DPolyRInfo::AttackType::BITFLIP_RANGE_ABSTRACT
	const double rangeMin = info.rangeMin; 
	const double rangeMax = info.rangeMax;
	if(info.rangeMin >= 0){
		activCst[info.curVirtualNeuronIndex].low = activCst[info.preNeuronIndex].low * rangeMin;
		activFac[info.curVirtualNeuronIndex].low = activFac[info.preNeuronIndex].low * rangeMin;
		activCst[info.curVirtualNeuronIndex].high = activCst[info.preNeuronIndex].high * rangeMax;
		activFac[info.curVirtualNeuronIndex].high = activFac[info.preNeuronIndex].high * rangeMax;
		dest[info.curVirtualNeuronIndex].low = dest[info.preNeuronIndex].low * rangeMin;
		dest[info.curVirtualNeuronIndex].high = dest[info.preNeuronIndex].high * rangeMax;
	}else if(info.rangeMax <= 0){
		activCst[info.curVirtualNeuronIndex].low = activCst[info.preNeuronIndex].high * rangeMin;
		activFac[info.curVirtualNeuronIndex].low = activFac[info.preNeuronIndex].high * rangeMin;
		activCst[info.curVirtualNeuronIndex].high = activCst[info.preNeuronIndex].low * rangeMax;
		activFac[info.curVirtualNeuronIndex].high = activFac[info.preNeuronIndex].low * rangeMax;
		dest[info.curVirtualNeuronIndex].low = dest[info.preNeuronIndex].high * rangeMin;
		dest[info.curVirtualNeuronIndex].high = dest[info.preNeuronIndex].low * rangeMax;
	}else  {
		//info.rangeMin < 0 && info.rangeMax > 0
		activCst[info.curVirtualNeuronIndex].low = activCst[info.preNeuronIndex].high * rangeMin;
		activFac[info.curVirtualNeuronIndex].low = activFac[info.preNeuronIndex].high * rangeMin;
		activCst[info.curVirtualNeuronIndex].high = activCst[info.preNeuronIndex].high * rangeMax;
		activFac[info.curVirtualNeuronIndex].high = activFac[info.preNeuronIndex].high * rangeMax;
		dest[info.curVirtualNeuronIndex].low = dest[info.preNeuronIndex].high * rangeMin;
		dest[info.curVirtualNeuronIndex].high = dest[info.preNeuronIndex].high * rangeMax;
	}
}

template<typename T>
void ReLU::applyInfo(Vector<T>& dest){
	
	if (infoPool.size() == 0)
		return;
	// std::cout<< __FILE__ << ":" << __LINE__ <<"\n";
	// infoPool.print();
	if (useAreaHeuristic)
		applyInfoDevice<T, true> << <(infoPool.size() + 255) / 256, 256 >> > (dest, activCst<T>(), activFac<T>(), nn.getConcreteBounds<T>(parent), this->outputSize, infoPool.data(), infoPool.size());
	else
		applyInfoDevice<T, false> << <(infoPool.size() + 255) / 256, 256 >> > (dest, activCst<T>(), activFac<T>(), nn.getConcreteBounds<T>(parent), this->outputSize, infoPool.data(), infoPool.size());
}

template <typename T>
void ReLU::eval(Vector<T>& dest, bool sound, bool precise)
{
	assert(dest.size() == this->outputSize);
	assert(dest.interval());
	if (precise)
	{
		nn.reEvaluateLayer(parent, ContainsZero<T>(), true, sound);
		nn.reEvaluateLayer(parent, ContainsZero<T>(), false, sound);
	}
	if(useAreaHeuristic){
		evalReLU<T, true> << <(this->inputSize + 255) / 256, 256 >> > (dest, activCst<T>(), activFac<T>(), nn.getConcreteBounds<T>(parent), this->inputSize);
		applyInfo<T>(dest);
	}
	else{
		evalReLU<T, false> << <(this->inputSize + 255) / 256, 256 >> > (dest, activCst<T>(), activFac<T>(), nn.getConcreteBounds<T>(parent), this->inputSize);
		applyInfo<T>(dest);
	}
	
	// std:: cout << "========\nReLU" << (precise ? "(precice) " : "")  << __FILE__ << ":" << __LINE__ << std::endl;
	// std :: cout << "Layer: " << parent + 1 << std::endl;
	//print ActivFac
	// std:: cout << "Fac\n" ;
	// activFac<T>().print();
	// std:: cout << "---a----\n" ;
	// activCst<T>().print();
	// std:: cout << "========\n" ;
}


template <int lgBlockSize, typename TA, typename Tdest, bool upper, typename T>
static __global__ void backSubstituteReLU(Tdest* destA, T* destb, const TA* exprA, const T* exprb, const Intv<T>* modelFac, const Intv<T>* modelCst, const size_t dest_N, const size_t expr_N, const size_t outputSize, const size_t inputSize,const DPolyRInfo *infopool, const size_t infosize)
{
	__shared__ T red[1 << lgBlockSize];
	size_t row = blockIdx.x;
	T res = (threadIdx.x == 0 && exprb) ? exprb[row] : T(0);
	for (size_t col = threadIdx.x; col < inputSize; col += (1 << lgBlockSize))
	{
		T in1 = Intv<T>::access_dr<false>(exprA[row * expr_N + col]);
		T coef1 = (in1 > 0 == upper) ? modelFac[col].high : modelFac[col].low;
		T off1 = (in1 > 0 == upper) ? modelCst[col].high : modelCst[col].low;
		Intv<T>::access_dr<false>(destA[row * dest_N + col]) = Intv<T>::template mul_dr<false>(in1, coef1);
		T res1 = Intv<T>::template fma_dr<upper>(off1, in1, res);
		if (std::is_same<Intv<T>, TA>::value)
		{
			T in2 = Intv<T>::access_dr<true>(exprA[row * expr_N + col]);
			T coef2 = (in2 > 0 == upper) ? modelFac[col].high : modelFac[col].low;
			T off2 = (in2 > 0 == upper) ? modelCst[col].high : modelCst[col].low;
			Intv<T>::access_dr<true>(destA[row * dest_N + col]) = Intv<T>::template mul_dr<true>(in2, coef2);
			T res2 = Intv<T>::template fma_dr<upper>(off2, in2, res);
			res = upper ? Intv<T>::max(res1, res2) : Intv<T>::min(res1, res2);
		}
		else
		{
			if (std::is_same<Intv<T>, Tdest>::value)
				Intv<T>::access_dr<true>(destA[row * dest_N + col]) = Intv<T>::template mul_dr<true>(in1, coef1);
			res = res1;
		}
	}
	__syncthreads();
	//process information.
	for (size_t col = threadIdx.x; inputSize <= col && col < outputSize; col += (1 << lgBlockSize))
	{
		int found = 0;
		size_t realCol = col;

		for(size_t i = 0; i < infosize; i++){
			if(infopool[i].curVirtualNeuronIndex == col){
				realCol = infopool[i].preNeuronIndex;
				found = 1;
				break;
			}
		}
		if(!found) break;

		T in1 = Intv<T>::access_dr<false>(exprA[row * expr_N + col]);
		T coef1 = (in1 > 0 == upper) ? modelFac[col].high : modelFac[col].low;
		T off1 = (in1 > 0 == upper) ? modelCst[col].high : modelCst[col].low;
		Intv<T>::access_dr<false>(destA[row * dest_N + realCol]) += Intv<T>::template mul_dr<false>(in1, coef1);
		//! My V100 do not support doubleAtomic. Using naive instead. Requires the Attack Information Does no incurr data racing.
		// atomicAdd((T*) & (Intv<T>::access_dr<false>(destA[row * dest_N + realCol])) , Intv<T>::template mul_dr<false>(in1, coef1).low);
		T res1 = Intv<T>::template fma_dr<upper>(off1, in1, res);
		if (std::is_same<Intv<T>, TA>::value)
		{
			T in2 = Intv<T>::access_dr<true>(exprA[row * expr_N + col]);
			T coef2 = (in2 > 0 == upper) ? modelFac[col].high : modelFac[col].low;
			T off2 = (in2 > 0 == upper) ? modelCst[col].high : modelCst[col].low;
			Intv<T>::access_dr<true>(destA[row * dest_N + realCol]) += Intv<T>::template mul_dr<true>(in2, coef2);
			// atomicAdd((T*)&Intv<T>::access_dr<true>(destA[row * dest_N + realCol]), Intv<T>::template mul_dr<true>(in2, coef2).high);
			T res2 = Intv<T>::template fma_dr<upper>(off2, in2, res);
			res = upper ? Intv<T>::max(res1, res2) : Intv<T>::min(res1, res2);
		}
		else
		{
			if (std::is_same<Intv<T>, Tdest>::value){
				Intv<T>::access_dr<true>(destA[row * dest_N + realCol]) += Intv<T>::template mul_dr<true>(in1, coef1);
				// atomicAdd((T*)&Intv<T>::access_dr<true>(destA[row * dest_N + realCol]), Intv<T>::template mul_dr<true>(in1, coef1).high);
			}
			res = res1;
		}
	}
	__syncthreads();
#pragma unroll
	for (int j = 0; j < lgBlockSize; j++)
	{
		red[threadIdx.x] = res;
		__syncthreads();
		int k = threadIdx.x + (1 << j);
		if (k < (1 << lgBlockSize))
			res = Intv<T>::template add_dr<upper>(res, red[k]);
		__syncthreads();
	}
	if (threadIdx.x == 0)
		destb[row] = res;
}

template <int lgBlockSize, bool upper, typename T>
static __global__ void backSubstituteReLUInit(
	T* destA, const size_t N, const size_t n,
	const int* rows,
	const Intv<T>* modelFac)
{
	int row = blockIdx.y;
	int col = threadIdx.x + (blockIdx.x << lgBlockSize);
	if (col < n)
	{
		int realRow = rows[row];
		destA[row * N + col] = (realRow == col) ? Intv<T>::access_dr<upper>(modelFac[col]) : T(0);
	}
}

template <typename T>
void ReLU::backSubstitute(typename AffineExpr<T>::Queue& queue, const AffineExpr<T>& expr) const
{
	// std::cout<< "ReLU::backSubstitute with expr.up= " << expr.up << " " << __FILE__ << ":" << __LINE__ <<"\n";
	if (expr.A)
	{
		
		std::shared_ptr<Matrix<T>> A;
		auto b = std::make_shared<Vector<T>>(expr.m, false);
		dim3 block(1024, 1, 1);
		dim3 grid(expr.m, 1, 1);
		// this->infoPool.print();
		if (expr.sound)
		{
			A = std::make_shared<Matrix<T>>(expr.m, this->inputSize, true);
			if (expr.A->interval())
				if (expr.up)
					backSubstituteReLU<10, Intv<T>, Intv<T>, true,T> << <grid, block >> > (*A, *b, (const Intv<T>*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize, infoPool.data(), infoPool.size());
				else
					backSubstituteReLU<10, Intv<T>, Intv<T>, false,T> << <grid, block >> > (*A, *b, (const Intv<T>*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize, infoPool.data(), infoPool.size());
			else
				if (expr.up)
					backSubstituteReLU<10, T, Intv<T>, true,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());
				else
					backSubstituteReLU<10, T, Intv<T>, false,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());
		}
		else
		{
			assert(!expr.A->interval());
			A = std::make_shared<Matrix<T>>(expr.m, this->inputSize, false);
			if (expr.up)
				backSubstituteReLU<10, T, T, true,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());
			else
				backSubstituteReLU<10, T, T, false,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());

		}
		gpuChkKer();
		// std::cout << "enter expr.A\n"; 
		// if(expr.A.get()){ 
		// 	std::cout << "expr.A=";
		// 	expr.A -> print(); 
		// }
		// if(expr.b.get()){
		// 	std::cout << "expr.b=";
		// 	expr.b -> print();
		// }
		// if(A.get()){
		// 	std::cout << "A=";
		// 	A -> print();
		// }
		// if(b.get()){
		// 	std::cout << "b=";
		// 	b -> print();
		// }
		// std::cout << "Fac="; activFac<T>().print(); std::cout << "Cst="; activCst<T>().print();
		queue.emplace(expr.m, this->inputSize, parent, expr.up, expr.rows, A, b, expr.cs,expr.sound);
		// std::cout << "Leave expr.A\n";
	}
	else
	{
		
		auto A = std::make_shared<Matrix<T>>(expr.m, this->inputSize, false);
		auto b = std::make_shared<Vector<T>>(expr.evaluate(activCst<T>()));
		dim3 block(1024, 1, 1);
		dim3 grid((this->outputSize + 1023) / 1024, expr.m, 1);
		if (expr.up)
			backSubstituteReLUInit<10, true,T> << <grid, block >> > (
				*A, A->pitch(), this->outputSize,
				expr.rows,
				activFac<T>());
		else
			backSubstituteReLUInit<10, false,T> << <grid, block >> > (
				*A, A->pitch(), this->outputSize,
				expr.rows,
				activFac<T>());
		// std::cout << "Enter Else\n";
		// if(expr.A.get()){ 
		// 	std::cout << "expr.A=";
		// 	expr.A -> print(); 
		// }
		// if(expr.b.get()){
		// 	std::cout << "expr.b=";
		// 	expr.b -> print();
		// }
		// if(A.get()){
		// 	std::cout << "A=";
		// 	A -> print();
		// }
		// if(b.get()){
		// 	std::cout << "b=";
		// 	b -> print();
		// }
		queue.emplace(expr.m, this->inputSize, parent, expr.up, expr.rows, A, b,ConvShape::diagonal(this->inputSize),expr.sound);
		// std::cout << "Leave Else\n";
	}
}

