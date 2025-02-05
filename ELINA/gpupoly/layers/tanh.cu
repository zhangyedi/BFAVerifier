
#include "tanh.h"
#include "../intv.h"
#include "../filters.h"
#include <type_traits>
#include <cuda_runtime.h>

template<> inline Vector<double>& Tanh::activCst<double>() { return activCstD; }
template<> inline Vector<double>& Tanh::activFac<double>() { return activFacD; }
template<> inline Vector<float>& Tanh::activCst<float>() { return activCstS; }
template<> inline Vector<float>& Tanh::activFac<float>() { return activFacS; }
template<> inline const Vector<double>& Tanh::activCst<double>()const { return activCstD; }
template<> inline const Vector<double>& Tanh::activFac<double>() const { return activFacD; }
template<> inline const Vector<float>& Tanh::activCst<float>() const { return activCstS; }
template<> inline const Vector<float>& Tanh::activFac<float>() const { return activFacS; }


void Tanh::eval(Vector<double>& dest, bool sound, bool precise) {eval<double>(dest, sound, precise);}
void Tanh::eval(Vector<float>& dest, bool sound, bool precise) {eval<float>(dest, sound, precise);}

void Tanh::backSubstitute(typename AffineExpr<double>::Queue& queue, const AffineExpr<double>& expr) const {backSubstitute<double>(queue, expr);}
void Tanh::backSubstitute(typename AffineExpr<float>::Queue& queue, const AffineExpr<float>& expr) const {backSubstitute<float>(queue, expr);}


#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>


template<typename T>
struct slope_point {
    T k;
    T b;
};


template<typename T>
__device__ T tanh(T z) {
    T e_x = exp(z);
    T e_nx = exp(-z);
    return (e_x - e_nx) / (e_x + e_nx);
}


template<typename T>
__device__ T d_tanh(T z) {
    T y = tanh(z);
    return 1 - y * y;
}


template<typename T>
__device__ slope_point<T> slope_point_repre(T k, T x, T y) {
    slope_point<T> sp;
    sp.k = k;
    sp.b = y - k * x;
    return sp;
}

/**
 * tanh_abstract computes two linear bounds for tanh(wx) in the range:
 *    k1 * x + b1 <= tanh(wx) <= k2 * x + b2
 * For intervals [wl, wu] (possible values of w) and [xl, xu] (possible values of x).
 *
 * The result is returned via sp1 and sp2 (two slope_point structs).
 * That is:
 *    sp1 -> (k1, b1)
 *    sp2 -> (k2, b2)
 */
template<typename T>
__device__ void tanh_weighted_abstract(T wl, T wu,
                                 T xl, T xu,
                                 slope_point<T> *sp1,  // lower bound
                                 slope_point<T> *sp2)  // upper bound
{

    // If xl == xu, effectively no "range" for x, so pick that single point.
    if (fabs(xl - xu) < 1e-10) {
        T x = xl;
        // Return (0, wl * tanh(x)) and (0, wu * tanh(x)).
        T val = tanh(x);
        T comb1 = wl * val;
        T comb2 = wu * val;
        if (comb1 <= comb2) {
            *sp1 = slope_point<T>{0, comb1};
            *sp2 = slope_point<T>{0, comb2};
        } else {
            *sp1 = slope_point<T>{0, comb2};
            *sp2 = slope_point<T>{0, comb1};
        }
        return;
    }

    // Slightly expand xl, xu for numerical safety.
    T floating_point_safeguard = 1e-5;
    xl -= floating_point_safeguard;
    xu += floating_point_safeguard;

    T gl = tanh(xl);
    T gu = tanh(xu);
    T gl_prime = d_tanh(xl);
    T gu_prime = d_tanh(xu);

    T kappa = (gu - gl) / (xu - xl);
    T kappa_prime = fmin(gl_prime, gu_prime);

        // -----------------------------
    //  Cases depending on sign of x-range and w-range
    // -----------------------------
    // 1) xl >= 0
    if (xl >= 0.0) {
        // 1a) wl >= 0
        if (wl >= 0.0) {
            *sp1 = slope_point_repre(wl * kappa, xl, wl * gl);
            *sp2 = slope_point_repre(wu * kappa_prime, xu, wu * gu);
            return;
        }
        // 1b) wu <= 0
        else if (wu <= 0.0) {
            *sp1 = slope_point_repre(wl * kappa_prime, xu, wl * gu);
            *sp2 = slope_point_repre(wu * kappa, xl, wu * gl);
            return;
        }
    }
    // 2) xu <= 0
    else if (xu <= 0.0) {
        // 2a) wl >= 0
        if (wl >= 0.0) {
            *sp1 = slope_point_repre(wu * kappa_prime, xl, wu * gl);
            *sp2 = slope_point_repre(wl * kappa, xu, wl * gu);
            return;
        }
        // 2b) wu <= 0
        else if (wu <= 0.0) {
            *sp1 = slope_point_repre(wu * kappa, xu, wu * gu);
            *sp2 = slope_point_repre(wl * kappa_prime, xl, wl * gl);
            return;
        }
    }
    // 3) xl < 0 < xu
    else {
        // 3a) wl >= 0
        if (wl >= 0.0) {
            *sp1 = slope_point_repre(wl * kappa_prime, xl, wu * gl);
            *sp2 = slope_point_repre(wl * kappa_prime, xu, wu * gu);
            return;
        }
        // 3b) wu <= 0
        else if (wu <= 0.0) {
            *sp1 = slope_point_repre(wu * kappa_prime, xu, wl * gu);
            *sp2 = slope_point_repre(wu * kappa_prime, xl, wl * gl);
            return;
        }
    }

    return;
}

/**
 * tanh_abstract computes two linear bounds for tanh(wx) in the range:
 *    k1 * x + b1 <= tanh(wx) <= k2 * x + b2
 * For intervals [wl, wu] (possible values of w) and [xl, xu] (possible values of x).
 *
 * The result is returned via sp1 and sp2 (two slope_point structs).
 * That is:
 *    sp1 -> (k1, b1)
 *    sp2 -> (k2, b2)
 */
template<typename T>
__device__ void tanh_abstract(T xl, T xu,
                                 slope_point<T> *sp1,  // lower bound
                                 slope_point<T> *sp2)  // upper bound
{

    // If xl == xu, effectively no "range" for x, so pick that single point.
    if (fabs(xl - xu) < 1e-10) {
        // Return (0, wl * tanh(x)) and (0, wu * tanh(x)).
        T val = tanh(xl);
        *sp1 = slope_point<T>{0, val};
        *sp2 = slope_point<T>{0, val};
        return;
    }

    // Slightly expand xl, xu for numerical safety.
    T floating_point_safeguard = 1e-5;
    xl -= floating_point_safeguard;
    xu += floating_point_safeguard;

    T gl = tanh(xl);
    T gu = tanh(xu);
    T gl_prime = d_tanh(xl);
    T gu_prime = d_tanh(xu);

    T kappa = (gu - gl) / (xu - xl);
    T kappa_prime = fmin(gl_prime, gu_prime);

        // -----------------------------
    //  Cases depending on sign of x-range and w-range
    // -----------------------------
    // 1) xl >= 0
    if (xl >= 0.0) {
        *sp1 = slope_point_repre(kappa, xl, gl);
        *sp2 = slope_point_repre(kappa_prime, xu, gu);
        return;
    }
    // 2) xu <= 0
    else if (xu <= 0.0) {
        *sp1 = slope_point_repre(kappa_prime, xl, gl);
        *sp2 = slope_point_repre(kappa, xu, gu);
        return;
        
    }
    // 3) xl < 0 < xu
    else {
        // 3a) wl >= 0
        *sp1 = slope_point_repre(kappa_prime, xl, gl);
        *sp2 = slope_point_repre(kappa_prime, xu, gu);
        return;
    }

    return;
}


template<typename T>
__global__ void evalTanhKernel(Intv<T>* dest, Intv<T>* activCst, Intv<T>* activFac, const Intv<T>* inputs, const size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
                Intv<T> input = inputs[idx];
                slope_point<T> lb_line, ub_line; 
                tanh_abstract(input.low, input.high, &lb_line, &ub_line);
                activFac[idx] = Intv<T>(lb_line.k, ub_line.k);
                activCst[idx] = Intv<T>(lb_line.b, ub_line.b);
                dest[idx] = Intv<T>(tanh(input.low), tanh(input.high));
        }
}

template<typename T>
__global__ void applyInfoTanh(Intv<T>* dest, Intv<T>* activCst, Intv<T>* activFac, const Intv<T>* inputs, const DPolyRInfo* infoPool, const size_t infoSize){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < infoSize) {
                DPolyRInfo info = infoPool[idx];
                if(info.attackType == DPolyRInfo::AttackType::BITFLIP_RANGE_ABSTRACT){
                        size_t cur_index = info.curVirtualNeuronIndex; 
                        size_t pre_index = info.preNeuronIndex; 
                        double rangeMin = info.rangeMin;
                        double rangeMax = info.rangeMax;
                        Intv<T> input = inputs[pre_index];
                        slope_point<T> lb_line, ub_line;
                        tanh_weighted_abstract<T>(rangeMin, rangeMax, input.low, input.high, &lb_line, &ub_line);
                        activFac[cur_index] = Intv<T>(lb_line.k, ub_line.k);
                        activCst[cur_index] = Intv<T>(lb_line.b, ub_line.b);
                        dest[cur_index] = Intv<T>(rangeMin, rangeMax) * dest[pre_index];
                }
        }
}

template<typename T>
void Tanh::applyInfo(Vector<T>& dest){
        if(infoPool.size() > 0){
                applyInfoTanh<T> <<<(infoPool.size() + 15) / 16, 16>>>(dest, activCst<T>(), activFac<T>(), nn.getConcreteBounds<T>(parent), infoPool.data(), infoPool.size());
        }
}

template<typename T>
void Tanh::eval(Vector<T>& dest, bool sound, bool precise) {
        assert(dest.size() == this->outputSize); 
        assert(dest.interval());
        if (precise){
                nn.reEvaluateLayer(parent, AlwaysKeep<T>(), true, sound);
                nn.reEvaluateLayer(parent, AlwaysKeep<T>(), false, sound);
        }
        evalTanhKernel<T> <<<(this->inputSize + 255) / 256, 256>>>(dest, activCst<T>(), activFac<T>(), nn.getConcreteBounds<T>(parent), this->inputSize);
	applyInfo(dest);
	// printf("inputs: "); nn.getConcreteBounds<T>(parent).print();
	// printf("activFac"); activFac<T>().print();
	// printf("activCst"); activCst<T>().print();
	// printf("\n");
}

template <int lgBlockSize, typename TA, typename Tdest, bool upper, typename T>
static __global__ void backSubstituteTanh(Tdest* destA, T* destb, const TA* exprA, const T* exprb, const Intv<T>* modelFac, const Intv<T>* modelCst, const size_t dest_N, const size_t expr_N, const size_t outputSize, const size_t inputSize,const DPolyRInfo *infopool, const size_t infosize)
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
static __global__ void backSubstituteTanhInit(
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


template<typename T>
void Tanh::backSubstitute(typename AffineExpr<T>::Queue& queue, const AffineExpr<T>& expr) const {
	// std::cout<< "ReLU::backSubstitute with expr.up= " << expr.up << " " << __FILE__ << ":" << __LINE__ <<"\n";
	if (expr.A)
	{
		
		std::shared_ptr<Matrix<T>> A;
		auto b = std::make_shared<Vector<T>>(expr.m, false);
		dim3 block(256, 1, 1);
		dim3 grid(expr.m, 1, 1);
		// this->infoPool.print();
		if (expr.sound)
		{
			A = std::make_shared<Matrix<T>>(expr.m, this->inputSize, true);
			if (expr.A->interval())
				if (expr.up)
					backSubstituteTanh<8, Intv<T>, Intv<T>, true,T> << <grid, block >> > (*A, *b, (const Intv<T>*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize, infoPool.data(), infoPool.size());
				else
					backSubstituteTanh<8, Intv<T>, Intv<T>, false,T> << <grid, block >> > (*A, *b, (const Intv<T>*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize, infoPool.data(), infoPool.size());
			else
				if (expr.up)
					backSubstituteTanh<8, T, Intv<T>, true,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());
				else
					backSubstituteTanh<8, T, Intv<T>, false,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());
		}
		else
		{
			assert(!expr.A->interval());
			A = std::make_shared<Matrix<T>>(expr.m, this->inputSize, false);
			if (expr.up)
				backSubstituteTanh<8, T, T, true,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());
			else
				backSubstituteTanh<8, T, T, false,T> << <grid, block >> > (*A, *b, (const T*)*expr.A, expr.b ? (const T*)*expr.b : nullptr, activFac<T>(), activCst<T>(), A->pitch(), expr.A->pitch(), this->outputSize, this->inputSize,infoPool.data(),infoPool.size());

		}
		// gpuChkKer();
		// std::cout << "enter expr.A\n"; 
		// if(expr.A.get()){ 
		// 	std::cout << "expr.A=";
		// 	expr.A -> print(); 
		// }else std::cout << "expr.A=nullptr\n";
		// if(expr.b.get()){
		// 	std::cout << "expr.b=";
		// 	expr.b -> print();
		// }else std::cout << "expr.b=nullptr\n";
		// if(A.get()){
		// 	std::cout << "A=";
		// 	A -> print();
		// }else std::cout << "A=nullptr\n";
		// if(b.get()){
		// 	std::cout << "b=";
		// 	b -> print();
		// }else std::cout << "b=nullptr\n";
		// std::cout << "Fac="; activFac<T>().print(); std::cout << "Cst="; activCst<T>().print();
		queue.emplace(expr.m, this->inputSize, parent, expr.up, expr.rows, A, b, expr.cs,expr.sound);
		// std::cout << "Leave expr.A\n";
	}
	else
	{
		
		auto A = std::make_shared<Matrix<T>>(expr.m, this->inputSize, false);
		auto b = std::make_shared<Vector<T>>(expr.evaluate(activCst<T>()));
		dim3 block(256, 1, 1);
		dim3 grid((this->outputSize + 255) / 256, expr.m, 1);
		if (expr.up)
			backSubstituteTanhInit<8, true,T> << <grid, block >> > (
				*A, A->pitch(), this->outputSize,
				expr.rows,
				activFac<T>());
		else
			backSubstituteTanhInit<8, false,T> << <grid, block >> > (
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