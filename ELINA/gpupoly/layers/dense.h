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
   \file src/layers/dense.h
   \brief Dense linear layer.
   \author Fran&ccedil;ois Serre

   Neural network layer that performs a linear transform (a.k.a. a matrix-vector multiplication).
  */



#pragma once
#include "../network.h"
#include "../dpolyr.h"

//! Dense linear layer. 
/*! Neural network layer that performs a linear transform (a.k.a. multiplies its input vector with a matrix). It does not add bias nor perform any activation.
*/
template<typename T>
class Dense :public NeuralNetwork::Layer
{
	const int parent; //!< Index of the parent layer
	Matrix<T> A; //!< Linear coefficients
	DPolyRInfoPool infoPool; //!< Pool of information for the DeepPolyR model
#ifdef STRONG_FP_SOUNDNESS
	std::shared_ptr<Matrix<float>> Af;
#endif
public:
	//! Constructor
	/*!
	  Constructs a new dense layer.

	  \param A A matrix containing the coefficients of the linear transform.
	  \param parent Index of the parent layer (or 0 for the input layer).
	*/
	Dense(NeuralNetwork& nn, const Matrix<T>& A, const int parent);


	virtual void eval(Vector<double>& dest, bool sound, bool precise);
	virtual void eval(Vector<float>& dest, bool sound, bool precise);
	virtual void backSubstitute(typename AffineExpr<double>::Queue& queue, const AffineExpr<double>& expr) const;
	virtual void backSubstitute(typename AffineExpr<float>::Queue& queue, const AffineExpr<float>& expr) const;

	//! DeepPolyR change weight 
	void modifyWeight_single(size_t row, size_t col, T value);
	void modifyWeight_column(size_t col, T values);

	void modifyWeight_single_range(size_t row, size_t col, T lb, T ub);

	int getRowsSize()const{
		return A.m();
	}
	int getColsSize()const{
		return A.n();
	}
	void addInfo(const DPolyRInfo& info){
		infoPool.addInfo(info);
	}
	void clearInfo(){
		infoPool.clear();
	}
	void convertToIntv();
};
