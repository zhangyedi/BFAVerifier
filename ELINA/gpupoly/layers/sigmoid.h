
#pragma once
#include "../network.h"
#include "../dpolyr.h"
#include <cassert>



//! Sigmoid layer

class Sigmoid: public NeuralNetwork::Layer {
        const int parent; //!< Index of the parent layer
        const int inputSize;

        DPolyRInfoPool infoPool; //!< Pool of information for the DeepPolyR model
        Vector<double> activCstD; //!< Model constants
        Vector<float> activCstS; //!< Model constants
        Vector<double> activFacD; //!< Model factors
        Vector<float> activFacS; //!< Model factors

        template<typename T> Vector<T>& activCst();
        template<typename T> Vector<T>& activFac();
        template<typename T> const Vector<T>& activCst() const;
        template<typename T> const Vector<T>& activFac() const;

public:
        Sigmoid(NeuralNetwork& nn, int size, const int parent) :NeuralNetwork::Layer(nn, size), activCstD(size, true), activCstS(size,true), activFacS(size, true), activFacD(size, true), parent(parent), inputSize(size) {}
        Sigmoid(NeuralNetwork& nn, int inputSize, int outputSize, const int parent) :NeuralNetwork::Layer(nn, outputSize), activCstD(outputSize, true), activCstS(outputSize,true), activFacS(outputSize, true), activFacD(outputSize, true), parent(parent), inputSize(inputSize) {}

        template <typename T>
        void eval(Vector<T>& dest, bool sound, bool precise);
        template <typename T>
        void backSubstitute(typename AffineExpr<T>::Queue& queue, const AffineExpr<T>& expr) const;
        
        virtual void eval(Vector<double>& dest, bool sound, bool precise);
        virtual void eval(Vector<float>& dest, bool sound, bool precise);
        virtual void backSubstitute(typename AffineExpr<double>::Queue& queue, const AffineExpr<double>& expr) const;
        virtual void backSubstitute(typename AffineExpr<float>::Queue& queue, const AffineExpr<float>& expr) const;

        virtual bool hasInternalModel() const { return true; }

        //DeepPolyR model
        int addInfo(const DPolyRInfo& info) { 
                if (info.rangeMin < 0 && info.rangeMax > 0) {
                        std::cout << "Error, Sigmoid layer only supports non-crossing zero wl and wu" << std::endl;
                        info.print();
                        return -1;
                }
                infoPool.addInfo(info);
                return 0;
        }
        void clearInfo() { infoPool.clear(); }
        int infoSize() const { return infoPool.size(); }
        template<typename T>
        void applyInfo(Vector<T>&);

};