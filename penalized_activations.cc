/***

This op implements two simple revised activation functions outlined in:
Xu, Huang, Li: Revise Saturated Activation Functions, May 2016.
(https://arxiv.org/pdf/1602.05980v2.pdf)

***/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("PenalizedTanh")
	.Input("features: float")
	.Input("a: float")
	.Output("activations: float");
	
class PenalizedTanhOp : public OpKernel {
public:
	explicit PenalizedTanhOp(OpKernelConstruction* context) : OpKernel(context) {}
	
	void Compute(OpKernelContext* context) override {
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<float>();

		const float a = context->input(1).flat<float>()(0);

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
		auto output = output_tensor->flat<float>();

		const int N = input.size();
		for (int i = 0; i < N; i++) {
			float x = input(i);
			output(i) = x > 0 ? tanh(x) : a * tanh(x);
		}
	}
};

REGISTER_KERNEL_BUILDER(Name("PenalizedTanh").Device(DEVICE_CPU), PenalizedTanhOp);


REGISTER_OP("LeakyRelu")
	.Input("features: float")
	.Input("a: float")
	.Output("activations: float");
	
class LeakyReluOp : public OpKernel {
public:
	explicit LeakyReluOp(OpKernelConstruction* context) : OpKernel(context) {}
	
	void Compute(OpKernelContext* context) override {
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<float>();

		const float a = context->input(1).flat<float>()(0);

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
		auto output = output_tensor->flat<float>();

		const int N = input.size();
		for (int i = 0; i < N; i++) {
			float x = input(i);
			output(i) = x > 0 ? x : a * x;
		}
	}
};

REGISTER_KERNEL_BUILDER(Name("LeakyRelu").Device(DEVICE_CPU), LeakyReluOp);