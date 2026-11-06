#include <torch/extension.h>
void reference_rhs(at::Tensor,at::Tensor,at::Tensor,at::Tensor);
void reference_advance(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,int64_t);
void reference_rl_check(at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
void experimental_euler(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("euler",&experimental_euler);m.def("rhs",&reference_rhs);m.def("advance",&reference_advance);m.def("rl_check",&reference_rl_check);}
