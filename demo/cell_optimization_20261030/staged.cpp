#include <torch/extension.h>
void staged_advance(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,double,int64_t,bool,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("advance",&staged_advance);}
