#include <torch/extension.h>
void shared_init(at::Tensor,at::Tensor,double);
void shared_step(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("initialize",&shared_init);m.def("step",&shared_step);}
