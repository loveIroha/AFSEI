# AFSEI development

- This project is initialized from AFSI_GPU commit f4f88204c5a494265611237ba6d3ce317bc179bd.
- Keep the upstream repository and baseline provenance identifiable.
- The distribution is afsei; existing solver modules remain under afsi_torch until an intentional migration.
- Monodomain assembly and time integration use PyTorch; the user has authorized
  AmgX PCG with AMG preconditioning as an optional GPU linear-solve backend.
  GPB/Land ODEs may use CUDA kernels.
- Do not change numerical models while performing environment-only tasks.
- Do not commit mesh inputs, simulation outputs, compiled binaries, caches, credentials, or machine environments.
- Record actual GPU validation; CPU-only tests cannot establish CUDA kernel correctness.
- Unimplemented EP/FSI/circulation integration must be labeled as pending.
