"""Native IBAMR pilot inputs: explicitly labelled compatibility tests."""
from pathlib import Path
root=Path(__file__).resolve().parent
base='''N = 64
OPEN_BOUNDARY = "AFSEI_PRESSURE_NEUMANN"
MESH_FILE = "/work/ibamr_inputs/mesh_solid1.bin"
ACTIVE_FILE = "/work/ibamr_inputs/constant_60ms_active.bin"
OUTPUT_DIR = "OUTPUT_PLACEHOLDER"
FIELD_EVERY = FIELD_PLACEHOLDER
LOG_EVERY = 20
WALL_E = 300000.0
WALL_NU = 0.3
WALL_KAPPA = 538461.5384615385
END_SPRING = 1000000.0
END_TAPER = 0.15
TUBE_START = 0.4
TUBE_LENGTH = 2.0
IBHierarchyIntegrator {
 start_time = 0.0
 end_time = 0.002
 dt_max = DT_PLACEHOLDER
 grow_dt = 1.0
 num_cycles = 1
 time_stepping_type = "MIDPOINT_RULE"
 enable_logging = FALSE
 error_on_dt_change = TRUE
}
IBFEMethod {
 IB_delta_fcn = "IB_4"
 split_forces = FALSE
 use_jump_conditions = FALSE
 use_consistent_mass_matrix = TRUE
 IB_point_density = 2.0
 interp_use_consistent_mass_matrix = TRUE
}
INSStaggeredHierarchyIntegrator {
 mu = 0.04
 rho = 1.0
 start_time = 0.0
 end_time = 0.002
 dt_max = DT_PLACEHOLDER
 grow_dt = 1.0
 viscous_time_stepping_type = "TRAPEZOIDAL_RULE"
 convective_time_stepping_type = "ADAMS_BASHFORTH"
 convective_op_type = "PPM"
 convective_difference_form = "ADVECTIVE"
 normalize_pressure = FALSE
 cfl = 0.5
 using_vorticity_tagging = FALSE
 tag_buffer = 1
 output_U = FALSE
 output_P = FALSE
 enable_logging = FALSE
}
Main {
 solver_type = "STAGGERED"
 log_file_name = "native_ibamr.log"
 log_all_nodes = FALSE
 viz_dump_interval = 0
 restart_dump_interval = 0
 data_dump_interval = 0
 timer_dump_interval = 0
}
CartesianGeometry {
 domain_boxes = [(0,0,0),(N - 1,N - 1,N - 1)]
 x_lo = -0.7,-0.7,0.0
 x_up = 2.1,2.1,2.8
 periodic_dimension = 0,0,0
}
GriddingAlgorithm {
 max_levels = 1
 largest_patch_size { level_0 = 32,32,32 }
 smallest_patch_size { level_0 = 8,8,8 }
 efficiency_tolerance = 0.85
 combine_efficiency = 0.85
}
StandardTagAndInitialize { tagging_method = "GRADIENT_DETECTOR" }
LoadBalancer {
 bin_pack_method = "SPATIAL"
 max_workload_factor = 1.0
}
'''
for component in range(3):
    base+=f'VelocityBcCoefs_{component} {{\n'
    for kind,value in [('acoef','0.0'),('bcoef','1.0'),('gcoef','0.0')]:
        for face in range(6):base+=f' {kind}_function_{face} = "{value}"\n'
    base+='}\n'
for microseconds in (50,20,10,5):
    name=f'constant_active_dt{microseconds}us'
    text=base.replace('DT_PLACEHOLDER',repr(microseconds*1e-6)).replace('FIELD_PLACEHOLDER',str(round(1000/microseconds))).replace('OUTPUT_PLACEHOLDER',f'/work/ibamr_pressure_pilots/{name}')
    (root/(name+'.input')).write_text(text,newline='\n')
print('Created native IBAMR compatibility inputs, not formal timing claims.')
