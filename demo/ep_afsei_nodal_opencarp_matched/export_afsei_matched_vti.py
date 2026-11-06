"""Lossless point-data visualization on the native Cartesian vertex lattice.

VTI is visualization only: simulation/assembly used the original tetrahedra.
Native x-fastest node ordering equals VTK ImageData point ordering exactly.
"""
from pathlib import Path
import argparse,json,struct,shutil,xml.etree.ElementTree as ET
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,
    default=Path('/mnt/large2/gjh/AFSEI/results/ep_afsei_nodal_opencarp_matched_20261030'))
root=parser.parse_args().source
report=json.loads((root/'report.json').read_text());assert report['status']=='completed'
out=root/'visualization';out.mkdir(exist_ok=True)
N=report['nodes'];frames=round(report['duration_ms'])+1
vm=np.memmap(root/'vm.igb',dtype='<f4',mode='r',offset=1024,shape=(frames,N))
def write_image(path,array,name):
    arr=np.asarray(array,dtype='<f4');body=arr.tobytes()
    header=f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
<ImageData WholeExtent="0 320 0 112 0 48" Origin="0 0 0" Spacing="0.0625 0.0625 0.0625">
<Piece Extent="0 320 0 112 0 48"><PointData Scalars="{name}">
<DataArray type="Float32" Name="{name}" format="appended" offset="0"/>
</PointData><CellData/></Piece></ImageData><AppendedData encoding="raw">_'''
    with path.open('wb') as f:
        f.write(header.encode());f.write(struct.pack('<Q',len(body)));f.write(body)
        f.write(b'</AppendedData></VTKFile>')
vtk=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
col=ET.SubElement(vtk,'Collection')
for k,frame in enumerate(vm):
    filename=f'ep_{k:03d}.vti';write_image(out/filename,frame,'voltage_mV')
    ET.SubElement(col,'DataSet',timestep=str(k*.001),group='',part='0',file=filename)
ET.indent(vtk);ET.ElementTree(vtk).write(out/'ep.pvd',encoding='utf-8',xml_declaration=True)
write_image(out/'activation.vti',np.load(root/'activation_ms.npy'),'activation_time_ms')
resolved=json.loads((root/'configuration.json').read_text())
resolved['resolved_monodomain_parameters']=json.loads((Path(resolved['operator_cache'])/'common_configuration.json').read_text())['parameters']
resolved['initial_voltage_mv']=report['initial_voltage_mv'];resolved['visualization_geometry_units']='mm'
(root/'resolved_configuration.json').write_text(json.dumps(resolved,indent=2))
(out/'README_zh.md').write_text(f'''打开 ep.pvd：完整 0–{report['duration_ms']:g} ms、间隔 1 ms、{frames} 帧三维电位。PVD 的时间单位为 s，几何坐标为 mm。
选择 voltage_mV；固定色标建议 [-90,80] mV，关闭自动逐帧缩放。初始 −80.9763 mV 在该固定色标下接近蓝色。
activation.vti 为完整三维激活时间（ms）。
VTI 只用于显示原始四面体网格的笛卡尔节点场，逐节点对应且没有重采样；求解仍在原始 8,601,600 个四面体上进行。VTI 单元的六面体显示拓扑不是求解单元。
''')
assert len(list(out.glob('ep_*.vti')))==frames
for name in ['view_voltage_fixed_range.pvsm','initial_fixed_range.png','initial_visualization_audit.json']:
    source=Path(__file__).resolve().parent/name
    if source.exists():shutil.copy2(source,out/name)
print(json.dumps(dict(pvd=str(out/'ep.pvd'),frames=frames,points_per_frame=N,geometry_units='mm',time_units='s',no_resampling=True),indent=2))
