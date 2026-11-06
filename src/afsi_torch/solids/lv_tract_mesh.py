"""CPU preparation of an explicit passive housing and two hollow LV tracts.

All original myocardial coordinates/connectivity are retained. The annular
basal FE surface is the conforming material interface, never a fluid cap.
Geometry is an idealization with explicit dimensions, not patient anatomy.
"""
import numpy as np


def ordered_loop(edges):
    graph={}
    for a,b in np.asarray(edges):
        graph.setdefault(int(a),[]).append(int(b));graph.setdefault(int(b),[]).append(int(a))
    if any(len(v)!=2 for v in graph.values()):raise ValueError('expected one closed basal rim')
    loop=[min(graph)];previous=None
    while True:
        choices=[i for i in graph[loop[-1]] if i!=previous]
        nxt=choices[0];previous=loop[-1]
        if nxt==loop[0]:break
        if nxt in loop:raise ValueError('multiple basal loops')
        loop.append(nxt)
    if len(loop)!=len(graph):raise ValueError('disconnected basal rim')
    return np.array(loop)


def generate_housing(X,base,inner_edges,outer_edges,*,roof_inner_z=11.85,roof_outer_z=12.,
                     top_z=12.55,chamber_radius=2.5,housing_radius=2.75,
                     port_radii=(1.,.8),port_offsets=(-1.2,1.2),wall=.2,h=.20):
    import gmsh
    X=np.asarray(X);base=np.asarray(base)
    inner=ordered_loop(inner_edges);outer=ordered_loop(outer_edges)
    center=X[inner,:2].mean(0)
    if not (X[base,2].max()<roof_inner_z<roof_outer_z<top_z):raise ValueError('ordered connector levels required')
    if any(abs(d)+r+wall>=chamber_radius for d,r in zip(port_offsets,port_radii)):
        raise ValueError('ports must fit inside the chamber with wall clearance')
    if abs(port_offsets[1]-port_offsets[0])<=sum(port_radii)+2*wall:
        raise ValueError('port walls overlap')
    points=list(X.copy());patches=[];names=[]
    def append(p):
        start=len(points);points.extend(np.asarray(p));return np.arange(start,len(points))
    def ccw(ids):
        p=np.array(points)[ids,:2];area=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
        return ids if area>0 else ids[::-1]
    inner,outer=ccw(inner),ccw(outer)
    def ring_like(ids,r,z):
        d=X[ids,:2]-center;d/=np.linalg.norm(d,axis=1)[:,None]
        return append(np.column_stack((center+r*d,np.full(len(ids),z))))
    ri=ring_like(inner,chamber_radius,roof_inner_z);ro=ring_like(outer,housing_radius,roof_outer_z)
    def circle(c,r,z):
        t=np.arange(40)*2*np.pi/40
        return append(np.column_stack((c[0]+r*np.cos(t),c[1]+r*np.sin(t),np.full(len(t),z))))
    lowers=[];uppers=[];lower_outer=[];upper_outer=[]
    for offset,r in zip(port_offsets,port_radii):
        c=center+np.array([offset,0.])
        lowers.append(circle(c,r,roof_inner_z));uppers.append(circle(c,r,top_z))
        lower_outer.append(circle(c,r+wall,roof_outer_z));upper_outer.append(circle(c,r+wall,top_z))
    def patch(name,faces):names.append(name);patches.append(np.asarray(faces,dtype=np.int64))
    # Every surface is oriented automatically after tetrahedralization.
    patch('basal_interface',base)
    def loft(name,a,b):
        rows=[]
        # Multiple axial layers bound aspect ratio without remeshing basal edges.
        layers=max(1,int(np.ceil(abs(np.array(points)[b,2].mean()-np.array(points)[a,2].mean())/h)))
        prev=a
        for k in range(1,layers+1):
            ids=b if k==layers else append((1-k/layers)*np.array(points)[a]+k/layers*np.array(points)[b])
            for j in range(len(a)):
                n=(j+1)%len(a);rows.extend([[prev[j],prev[n],ids[n]],[prev[j],ids[n],ids[j]]])
            prev=ids
        patch(name,rows)
    loft('inner_chamber',inner,ri);loft('outer_housing',outer,ro)
    for i in range(2):
        loft(f'inner_tract_{i}',lowers[i],uppers[i]);loft(f'outer_tract_{i}',lower_outer[i],upper_outer[i])
    # Plane meshes preserve all polygon edges. No original myocardial node moves.
    def planar(name,border,holes):
        gmsh.initialize()
        try:
            gmsh.option.setNumber('General.Terminal',0);gmsh.model.add(name)
            pids={}
            def wire(ids):
                for i in ids:
                    if int(i) not in pids:pids[int(i)]=gmsh.model.geo.addPoint(*points[i],h)
                lines=[gmsh.model.geo.addLine(pids[int(a)],pids[int(b)]) for a,b in zip(ids,np.roll(ids,-1))]
                for l in lines:gmsh.model.geo.mesh.setTransfiniteCurve(l,2)
                return gmsh.model.geo.addCurveLoop(lines)
            surface=gmsh.model.geo.addPlaneSurface([wire(border)]+[wire(v) for v in holes])
            gmsh.model.geo.synchronize();gmsh.model.mesh.generate(2)
            tags,coords,_=gmsh.model.mesh.getNodes();coords=coords.reshape(-1,3)
            index={};known={tuple(np.round(points[i],10)):int(i) for i in pids}
            for tag,p in zip(tags,coords):
                key=tuple(np.round(p,10))
                if key not in known:known[key]=len(points);points.append(p)
                index[int(tag)]=known[key]
            _,_,node_tags=gmsh.model.mesh.getElements(2,surface)
            patch(name,np.array([index[int(v)] for v in node_tags[0]]).reshape(-1,3))
        finally:gmsh.finalize()
    planar('inner_roof',ri,lowers);planar('outer_roof',ro,lower_outer)
    for i in range(2):planar(f'fixed_end_{i}',upper_outer[i],[uppers[i]])
    used=np.unique(np.concatenate(patches));coords=np.array(points)
    # Verify closed manifold before passing any surface to the tetrahedralizer.
    triangles=np.concatenate(patches);edges=np.sort(triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    if not np.all(counts==2):raise ValueError(f'housing surface is not closed: {np.bincount(counts).tolist()}')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('housing')
        surfaces=[];seen=set()
        for faces in patches:
            tag=gmsh.model.addDiscreteEntity(2);surfaces.append(tag)
            fresh=[int(v) for v in np.unique(faces) if int(v) not in seen];seen.update(fresh)
            gmsh.model.mesh.addNodes(2,tag,(np.array(fresh)+1).tolist(),coords[fresh].ravel().tolist())
            gmsh.model.mesh.addElementsByType(tag,2,[],(faces+1).ravel().tolist())
        shell=gmsh.model.geo.addSurfaceLoop(surfaces);volume=gmsh.model.geo.addVolume([shell]);gmsh.model.geo.synchronize()
        gmsh.option.setNumber('Mesh.Renumber',0)
        gmsh.option.setNumber('Mesh.MeshSizeMax',h);gmsh.option.setNumber('Mesh.MeshOnlyEmpty',1)
        gmsh.model.mesh.generate(3)
        tags,new,_=gmsh.model.mesh.getNodes();new=new.reshape(-1,3);mapping={}
        for tag,p in zip(tags,new):
            if int(tag)<=len(coords):mapping[int(tag)]=int(tag)-1
            else:mapping[int(tag)]=len(points);points.append(p)
        types,_,conn=gmsh.model.mesh.getElements(3,volume)
        if list(types)!=[4]:raise ValueError('P1 tetrahedral housing required')
        cells=np.array([mapping[int(v)] for v in conn[0]]).reshape(-1,4)
    finally:gmsh.finalize()
    coords=np.array(points);D=coords[cells[:,1:]]-coords[cells[:,:1]];J=np.linalg.det(D)
    neg=J<0;cells[neg,:2]=cells[neg,1::-1]
    if np.any(abs(J)<1e-12):raise ValueError(f'degenerate housing tetrahedron: count={int((abs(J)<1e-12).sum())}, min={abs(J).min()}, cells={cells[abs(J)<1e-12][:3].tolist()}')
    return dict(X=coords,cells=cells,base=base,inner_rim=inner,outer_rim=outer,
        **{name:faces for name,faces in zip(names,patches)},port_centers=np.array([center+[d,0] for d in port_offsets]),
        port_radii=np.array(port_radii),top_z=np.array(top_z),original_nodes=np.array(len(X)))
