"""Conservative frozen-voltage common-subexpression extraction.

Generated RHS equations and original source files remain untouched. Hoisting
is permitted only for expressions provably independent of every cell state
and time. The cache is per node, per EP call; no cross-node averaging.
"""
from pathlib import Path
import re,json,hashlib,argparse

def generate(source,target,*,native=False):
    text=source.read_text();original=text
    fn='rhs' if native else 'gpb_full_rhs';newfn='rhs' if native else 'gpb_cached_rhs'
    if native:text=text.replace('namespace reference_grandi_sdirk','namespace cached_grandi_sdirk')
    sig=re.search(r'((?:AFSEI_CELL_HD|__device__) inline double '+fn+r'\([^{}]+)\{',text)
    assert sig
    start=sig.end();end=text.rfind('}') if not native else text.index('\n}',start)
    body=text[start:end];voltage='V' if native else 'Vm'
    definitions={};spans={}
    for m in re.finditer(r'\b(?:const\s+)?(?:double|int)\s+(\w+)\s*=\s*([^;{}]+);',body):
        name,expr=m.groups()
        # Only single-assignment declarations. Branch-mutated parameters are excluded.
        assignments=re.findall(r'\b'+re.escape(name)+r'\s*(?:[+*/-]=|=(?!=))',body)
        if len(assignments)==1:
            definitions[name]=expr;spans[name]=(m.start(),m.end())
    macros=dict(re.findall(r'^#define\s+(\w+)\s+(.+)$',text,re.M))
    known=set('exp pow log sqrt fabs fmax fmin expm1 log1p sin cos tanh std M_PI true false double int'.split())
    memo={};visiting=set()
    def classify(expr):
        expr=expr.replace('y[38]','Vm') if not native else expr
        expr=re.sub(r'p->\w+','0',expr) if native else expr
        tokens=re.findall(r'\b[A-Za-z_]\w*\b',re.sub(r'\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?','0',expr))
        dep=False
        for word in tokens:
            if word==voltage:dep=True
            elif word in known:continue
            elif word in definitions or word in macros:
                key=word
                if key in visiting:return None
                if key not in memo:
                    visiting.add(key);memo[key]=classify(definitions.get(key,macros.get(key)));visiting.remove(key)
                if memo[key] is None:return None
                dep|=memo[key]
            else:return None
        return dep
    pure={name:classify(expr) for name,expr in definitions.items()}
    pure={name:dep for name,dep in pure.items() if dep is not None}
    removed=[spans[name] for name in pure]
    calls=[]
    for m in re.finditer(r'\b(exp|pow|log|sqrt|expm1|log1p)\s*\(',body):
        a=m.start();b=m.end();depth=1
        while b<len(body) and depth:
            depth+=(body[b]=='(')-(body[b]==')');b+=1
        if depth or any(a>=lo and a<hi for lo,hi in removed):continue
        if any(a>=lo and a<hi for lo,hi,_,_ in calls):continue
        expression=body[a:b]
        if classify(expression) is True:calls.append((a,b,expression,None))
    used_text=body
    for a,b in sorted(removed+[(a,b) for a,b,_,_ in calls],reverse=True):used_text=used_text[:a]+' '+used_text[b:]
    leaves=[name for name in definitions if pure.get(name) is True and re.search(r'\b'+name+r'\b',used_text)]
    expressions={};replacement=[]
    for name in leaves:
        a,b=spans[name];replacement.append((a,b,'double '+name+'=cache.'+name+';'))
    for a,b,expr,_ in calls:
        if expr not in expressions:expressions[expr]='math_'+str(len(expressions))
        replacement.append((a,b,'cache.'+expressions[expr]))
    for a,b,repl in sorted(replacement,reverse=True):body=body[:a]+repl+body[b:]
    hd='AFSEI_CELL_HD' if native else '__device__'
    cache='GrandiVoltageCache' if native else 'GPBVoltageCache'
    make='make_voltage_cache' if native else 'make_gpb_voltage_cache'
    fields=leaves+list(expressions.values())
    helper='\nstruct '+cache+' {\n'+''.join(' double '+n+';\n' for n in fields)+'};\n'
    helper+=hd+' inline '+cache+' '+make+'(double '+voltage+') {\n'
    if native:
        prologue=text[start:text.index('Grandi_state value',start)]
        helper+=prologue
    for name,expr in definitions.items():
        if name in pure:
            expr=expr if native else expr.replace('y[38]','Vm')
            helper+=' double '+name+'='+expr+';\n'
    helper+=cache+' cache;\n'+''.join(' cache.'+name+'='+name+';\n' for name in leaves)
    helper+=''.join(' cache.'+name+'='+ (expr if native else expr.replace('y[38]','Vm'))+';\n' for expr,name in expressions.items())
    helper+='return cache;\n}\n'
    signature=sig.group(1).replace('double '+fn+'(','double '+newfn+'(')
    signature=signature.rstrip();assert signature.endswith(')')
    signature=signature[:-1]+',const '+cache+'& cache) {'
    generated=text[:sig.start()]+helper+signature+body+text[end:]
    target.write_text(generated)
    return dict(source=str(source),source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        cache_fields=fields,cache_count=len(fields),cached_variables={n:definitions[n] for n in leaves},
        cached_math_calls={n:e for e,n in expressions.items()},method_changed=False,precision='float64',
        classification='Only single-assignment expressions independent of every ionic state and time; y[38] is the external frozen Vm')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path);p.add_argument('--native',action='store_true')
    args=p.parse_args();report=generate(args.source,args.target,native=args.native)
    args.target.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
