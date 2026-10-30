"""Literal-golden harness; independent kernel and standard library only."""
import ast
from hashlib import sha256
import json
from math import isfinite,nextafter,inf
import sys
from sara_engine.evaluation.route_independent_scalar import calculate

ROW_SHA='6b577f8f687212ab1417d72b841f979f18b7a251f295e7fca1002258a2719953'
FORBIDDEN_MODULES=('sparse_route_reactivation','sparse_factor_routing','route_memory_factorial_v4',
 'route_error_binding_preflight','route_numeric_delta_preflight','route_delta_ledger_preflight',
 'route_two_phase_preflight','route_snapshot_contract')


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def audit_kernel_source(source):
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and type(node.value.value) is str:continue
        if isinstance(node,ast.ImportFrom):
            require(node.module=='math' and node.level==0 and [(n.name,n.asname) for n in node.names]==[('isfinite',None)],'Kernel imported an unregistered helper')
        else:require(isinstance(node,ast.FunctionDef) and node.name=='calculate','Kernel contains global state or extra definitions')
    functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]
    require(len(functions)==1,'Kernel has nested/extra functions')
    args=functions[0].args
    require([a.arg for a in args.args]==['tick','history','error_weight','allocated','n0','n1','side'],'Kernel received expected values or wrong input API')
    require(not args.defaults and not args.kwonlyargs and args.vararg is None and args.kwarg is None and not args.posonlyargs,'Kernel has stateful/default/extended arguments')
    require(not functions[0].decorator_list,'Kernel decorator may hide state')
    for node in ast.walk(tree):
        require(not isinstance(node,(ast.Import,ast.ClassDef,ast.AsyncFunctionDef,ast.Lambda,ast.Global,ast.Nonlocal,ast.Attribute,ast.NamedExpr,ast.AugAssign)),'Kernel independence/state boundary violated')
        if isinstance(node,ast.Call):
            require(isinstance(node.func,ast.Name) and node.func.id in ('type','isfinite','ValueError'),'Dynamic or helper call in kernel')
        if isinstance(node,ast.Assign):
            require(all(isinstance(t,ast.Name) for t in node.targets),'Kernel writes nonlocal/container state')
    return {'source_only_math_import':True,'only_registered_scalar_calls':True,'no_global_state_or_defaults':True,
            'kernel_source_sha256':sha256(source.encode()).hexdigest()}


def assert_independent_modules():
    require(not any('sara_engine.evaluation.'+name in sys.modules for name in FORBIDDEN_MODULES),'Primary calculator or learner imported')


def compare_literal(actual,expected_probability,expected_increment):
    p=actual['probability'];e=actual['error_increment']
    require(p is None or (type(p) is float and isfinite(p) and 0<=p<=1),'Invalid computed probability')
    require(expected_probability is None or (type(expected_probability) is float and isfinite(expected_probability) and 0<=expected_probability<=1),'Golden probability type/finite mismatch')
    require(type(e) is float and isfinite(e) and e>0 and type(expected_increment) is float and isfinite(expected_increment) and expected_increment>0,'Golden/computed increment type mismatch')
    p_equal=(p is None and expected_probability is None) or (type(p) is float and type(expected_probability) is float and p.hex()==expected_probability.hex())
    e_equal=e.hex()==expected_increment.hex()
    return {'probability_equal':p_equal,'error_increment_equal':e_equal,'exact_match':p_equal and e_equal}


def validate_rows(rows):
    require(type(rows) is list and len(rows)==11 and sha256(canonical(rows)).hexdigest()==ROW_SHA,'Frozen literal row roster/digest changed')


def fault_preflight(rows,source):
    # Comparator faults use literal copies, not extra kernel evaluations.
    a={'probability':rows[0][7],'error_increment':rows[0][8]}
    absent={'probability':rows[9][7],'error_increment':rows[9][8]};faults=[]
    def detected(name,operation):
        try:match=operation()
        except (ValueError,SyntaxError,TypeError):match=False
        require(not bool(match),'Undetected independent-scalar fault: '+name)
        faults.append({'fault':name,'detected':True})
    detected('golden_probability_ulp',lambda:compare_literal(a,nextafter(rows[0][7],inf),rows[0][8])['exact_match'])
    detected('golden_increment_ulp',lambda:compare_literal(a,rows[0][7],nextafter(rows[0][8],inf))['exact_match'])
    detected('missing_seen_golden',lambda:compare_literal(a,None,rows[0][8])['exact_match'])
    detected('invented_absent_golden',lambda:compare_literal(absent,.5,rows[9][8])['exact_match'])
    detected('integer_probability_golden',lambda:compare_literal(a,1,rows[0][8])['exact_match'])
    detected('integer_increment_golden',lambda:compare_literal(a,rows[0][7],1)['exact_match'])
    changed=[list(row) for row in rows];changed[0][4]=3
    detected('changed_row_digest',lambda:validate_rows(changed) or True)
    detected('primary_helper_import_source',lambda:audit_kernel_source('from sara_engine.evaluation.route_error_binding_preflight import local_check\n'+source) or True)
    detected('mutable_global_source',lambda:audit_kernel_source('CACHE = []\n'+source) or True)
    detected('dynamic_call_source',lambda:audit_kernel_source(source.replace("    public_scale =", "    eval('1')\n    public_scale =",1)) or True)
    detected('default_argument_source',lambda:audit_kernel_source(source.replace('n1, side):',"n1, side='mass_left'):")) or True)
    return faults


def analyze(spec,rows,source):
    require(spec['schema']=='sara-route-independent-scalar-selection-v1','Wrong independent scalar contract')
    require(spec['input']['rows_sha256']==ROW_SHA,'Registered literal digest changed')
    require(spec['budgets']=={'input_files':1,'input_bytes':262144,'records_per_phase':11,'value_comparisons_per_phase':22,
                'phases':2,'fault_cases':11,'result_bytes':262144,'all_artifact_bytes':2097152,
                'new_learner_executions':0,'task_prediction_scores':0,'new_acquisitions':0,'new_resource_measurements':0},'Finite scalar budget changed')
    assert_independent_modules();independence=audit_kernel_source(source);validate_rows(rows)
    before=canonical(rows);records=[]
    for index,row in enumerate(rows):
        # Expected fields are intentionally excluded from the kernel arguments.
        actual=calculate(*row[:7])
        comparison=compare_literal(actual,row[7],row[8])
        records.append({'index':index,'inputs':row[:7],'expected_probability':row[7],'expected_increment':row[8],
                        'actual':actual,'hex':{'probability':None if actual['probability'] is None else actual['probability'].hex(),
                                               'increment':actual['error_increment'].hex()},'comparison':comparison})
    require(canonical(rows)==before,'Scalar arithmetic mutated hand records')
    faults=fault_preflight(rows,source)
    require([f['fault'] for f in faults]==spec['faults'],'Independent-scalar fault roster changed')
    return {'schema':'sara-route-independent-scalar-result-v1','summary':{'records':11,'value_comparisons':22,
               'exact_matches':sum(r['comparison']['exact_match'] for r in records),
               'all_exact':all(r['comparison']['exact_match'] for r in records),'faults_detected':11,
               'scope':'independent_stateless_literal_fixture_arithmetic_only'},'records':records,
            'source_independence':independence,'fault_checks':faults,
            'status':{'independent_fixture_arithmetic_checked':True,'new_learner_executions':0,
                      'actual_learning_updates':0,'task_prediction_scores':0,'new_acquisitions':0,'new_resource_measurements':0,
                      'test_opened':False,'conditions_relaxed':False,'runtime_learner_equivalence_tested':False,
                      'runtime_causal_provenance_authenticated':False,'independent_task_usefulness_established':False,
                      'snn_advantage_established':False,'original_adaptive_component_passed':False,'training_authorized':False,'promotion_authorized':False}}
