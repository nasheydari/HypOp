#from z3 import *
from src.data_reading import read_uf
import time
def solve_uf_unknown(path):
    constraints, header = read_uf(path)
    variables = Bools(' '.join([str(i) for i in range(header['num_nodes'])]))
    s2 = Solver()
    ands = []
    for con in constraints:
        temp = []
        for node in con:
            ind = abs(node) - 1

            if node > 0:
                temp.append(variables[ind])
            else:
                temp.append(Not(variables[ind]))
        ands.append(Or(temp[0], temp[1], temp[2]))
    s2.add(And(ands))
    res2 = s2.check()
    if res2.r == 1:
        return s2.model()
    else:
        print('UNSAT')
        print('Please using solve_maxcut_z3 or solve_maxcut_FM to get the maxcut result')
        
class SubsetSolver:
    """
        Original: https://github.com/Z3Prover/z3/blob/master/examples/python/mus/marco.py
    """
    
    constraints = []
    n = 0
    s = Solver()
    varcache = {}
    idcache = {}

    def __init__(self, constraints):
        self.constraints = constraints
        self.n = len(constraints)
        for i in range(self.n):
            self.s.add(Implies(self.c_var(i), constraints[i]))

    def c_var(self, i):
        if i not in self.varcache:
            v = Bool(str(self.constraints[abs(i)]))
            self.idcache[get_id(v)] = abs(i)
            if i >= 0:
                self.varcache[i] = v
            else:
                self.varcache[i] = Not(v)
        return self.varcache[i]

    def check_subset(self, seed):
        assumptions = self.to_c_lits(seed)
        return (self.s.check(assumptions) == sat)
        
    def to_c_lits(self, seed):
        return [self.c_var(i) for i in seed]

    def complement(self, aset):
        return set(range(self.n)).difference(aset)

    def seed_from_core(self):
        core = self.s.unsat_core()
        return [self.idcache[get_id(x)] for x in core]

    def shrink(self, seed):
        current = set(seed)
        for i in seed:
            if i not in current:
                continue
            current.remove(i)
            if not self.check_subset(current):
                current = set(self.seed_from_core())
            else:
                current.add(i)
        return current

    def grow(self, seed):
        current = seed
        for i in self.complement(current):
            current.append(i)
            if not self.check_subset(current):
                current.pop()
        return current
    
class MapSolver:
    """
        original: https://github.com/Z3Prover/z3/blob/master/examples/python/mus/marco.py
    """
    def __init__(self, n):
        """Initialization.
              Args:
             n: The number of constraints to map.
        """
        self.solver = Solver()
        self.n = n
        self.all_n = set(range(n))  # used in complement fairly frequently

    def next_seed(self):
        """Get the seed from the current model, if there is one.
             Returns:
             A seed as an array of 0-based constraint indexes.
        """
        if self.solver.check() == unsat:
             return None
        seed = self.all_n.copy()  # default to all True for "high bias"
        model = self.solver.model()
        for x in model:
             if is_false(model[x]):
                seed.remove(int(x.name()))
        return list(seed)

    def complement(self, aset):
        """Return the complement of a given set w.r.t. the set of mapped constraints."""
        return self.all_n.difference(aset)

    def block_down(self, frompoint):
        """Block down from a given set."""
        comp = self.complement(frompoint)
        self.solver.add( Or( [Bool(str(i)) for i in comp] ) )

    def block_up(self, frompoint):
        """Block up from a given set."""
        self.solver.add( Or( [Not(Bool(str(i))) for i in frompoint] ) )
        
def enumerate_sets(csolver, map, k_iter):
    """Basic MUS/MCS enumeration, as a simple example."""
    i = 0 
    while True and i < k_iter:
        seed = map.next_seed()
        if seed is None:
            return
        if csolver.check_subset(seed):
            i+=1
            MSS = csolver.grow(seed)
            yield ("MSS", csolver.to_c_lits(MSS))
            map.block_down(MSS)
        else:
            seed = csolver.seed_from_core()
            MUS = csolver.shrink(seed)
            #yield ("MUS", csolver.to_c_lits(MUS))
            map.block_up(MUS)
            
def get_id(x):
    return Z3_get_ast_id(x.ctx.ref(),x.as_ast())

def solve_maxsat_z3(path, k_iter):
    start = time.time()
    constraints, header = read_uf(path)
    variables = Bools(' '.join([str(i) for i in range(header['num_nodes'])]))
    ands = []
    for con in constraints:
        temp = []
        for node in con:
            ind = abs(node) - 1
            if node > 0:
                temp.append(variables[ind])
            else:
                temp.append(Not(variables[ind]))
        ands.append(Or(temp[0], temp[1], temp[2]))

    
    csolver = SubsetSolver(ands)
    msolver = MapSolver(n=csolver.n)
    ans_n = 0
    i = 0
    for orig, lits in enumerate_sets(csolver, msolver, k_iter):
        i += 1
        if len(lits) > ans_n:
            ans = lits
            ans_n = len(lits)
            print(ans_n, i)
    print("total time:", time.time() - start)
    s2 = Solver()
    s2.add(And(ans))
    s2.check()
    return s2.model(), ans, ans_n

from pysat.examples.fm import FM
from pysat.formula import WCNF
def solve_maxsat_FM(path):
    cnf = WCNF()
    constraints, header = read_uf(path)
    for c in constraints:
        cnf.append(c, weight=100)
    fm = FM(cnf, verbose=0)
    fm.compute()
    fm.init(cnf)
    res = fm.model
    num_vio = 0
    for c in constraints:
        flag=True
        for node in c:
            if node * res[abs(node)-1] > 0:
                flag = False
                break
        if flag:
            num_vio+=1
    return num_vio, res