from qiskit import *
import numpy as np
import re
import sys

class QasmToQuantumCircuitConverter:
    """A utility convert OpenQASM3 code to Qiskit `QuantumCircuit` Object.
    
    The goal of this is not to write the most efficient code or a complete code
    which follows the full language specification for OpenQASM3. But to demonstrate that
    the Task-3 of QOSF Mentorship 2021 can be achieved by just using simple built-ins and
    a few Regular expression evaluatiton techniques.
    
    NOTE: This code in no way guarantees or claims to be implementing thecomplete language
    speficiation of OpenQASM3. The original language specification is highly detailed and very
    vast and hence, this follows just a part of the OpenQASM3 specification which is more or less
    similar to OpenQASM2, but with a few extended features."""
    
    def __init__(self, program):
        self.program = program
        self.vars = {}
        self.index = 0
    
    # Get a dictionary of all variables stored in the context
    def get_variables(self):
        return self.vars
    
    # Function to evaluate math expressions defined in
    # OpenQASM code (including variables) for calculating
    # input parameters to gates
    def eval_expr(self, expr):
        modified_expr = ''
        expr_var = re.search('[_a-zA-Z]+', expr)
        if expr_var is not None and expr_var.group() != 'pi':
            var_start, var_end = expr_var.span()

            while True:
                if expr_var.group() != 'pi':
                    modified_expr = modified_expr + (expr[:var_start] + str(self.vars[expr_var.group()][1]))

                expr_var = re.search('[_a-zA-Z]+', expr[var_end:])
                if expr_var is not None:
                    var_start, var_end = expr_var.span()
                else:
                    break
            modified_expr += expr[var_end:]
            modified_expr = modified_expr.replace('pi', 'np.pi')
            
        else:
            modified_expr = expr.replace('pi', 'np.pi')
            
        return eval(modified_expr)
    
    
    # Function to get next line of code from
    # the input OpenQASM source code
    def next_line(self):
        idx = self.index
        while self.program[idx] != '\n':
            idx += 1
            if idx == len(self.program):
                break
        
        line = self.program[self.index:idx]
        self.index = idx + 1
        return line
    
    
    # Function to get previous line of code from
    # the input OpenQASM source code
    def prev_line(self, gates_start_index):
        idx = self.index
        while self.program[idx] != '\n':
            idx -= 1
            if idx <= gates_start_index:
                break
        
        line = self.program[idx:self.index]
        self.index = idx - 1
        return line
    
    
    # Function to tokenize one line of code
    def tokens(self, line):
        tokens = re.split('\s|\[|\]|\(|\)|,|;', line)
        try:
            while True:
                tokens.remove('')
        except ValueError:
            pass

        return tokens
    
    
    # Function to initialize quantum and classical registers
    # and store the corresponding objects in the program context
    def init_regs(self, tokens):
        reg_indices = [i for i in range(2, len(tokens))]
        if tokens[0] in ['qubit', 'qreg']:
            self.vars[tokens[2]] = QuantumRegister(int(tokens[1]), name=tokens[2])
            if len(tokens) > 3:
                remaining_regs = tokens[3:]
                for i in range(len(remaining_regs)):
                    self.vars[remaining_regs[i]] = QuantumRegister(int(tokens[1]), name=remaining_regs[i])

        elif tokens[0] in ['bit', 'creg']:
            self.vars[tokens[2]] = ClassicalRegister(int(tokens[1]), name=tokens[2])
            if len(tokens) > 3:
                remaining_regs = tokens[3:]
                for i in range(len(remaining_regs)):
                    self.vars[remaining_regs[i]] = ClassicalRegister(int(tokens[1]), name=remaining_regs[i])
                    
        return reg_indices
    
    
    # Function to initialize program variables
    # and store them in the program context
    def init_vars(self, tokens):
        # Limit one variable declaration per line (TODO: fix this later)
        self.vars[tokens[1]] = (tokens[0], tokens[3])
    
    
    # Wrapper for X-gate and CX-gate
    def x(self, qc, tokens, control=False):
        if not control:
            qc.x(self.vars[tokens[1]][int(tokens[2])])
        else:
            self.cx(qc, tokens, False)

        return qc
    
    
    # Wrapper for Y-gate and CY-gate
    def y(self, qc, tokens, control=False):
        if not control:
            qc.y(self.vars[tokens[1]][int(tokens[2])])
        else:
            qc.cy(self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])

        return qc
    
    
    # Wrapper for Z-gate and CZ-gate
    def z(self, qc, tokens, control=False):
        if not control:
            qc.z(self.vars[tokens[1]][int(tokens[2])])
        else:
            qc.cz(self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])

        return qc
    
    
    # Wrapper for H-gate and CH-gate
    def h(self, qc, tokens, control=False):
        if not control:
            qc.h(self.vars[tokens[1]][int(tokens[2])])
        else:
            qc.ch(self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])

        return qc
    
    
    # Wrapper for S-gate and Controlled-S-gate
    def s(self, qc, tokens, inverse=False, control=False):
        if not inverse:
            if not control:
                qc.s(self.vars[tokens[1]][int(tokens[2])])
            else:
                qc.cp(np.pi/2, self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])
        else:
            qc = self.sdg(qc, tokens, False, control)

        return qc
    
    
    # Wrapper for SDG-gate and Controlled-SDG-gate
    def sdg(self, qc, tokens, inverse=False, control=False):
        if not inverse:
            if not control:
                qc.sdg(self.vars[tokens[1]][int(tokens[2])])
            else:
                qc.cp(-np.pi/2, self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])
        else:
            qc = self.s(qc, tokens, False, control)

        return qc
    
    
    # Wrapper for T-gate and Controlled-T-gate
    def t(self, qc, tokens, inverse=False, control=False):
        if not inverse:
            if not control:
                qc.t(self.vars[tokens[1]][int(tokens[2])])
            else:
                qc.cp(np.pi/4, self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])
        else:
            qc = self.tdg(qc, tokens, False, control)

        return qc
    
    
    # Wrapper for TDG-gate  and Controlled-TDG-gate
    def tdg(self, qc, tokens, inverse=False, control=False):
        if not inverse:
            if not control:
                qc.tdg(self.vars[tokens[1]][int(tokens[2])])
            else:
                qc.cp(-np.pi/4, self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])
        else:
            qc = self.t(qc, tokens, False, control)

        return qc
    
    
    # Wrapper for RX-gate and CRX-gate
    def rx(self, qc, tokens, inverse, control):
        if not control:
            qc.rx(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                  self.vars[tokens[2]][int(tokens[3])])
        else:
            qc.crx(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                   self.vars[tokens[2]][int(tokens[3])],
                   self.vars[tokens[4]][int(tokens[5])])
        
        return qc
    
    
    # Wrapper for RY-gate and CRY-gate
    def ry(self, qc, tokens, inverse, control):
        if not control:
            qc.ry(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                  self.vars[tokens[2]][int(tokens[3])])
        else:
            qc.cry(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                   self.vars[tokens[2]][int(tokens[3])],
                   self.vars[tokens[4]][int(tokens[5])])
        
        return qc
    
    
    # Wrapper for RZ-gate and CRZ-gate
    def rz(self, qc, tokens, inverse, control):
        if not control:
            qc.rz(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                  self.vars[tokens[2]][int(tokens[3])])
        else:
            qc.crz(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                   self.vars[tokens[2]][int(tokens[3])],
                   self.vars[tokens[4]][int(tokens[5])])
        
        return qc
    
    
    # Wrapper for CX-gate
    def cx(self, qc, tokens, control):
        if not control:
            qc.cx(self.vars[tokens[1]][int(tokens[2])],
                  self.vars[tokens[3]][int(tokens[4])])
        else:
            qc = self.ccx(qc, tokens)
        return qc
    
    # Wrapper for CCX-gate
    def ccx(self, qc, tokens):
        qc.ccx(self.vars[tokens[1]][int(tokens[2])],
               self.vars[tokens[3]][int(tokens[4])],
               self.vars[tokens[5]][int(tokens[6])])
        return qc
    
    # Wrapper for SWAP-gate  and CsWAP-gate
    def swap(self, qc, tokens, control):
        if not control:
            qc.swap(self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])])
        else:
            qc.cswap(self.vars[tokens[1]][int(tokens[2])], self.vars[tokens[3]][int(tokens[4])], self.vars[tokens[5]][int(tokens[6])])
            
        return qc
    
    # Wrapper for U-gate and CU-gate
    def U(self, qc, tokens, inverse, control):
        if not control:
            qc.u(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                  self.eval_expr(tokens[2]) if not inverse else (-1 * self.eval_expr(tokens[2])),
                  self.eval_expr(tokens[3]) if not inverse else (-1 * self.eval_expr(tokens[3])),
                  self.vars[tokens[4]][int(tokens[5])])
        else:
            qc.cu(self.eval_expr(tokens[1]) if not inverse else (-1 * self.eval_expr(tokens[1])),
                  self.eval_expr(tokens[2]) if not inverse else (-1 * self.eval_expr(tokens[2])),
                  self.eval_expr(tokens[3]) if not inverse else (-1 * self.eval_expr(tokens[3])),
                  0,
                  self.vars[tokens[4]][int(tokens[5])], self.vars[tokens[6]][int(tokens[7])])
        
        return qc
    
    # Wrapper for Measure instruction
    def measure(self, qc, tokens):
        if tokens[0] == 'measure':
            qc.measure(self.vars[tokens[1]], self.vars[tokens[3]])
        else:
            qc.measure(self.vars[tokens[-1]], self.vars[tokens[0]])
            
        return qc
    
    # Function to apply different types of gates
    # based on the input token from the source code
    def apply_gate(self, qc, tokens, inverse=False, control=False):
        if tokens[0] == 'x':
            qc = self.x(qc, tokens, control)
        elif tokens[0] == 'y':
            qc = self.y(qc, tokens, control)
        elif tokens[0] == 'z':
            qc = self.z(qc, tokens, control)
        elif tokens[0] == 'h':
            qc = self.h(qc, tokens, control)
        elif tokens[0] == 's':
            qc = self.s(qc, tokens, inverse, control)
        elif tokens[0] == 'sdg':
            qc = self.sdg(qc, tokens, inverse, control)
        elif tokens[0] == 't':
            qc = self.t(qc, tokens, inverse, control)
        elif tokens[0] == 'tdg':
            qc = self.tdg(qc, tokens, inverse, conrol)
        elif tokens[0] == 'rx':
            qc = self.rx(qc, tokens, inverse, control)
        elif tokens[0] == 'ry':
            qc = self.ry(qc, tokens, inverse, control)
        elif tokens[0] == 'rz':
            qc = self.rz(qc, tokens, inverse, control)
        elif tokens[0] == 'cx':
            qc = self.cx(qc, tokens, control)
        elif tokens[0] in ['ccx', 'tofolli']:
            qc = self.ccx(qc, tokens)
        elif tokens[0] == 'swap':
            qc = self.swap(qc, tokens, control)
        elif tokens[0] == 'U':
            qc = self.U(qc, tokens, inverse, control)
        elif tokens[0] == 'reset':
            qc.reset(self.vars[tokens[1]])
        elif tokens[0] == 'barrier':
            qc.barrier(self.vars[tokens[1]])
        elif 'measure' in tokens:
            qc = self.measure(qc, tokens)
        elif tokens[0] in ['ctrl', 'negctrl']:
            # Limit to only one control (TODO: Fix this later)
            if not control:
                self.apply_gate(qc=qc, tokens=tokens[2:], inverse=inverse, control=True)
            else:
                print('Invalid to have more than one "ctrl" modifiers')
                sys.exit(-1)

        elif tokens[0] == 'inv':
            self.apply_gate(qc=qc, tokens=tokens[2:], inverse=True, control=control)
    
    
    # Function to convert OpenQASM code
    # to Qiskit QuantumCircuit object
    def to_qiskit_circuit(self):
        qc = QuantumCircuit()
        
        while self.index < len(self.program):
            line = self.next_line()
            tokens = self.tokens(line)

            if tokens == []:
                continue

            elif tokens[0] in ['OPENQASM', 'include']:
                continue

            if tokens[0] in ['qubit', 'qreg', 'bit', 'creg']:
                reg_indices = self.init_regs(tokens)
                for i in reg_indices:
                    qc.add_register(self.vars[tokens[i]])
            elif tokens[0] in ['const', 'int', 'float', 'angle']:
                self.init_vars(tokens)
            else:
                self.apply_gate(qc, tokens)
                
        return qc
    
    # Function to convert OpenQASM code to the 
    # inverse Qiskit QuantumCircuit object
    # (Not very efficient, but does the job)
    def to_inverse_qiskit_circuit(self):
        gates_start_index = 0
        prev_index = 0
        qc = QuantumCircuit()
        
        while self.index < len(self.program):
            prev_index = self.index
            
            line = self.next_line()
            tokens = self.tokens(line)

            if tokens == []:
                continue

            elif tokens[0] in ['OPENQASM', 'include']:
                continue

            if tokens[0] in ['qubit', 'qreg', 'bit', 'creg']:
                reg_indices = self.init_regs(tokens)
                for i in reg_indices:
                    qc.add_register(self.vars[tokens[i]])
            elif tokens[0] in ['const', 'int', 'float', 'angle']:
                self.init_vars(tokens)
            else:
                if gates_start_index == 0:
                    gates_start_index = prev_index
                continue
                
        self.index = len(self.program) - 1
        while self.index > gates_start_index:
            line = self.prev_line(gates_start_index)
            tokens = self.tokens(line)
            
            if tokens == []:
                continue
            
            self.apply_gate(qc, tokens, inverse=True)
            
        return qc