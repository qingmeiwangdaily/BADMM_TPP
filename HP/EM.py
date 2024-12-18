import numpy as np
from BADMM12 import BregmanADMM12   
from BADMM_nuclear import BregmanADMM_nuclear  

class HawkesProcessEM:
    def __init__(self, num_types, opt, num_iterations):
        self.num_types = num_types
        self.num_iterations = num_iterations
        self.mu = np.random.rand(self.num_types)
        self.A = np.random.uniform(0.5, 0.9,
                                 (self.num_types, self.num_types))
        self.count = 0
        self.opt = opt
        self.resp1 = None
        self.resp2 = None


    def fit(self, sequences):
        for iteration in range(self.num_iterations):
            # E-step
            responsibilities = self.compute_responsibilities(sequences)
            # M-step
            self.update_parameters(sequences, responsibilities)


    def compute_responsibilities(self, sequences):
        responsibilities = []
        before_resp = []
        for sequence in sequences:
            sequence_responsibilities = np.zeros((len(sequence), len(sequence)))
            for m, (t_m, c_m) in enumerate(sequence):
                intensity = self.compute_intensity(t_m, c_m, sequence)
                for m2, (t_m2, c_m2) in enumerate(sequence):
                    if m2 < m:
                        sequence_responsibilities[m][m2] = self.A[c_m][c_m2] * np.exp(-(t_m - t_m2)) / intensity
                sequence_responsibilities[m][m]= self.mu[c_m] / intensity

            before_resp.append(sequence_responsibilities)
            
            if self.opt.mode == 'BADMM12':
                sequence_responsibilities = BregmanADMM12(sequence_responsibilities,rho=self.opt.rho, lambd=self.opt.lambd, alpha=self.opt.alpha,num_iteration=self.opt.num_iteration)
            elif self.opt.mode == 'BADMM_nuclear':
                sequence_responsibilities = BregmanADMM_nuclear(sequence_responsibilities,rho=self.opt.rho, lambd=self.opt.lambd, alpha=self.opt.alpha,num_iteration=self.opt.num_iteration)
            elif self.opt.mode == 'EM':
                sequence_responsibilities = sequence_responsibilities #EM
            else:
                raise ValueError("No such model!")
            
            responsibilities.append(sequence_responsibilities)

            self.count += 1
        self.resp1 = before_resp
        self.resp2 = responsibilities
        return responsibilities

    def compute_intensity(self, t, c, sequence):
        intensity = self.mu[c]
        for t_n, c_n in sequence:
            if t_n < t:
                intensity += self.A[c][c_n] * np.exp(-(t - t_n))
        return intensity

    def update_parameters(self, sequences, responsibilities):
        for c in range(self.num_types):
            # Update A
            for c_prime in range(self.num_types):
                up = 0.
                down = 0.
                for i, sequence in enumerate(sequences):
                    T_i, _= sequence[-1]
                    for m, (t, c_m) in enumerate(sequence):
                        if c_m == c:
                            for m2, (t2, c_m2) in enumerate(sequence):
                                if c_m2 == c_prime and m2 < m:
                                    up += responsibilities[i][m][m2]
                        if c_m == c_prime:
                            down += 1 - np.exp(-(T_i - t))
                if down == 0. :
                    down +=1e-20
                self.A[c][c_prime] = up / down
            # Update mu
            up = 0.
            down = 0.
            for i, sequence in enumerate(sequences):
                T_i,_=sequence[-1]
                for m, (t, c_m) in enumerate(sequence):
                    if c_m == c:
                        up += responsibilities[i][m][m]
                down += T_i
            self.mu[c] = up / down

