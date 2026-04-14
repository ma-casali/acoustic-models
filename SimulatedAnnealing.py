import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tqdm

class SAOptimization:

    def __init__(
            self, 
            func: callable = None,
            constraint_func: callable = None,
            ndim: int = None,
            state_0: np.ndarray = None,
            state_lims: np.ndarray = None,
            state_inc: np.ndarray = None,
    ):
        
        """
        Docstring for __init__
        :param func: The function to be optimized. If None, the Rosenbrock function will be used.
        :type func: callable or None (default: None)
        :param constraint_func: A function that takes a state as input and returns a boolean indicating whether the state satisfies the constraints of the optimization problem. If None, no constraints will be applied.
        :type constraint_func: callable or None (default: None)
        :param ndim: The number of variables in the optimization problem.
        :type ndim: int (default: 4)
        :param state_0: np.ndarray of shape (ndim,) representing the initial state of the optimization. If None, a random state will be generated within the limits specified by state_lims.
        :type state_0: np.ndarray or None (default: None)
        :param state_lims: np.ndarray of shape (2, ndim) representing the limits for each dimension of the state. The first row should contain the lower limits and the second row should contain the upper limits. If None, the limits will be set to [-1, 1] for each dimension.
        :type state_lims: np.ndarray or None (default: None)
        :param state_inc: np.ndarray of shape (ndim,) representing the increment for each dimension when generating candidate states. If None, the increment will be set to 0.01 for each dimension.
        """
        
        # function related variables
        self.function_calls = 0
        self.function = func if func is not None else self.rosenbrock
        self.constraint_func = constraint_func if constraint_func is not None else lambda x: True

        # state variables
        if ndim is None and state_0 is not None:
            self.ndim = state_0.shape[0]
            self.state = state_0
            if state_lims is not None:
                self.state_lims = state_lims
            else:
                self.state_lims = np.array([[-2]*self.ndim, [2]*self.ndim])
        elif state_0 is None and ndim is not None:
            self.ndim = ndim
            if state_lims is not None:
                self.state = np.random.uniform(state_lims[0], state_lims[1], size = ndim)
                self.state_lims = state_lims
            else:
                self.state = np.random.uniform(-2, 2, size = ndim)
                self.state_lims = np.array([[-2]*ndim, [2]*ndim])
        elif state_0 is None and ndim is None:
            raise ValueError("Either ndim or state_0 must be provided")
        else:
            if ndim != state_0.shape[0]:
                raise ValueError("if both ndim and state_0 are provided, they must be consistent with each other (i.e. ndim must equal the length of state_0)")

            
        self.state_list = {}
        self.state_inc = state_inc if state_inc is not None else np.array([0.01]*self.ndim)
        self.state = np.int32(np.floor(self.state/self.state_inc))*self.state_inc # round initial state to nearest increment
        self.state_space = np.diff(self.state_lims, axis = 0)/self.state_inc
        self.accepted_states = self.state[np.newaxis, :]

        # energy variables
        self.energy = self.log_distortion(self.function(self.state))
        self.state_list[tuple(self.state)] = self.energy
        self.function_calls += 1
        self.energy_list = np.array([self.energy])
        self.accepted_energies = np.array([self.energy])

        # temperature variables
        self.k = 2*np.ones_like(self.state) # initial k
        self.k_list = np.array([self.k])
        self.temperature_0 = 0.5*np.ones_like(self.state) # initial temperature
        self.temperature = self.temperature_0.copy()
        self.temperature_history = np.array([self.temperature])
        self.sensitivity = self.get_sensitivity(self.log_distortion, self.rosenbrock, self.state, self.energy, self.state_inc, self.state_list, self.state_lims)
        self.function_calls += 1

        # meta-parameters
        self.reanneal_limit = np.round(np.linalg.norm(self.state_space)**(np.sqrt(np.shape(self.state)[0]/(np.shape(self.state)[0]+1))))
        self.old_reanneal_limit = self.reanneal_limit.copy()
        self.r = [0]
        self.function_calls_since_min = 0

        # tracking variables
        self.iter_since_reanneal = 0
        self.iter_since_energy_loss = 0
        self.reanneal_history = np.array(self.iter_since_reanneal)
        self.energy_loss_history = np.array(self.iter_since_energy_loss)
        self.global_min_energy = self.energy
        self.global_min_state = self.state
        self.search_list = np.array(self.state_space * self.state_inc)


    def rosenbrock(self, x):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2, axis = 0)
    
    def get_canonical_key(self, state):

        # state need not be lexicographically sorted
        coords = state.reshape(-1, 2)

        transforms = [
            [[1, 0], [0, 1]], # identity
            [[0, -1], [1, 0]], # rotate 90
            [[-1, 0], [0, -1]], # rotate 180
            [[0, 1], [-1, 0]], # rotate 270
            [[-1, 0], [0, 1]], # reflect across y-axis
            [[1, 0], [0, -1]], # reflect across x-axis
            [[0, 1], [1, 0]], # reflect across y=x
            [[0, -1], [-1, 0]], # reflect across y=-x
        ]

        possible_versions = []
        for mat in transforms:
            c = coords @ np.array(mat)
            c -= np.min(c, axis = 0) # translate so that min x,y is at the origin
            c = np.lexsort((c[:,1], c[:,0])) # sort lexicographically
            possible_versions.append(tuple(c.flatten()))

        return min(possible_versions)

    def get_sensitivity(self, distortion_fun, fun, x, y, dx, x_list, lims):

        """
        Calculates the direction of the graident of the energy function using a finite difference approximation,
        with distortion applied to the energy values to smooth the landscape and make it easier to optimize.
        """

        x_prime = np.zeros((np.size(dx), np.size(dx)))
        np.fill_diagonal(x_prime, x + dx)  
        y_prime = np.zeros_like(x, dtype = np.float64)
        for i1 in range(np.shape(x_prime)[0]):
            if tuple(x_prime[i1]) in self.state_list:
                y_prime[i1] = self.state_list[tuple(x_prime[i1])]
            else:
                y_prime[i1] = distortion_fun(fun(x_prime[i1]))

        return (y_prime - y)/(np.diagonal(x_prime) - x) * np.ceil(np.diff(lims, axis = 0).flatten())

    def log_distortion(self, x, beta = 0.5):
        """
        Distortion function to smooth the energy landscape and make it easier to optimize. This is a logarithmic 
        function that compresses the range of energy values, with a parameter beta that controls the amount of 
        distortion.
        """
        # return np.log(beta*x + 1)
        return x

    def optimize(self):

        tqdm.tqdm.write('Starting optimization with initial state: '+', '.join('{:.3f}'.format(k) for k in self.state)+' and initial energy: {:.3f}'.format(self.energy))
        update_val_old = 0
        num_vals_added = 0
        with tqdm.tqdm(total = 100, desc = "Annealing") as pbar:
            while self.r[-1] < self.reanneal_limit:

                # define new search parameters
                if len(self.accepted_energies) >= 2:
                    delta_state = np.diff(self.accepted_states[-2:,:], axis = 0)
                    if np.linalg.norm(delta_state) < np.min(self.state_inc):
                        momentum = np.zeros_like(self.state)
                    else:
                        momentum = np.diff(self.accepted_energies[-2:])/np.diff(self.accepted_states[-2:,:], axis = 0)
                        momentum[np.isnan(momentum)] = 0
                        momentum[np.isinf(momentum)] = 0
                else:
                    momentum = np.zeros_like(self.state)
                momentum = 2 - (1.5*np.exp(1))*np.exp(-(1+np.abs(momentum)))
                
                self.temperature = self.temperature_0 / (np.log(1 + self.k)) # new temperature
                
                search_dimensions = np.abs(momentum) * (np.round(self.state_space*(self.temperature/self.temperature_0)/2) * self.state_inc)
                search_dimensions = np.clip(search_dimensions, self.state_inc, self.state_space*self.state_inc)
                self.search_list = np.append(self.search_list, search_dimensions, axis = 0)

                state_distribution = []
                for i1 in range(self.ndim):
                    state_pdf = 1/(np.sqrt(2*np.pi*search_dimensions[0,i1]**2)) * np.exp(-(np.arange(self.state_lims[0,i1],self.state_lims[1,i1]+self.state_inc[i1],self.state_inc[i1]) - self.state[i1])**2/(2*search_dimensions[0,i1]**2))
                    state_pdf /= np.sum(state_pdf)
                    state_distribution.append(state_pdf)
                
                available_state_space = search_dimensions / self.state_inc
                self.reanneal_limit = np.round(np.linalg.norm(available_state_space)**(np.sqrt(np.shape(self.state)[0]/(np.shape(self.state)[0]+1))))

                # attempt to land on a new state 
                continue_flag = False
                iter = 0
                state_candidate = np.zeros_like(self.state, dtype = np.float64)
                while not continue_flag and iter < self.reanneal_limit:
                    iter += 1

                    for i1 in range(np.shape(self.state)[0]):
                        state_candidate[i1] = np.random.choice(np.arange(self.state_lims[0,i1],self.state_lims[1,i1]+self.state_inc[i1],self.state_inc[i1]), p = state_distribution[i1]) 
                    
                    # see if computation was already done
                    key = self.get_canonical_key(state_candidate)
                    if key in self.state_list:
                        energy_new = self.state_list[key]
                    else:
                        if not self.constraint_func(state_candidate):
                            continue_flag = False
                            continue 
                        else:
                            energy_new = self.log_distortion(self.function(state_candidate))
                            self.function_calls += 1

                        self.state_list[key] = energy_new
                        self.energy_list = np.append(self.energy_list, [energy_new])

                    # accept higher energies probabilistically 
                    if energy_new < self.energy:
                        continue_flag = True
                    elif 1/(1 + np.exp((energy_new - self.energy)/np.max(self.temperature))) > np.random.uniform(0,1):
                        continue_flag = True

                # if energy-search reach reanneal_limit, adjust anneal time variable k
                if iter == self.reanneal_limit: # or self.iter_since_energy_loss >= self.reanneal_limit:
                    # calculate new k using change in temperature and sensitivity
                    self.k = self.k/(1 + np.exp(self.sensitivity/np.linalg.norm(self.sensitivity)))
                    self.iter_since_reanneal = 0
                else:
                    # exponentially increase k
                    self.k = self.k + np.int64(2 * (self.iter_since_reanneal/self.reanneal_limit))
                    self.energy = energy_new
                    self.state = state_candidate
                    self.iter_since_reanneal += 1

                self.accepted_energies = np.append(self.accepted_energies, [self.energy])
                self.temperature_history = np.append(self.temperature_history, [self.temperature], axis = 0)
                self.accepted_states = np.append(self.accepted_states, [self.state], axis = 0)

                if self.energy < self.global_min_energy:
                    self.global_min_energy = self.energy
                    self.global_min_state = self.state
                    self.iter_since_energy_loss = 0

                    # calculate sensitivity using the directional gradient of energy
                    self.sensitivity = self.get_sensitivity(self.log_distortion, self.rosenbrock, self.state, self.energy, self.state_inc, self.state_list, self.state_lims)[0]
                    self.function_calls += 1
                    self.function_calls_since_min = self.function_calls
                else:
                    self.iter_since_energy_loss += 1

                self.reanneal_history = np.append(self.reanneal_history, self.iter_since_reanneal)
                self.energy_loss_history = np.append(self.energy_loss_history, self.iter_since_energy_loss)

                # check history to see how many reanneals have occurred since global energy decrease
                self.r = np.append(self.r, np.sum(self.reanneal_history[-(self.iter_since_energy_loss+1):] == 0))

                if self.iter_since_reanneal % 100 == 0:
                    update_val_new = int(self.r[-1]/self.reanneal_limit*100)
                    pbar.update(update_val_new - update_val_old)
                    update_val_old = update_val_new
                    pbar.set_postfix({
                        "num_vals_added": "{:.0f}".format(num_vals_added),
                        "Energy": "{:.3f}".format(self.energy),
                        "Best": "{:.3f}".format(self.global_min_energy),
                        "reanneal_limit": "{:.1f}".format(self.reanneal_limit),
                    })

                self.k_list = np.append(self.k_list, [self.k], axis = 0)

            tqdm.tqdm.write(f"Optimization Complete. Global Min: {self.global_min_energy:.3f}")

        return self.global_min_state, self.global_min_energy

    def plot_results(self):

        fig, axs = plt.subplots(3,1)
        axs[0].semilogy(self.accepted_energies)
        axs[0].grid(True)
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Energy')
        axs[0].scatter(np.where(self.accepted_energies == self.global_min_energy)[0][0], self.global_min_energy, marker = 'x', c ='r')  
        axs[1].plot(self.accepted_states)
        axs[1].grid(True)
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('states')

        axs[2].plot(self.k_list)
        axs[2].grid(True)
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('k-values')

        print('Function Calls: {:d} ({:d} extra)'.format(self.function_calls, self.function_calls - self.function_calls_since_min))
        print('Reanneal Limit: {:d} --> {:d}'.format(np.int64(self.old_reanneal_limit), np.int64(self.reanneal_limit)))
        print('Global Min @ f('+', '.join('{:.3f}'.format(k) for k in self.global_min_state)+') = {:.3f}'.format(self.global_min_energy))

if __name__ == "__main__":

    sa_opt = SAOptimization(ndim = 2)
    global_min_state, global_min_energy = sa_opt.optimize()
    sa_opt.plot_results()
    plt.show()