import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import scipy

class SAOptimization:

    def __init__(
            self, 
            func: callable = None,
            constraint_func: callable = None,
            ndim: int = None,
            state_0: np.ndarray = None,
            state_lims: np.ndarray = None,
            state_inc: np.ndarray = None,
            search_scaling: np.ndarray = None,
            max_calls: int = None
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

        # General rules of thumb: 
        # 1. The search dimensions should always be at least 2-3 times larger than the state increment
        
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
        self.search_scaling = search_scaling if search_scaling is not None else np.ones_like(self.state)

        # energy variables
        self.energy = self.log_distortion(self.function(self.state))
        self.state_list[tuple(self.state)] = self.energy
        self.function_calls += 1
        self.energy_list = np.array([self.energy])
        self.accepted_energies = np.array([self.energy])

        # temperature variables
        self.k = 2*np.ones_like(self.state) # initial k
        self.k_list = np.array([self.k])

        # - generate a starting temperature
        deltas = []
        for _ in range(20):
            test_state = self.state + np.random.uniform(-self.state_inc, self.state_inc, size = self.state.shape)
            test_state = np.clip(test_state, self.state_lims[0], self.state_lims[1])
            test_energy = self.log_distortion(self.function(test_state))

            if test_energy > self.energy:
                deltas.append(test_energy - self.energy)
            self.function_calls += 1

        if len(deltas) > 0:
            avg_delta = np.mean(deltas)
            t0_val = -avg_delta/np.log(0.99) # set initial temperature so that worse solutions are accepted with probability 99%
        else:
            t0_val = 1.0
        print(f"Initial temperature set to {t0_val:.3f} based on average energy increase of {avg_delta:.3f} from random perturbations around initial state")

        self.temperature_0 = t0_val * np.ones_like(self.state) # initial temperature
        
        # - other temperature-related variables
        self.temperature = self.temperature_0.copy()
        self.temperature_history = np.array([self.temperature])
        self.sensitivity = self.get_sensitivity(self.log_distortion, self.function, self.state, self.energy, self.state_inc, self.state_list, self.state_lims)
        self.function_calls += 1

        # meta-parameters
        # self.reanneal_limit = np.round(np.linalg.norm(self.state_space)**(np.sqrt(np.shape(self.state)[0]/(np.shape(self.state)[0]+1))))
        self.reanneal_limit = 10 * self.ndim
        self.r = [0]
        self.function_calls_since_min = 0
        self.momentum_scaling = 1.0
        self.momentum = np.zeros_like(self.state)

        # tracking variables
        self.iter_since_reanneal = 0
        self.iter_since_energy_loss = 0
        self.reanneal_history = np.array(self.iter_since_reanneal)
        self.energy_loss_history = np.array(self.iter_since_energy_loss)
        self.global_min_energy = self.energy
        self.global_min_state = self.state
        self.search_list = np.array(self.state_space * self.state_inc)
        self.max_calls = max_calls if max_calls is not None else np.inf


    def rosenbrock(self, x):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2, axis = 0)

    def get_sensitivity(self, distortion_fun, fun, x, y, dx, x_list, lims):

        """
        Calculates the direction of the gradient of the energy function using a finite difference approximation,
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
        proposed_states = 0
        accepted_states = 0
        used_previous_found = 0
        with tqdm.tqdm(total = 100, desc = "Annealing") as pbar:
            while self.r[-1] < 5 * self.ndim and self.function_calls < self.max_calls:

                # calculate momentum, the measure of how much the state is changing
                # this will generally tell you how the algorithm is trending towards a minimum and in what direction
                # momentum exists on a scale of 
                if len(self.accepted_energies) >= 2:
                    delta_state = np.diff(self.accepted_states[-2:,:], axis = 0).flatten()
                    self.momentum[delta_state != 0] += np.diff(self.accepted_energies[-2:])/delta_state[delta_state != 0]
                    self.momentum[delta_state == 0] += 0
                else:
                    self.momentum += np.zeros_like(self.state)
                
                self.temperature = self.temperature_0 / (np.log(1 + self.k)) # new temperature
                
                gamma = 0.5
                temp_ratio = (self.temperature/self.temperature_0) ** gamma
                search_dimensions = np.round(self.state_space * temp_ratio * self.search_scaling) * self.state_inc
                search_dimensions = np.clip(search_dimensions, self.state_inc * 3, self.state_space*self.state_inc)
                self.search_list = np.append(self.search_list, search_dimensions, axis = 0)

                # attempt to land on a new state 
                continue_flag = False
                state_candidate = np.zeros_like(self.state, dtype = np.float64)
                iter = 0
                while not continue_flag and iter < self.reanneal_limit:
                    iter += 1

                    for i1 in range(np.shape(self.state)[0]):
                        if self.momentum[i1] >= 100:
                            alpha = 1 * self.momentum_scaling
                        elif self.momentum[i1] <= -100:
                            alpha = -1 * self.momentum_scaling
                        else:
                            alpha = (2/(1 + np.exp(-self.momentum[i1])) - 1) * self.momentum_scaling

                        sigma = search_dimensions[0, i1]
                        
                        z = np.random.normal(0, 1)
                        w = np.random.normal(0, 1)
                        sample = alpha * abs(z) + w

                        state_candidate[i1] = self.state[i1] + (sample * sigma)
                        state_candidate[i1] = np.clip(state_candidate[i1], self.state_lims[0, i1], self.state_lims[1, i1])
                        state_candidate[i1] = np.round(state_candidate[i1]/self.state_inc[i1])*self.state_inc[i1] # round to nearest increment

                    key = tuple(state_candidate)
                    # see if computation was already done
                    if key in self.state_list:
                        energy_new = self.state_list[key]
                        used_previous_found += 1
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
                        accepted_states += 1
                    elif min(1, np.exp((self.energy - energy_new)/np.max(self.temperature))) > np.random.uniform(0,1): # metropolis criterion
                        continue_flag = True
                        accepted_states += 1

                    proposed_states += 1

                # if the number of accepted states has reached the reanneal limit, commence reannealing
                if accepted_states == self.reanneal_limit or iter == self.reanneal_limit: 
                    # calculate new k using change in temperature and sensitivity
                    self.k = self.k/(1 + np.exp(self.sensitivity/np.linalg.norm(self.sensitivity)))
                    self.iter_since_reanneal = 0
                    self.momentum_scaling = 1.0
                    accepted_states = 0
                else:
                    # exponentially increase k
                    if self.iter_since_energy_loss > (10 * self.ndim):
                        self.k += self.ndim
                    else:
                        self.k = self.k + np.int64(2 * (self.iter_since_reanneal/self.reanneal_limit))
                    self.energy = energy_new
                    self.state = state_candidate
                    self.iter_since_reanneal += 1
                    self.momentum_scaling *= 0.99

                self.accepted_energies = np.append(self.accepted_energies, [self.energy])
                self.temperature_history = np.append(self.temperature_history, [self.temperature], axis = 0)
                self.accepted_states = np.append(self.accepted_states, [self.state], axis = 0)

                if self.energy < self.global_min_energy:
                    self.global_min_energy = self.energy
                    self.global_min_state = self.state
                    self.iter_since_energy_loss = 0

                    # calculate sensitivity using the directional gradient of energy
                    self.sensitivity = self.get_sensitivity(self.log_distortion, self.function, self.state, self.energy, self.state_inc, self.state_list, self.state_lims)[0]
                    self.function_calls += 1
                    self.function_calls_since_min = self.function_calls
                else:
                    self.iter_since_energy_loss += 1

                self.reanneal_history = np.append(self.reanneal_history, self.iter_since_reanneal)
                self.energy_loss_history = np.append(self.energy_loss_history, self.iter_since_energy_loss)

                # check history to see how many reanneals have occurred since global energy decrease
                self.r = np.append(self.r, np.sum(self.reanneal_history[-(self.iter_since_energy_loss+1):] == 0))

                # if self.iter_since_reanneal % 100 == 0:
                update_val_new = int(self.r[-1]/(5 * self.ndim)*100)
                pbar.update(update_val_new - update_val_old)
                update_val_old = update_val_new
                pbar.set_postfix({
                    "search_dim_d": "({:d}, {:d})".format(int(np.min(search_dimensions[0, :15])/self.state_inc[14]), int(np.max(search_dimensions[0, :15])/self.state_inc[14])),
                    "search_dim_th": "({:d}, {:d})".format(int(np.min(search_dimensions[0, 15:])/self.state_inc[-1]), int(np.max(search_dimensions[0, 15:])/self.state_inc[-1])),
                    "function_calls": "{:d}".format(self.function_calls),
                    "Energy": "{:.2e}".format(self.energy),
                    "Best": "{:.2e}".format(self.global_min_energy),
                })

                self.k_list = np.append(self.k_list, [self.k], axis = 0)

            tqdm.tqdm.write(f"Optimization Complete. Global Min: {self.global_min_energy:.3f}")

        return self.global_min_state, self.global_min_energy, self.accepted_states, self.accepted_energies, proposed_states

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
        print('Global Min @ f('+', '.join('{:.3f}'.format(k) for k in self.global_min_state)+') = {:.3f}'.format(self.global_min_energy))

if __name__ == "__main__":

    results = np.zeros((3, 5, 2))
    for i in range(1,4):
        for j in range(5):
            opt = SAOptimization(ndim = 2**i, max_calls = 1e5)
            global_min_state, global_min_energy, accepted_states, accepted_energies, proposed_states = opt.optimize()
            results[i-1, j, 0] = global_min_energy
            results[i-1, j, 1] = len(accepted_energies)/proposed_states

            print(f"Accepted Energies: {len(accepted_energies)}, Proposed States: {proposed_states}, Acceptance Ratio: {len(accepted_energies)/proposed_states:.3f}")
            print("\n")

    print("Average optimized energy: ", np.mean(results[:, :, 0], axis = 1))
    print("Average acceptance ratio: ", np.mean(results[:,:,1], axis = 1))

    # fig, ax = plt.subplots()
    # x = sa_opt.state_lims[0,0] + np.arange(0, (sa_opt.state_lims[1,0] - sa_opt.state_lims[0,0])/sa_opt.state_inc[0])*sa_opt.state_inc[0]
    # y = sa_opt.state_lims[0,1] + np.arange(0, (sa_opt.state_lims[1,1] - sa_opt.state_lims[0,1])/sa_opt.state_inc[1])*sa_opt.state_inc[1]
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros_like(X)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         Z[i,j] = sa_opt.function(np.array([X[i,j], Y[i,j]]))
    # ax.contourf(X, Y, Z, levels = 50, cmap = 'viridis')
    # cmap = plt.get_cmap('turbo')
    # cmap_vals = cmap(np.linspace(0, 1, len(accepted_states)))
    # ax.scatter(accepted_states[:,0], accepted_states[:,1], c = cmap_vals, s = 5, alpha = 0.5)
    # ax.scatter(global_min_state[0], global_min_state[1], marker = 'x', s = 100, c = 'w')  
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Simulated Annealing Optimization Path')

    plt.show()