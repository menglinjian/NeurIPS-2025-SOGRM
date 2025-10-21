from copy import copy,deepcopy
import traceback
import numpy as np
import sys
import math
from numpy import double, dtype, seterr
import time

def setup_seed(seed):
  import random
  np.random.seed(seed)
  random.seed(seed)


class GeneralizedGame():
  def __init__(self) -> None:
    pass

  def get_loss_for_player(self, player, strategies):
    pass

  def compute_best_response(self, player, strategies, return_loss=False):
    g = self.get_loss_for_player(player, strategies=strategies)
    res_index = np.argmin(g)
    res = np.zeros(g.shape)
    res[res_index] = 1.0
    if not return_loss:
      return res
    else:
      return res, g

  def compute_duality_gap(self, strategies):
    for strategy in strategies:
      temp = float(np.sum(strategy) - 1.0)
      if temp > 1e-3 or temp< -1e-3:
        return 100
    res = 0
    for p in range(self.player_num):
      best_response, loss = self.compute_best_response(p, strategies=strategies, return_loss=True)
      res = res + float(np.matmul(loss.T, strategies[p] - best_response))
    return res


class TwoPlayerZeroSumGame(GeneralizedGame):
  def __init__(self, m, n, ratio=1.0):
    # max_x min_y x^T A y
    import scipy
    if ratio<1.0:
      A = scipy.sparse.rand(m,n,ratio)
    else:
      A = -1 + 2*np.random.random((m,n))
    self.A = A/2.0
    self.player_num = 2
    self.dim_for_player = {}
    self.dim_for_player[str(0)] = m
    self.dim_for_player[str(1)] = n
    self.dims = [m,n]
  
  def get_loss_for_player(self, player, strategies):
    opponent_strategy = strategies[1 - player]
    if player==0:
      g = np.matmul(-self.A, opponent_strategy)
    else:
      g = np.matmul(self.A.T, opponent_strategy)
    return g


def get_one_player_random_strategy(m):
  res = np.random.rand(m, 1)
  res = res/float(np.sum(res))
  return res


def get_one_player_uniform_strategy(m):
  res = np.zeros((m,1))
  res[:,:] = 1.0/m
  return res


def projection_simplex_sort(v, z=1):
    try:
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        # print(cond, u, cssv)
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w
    except:
        traceback.print_exc()
        exit()


"""
In all solvers, please only use " RMinmizer = 'RM+' ", do not use " RMinmizer = 'RM' " since we do not test " RMinmizer = 'RM' "
"""

class RMSolver():
  def __init__(self, game, RMinmizer = 'RM', random_init_strategy=False, return_current_strategy=False, alternating_update=False, linear_average=False) -> None:
    self.game = game
    self.return_current_strategy = return_current_strategy
    self.alternating_update = alternating_update
    self.RMinmizer = RMinmizer
    self.iteration = 0
    self.current_strategy = []
    if not random_init_strategy:
      for p in range(self.game.player_num):
        self.current_strategy.append(get_one_player_uniform_strategy(self.game.dim_for_player[str(p)]))
    else:
      for p in range(self.game.player_num):
        self.current_strategy.append(get_one_player_uniform_strategy(self.game.dim_for_player[str(p)]))
    
    self.average_strategy = []
    for p in range(self.game.player_num):
      self.average_strategy.append(np.zeros((self.game.dim_for_player[str(p)],1)))

    self.accumulated_regret = []
    for p in range(self.game.player_num):
      self.accumulated_regret.append(np.zeros((self.game.dim_for_player[str(p)],1)))

    self.linear_average = linear_average
  
  def evaluate_and_update_policy(self):
    self.iteration = self.iteration + 1
    self.eta = 1#1/self.iteration
    if not self.alternating_update:
      for i in range(self.game.player_num):
        strategy_q_value = - self.game.get_loss_for_player(i, self.current_strategy)
        strategy_value = np.matmul(self.current_strategy[i].T, strategy_q_value)
        current_regret = strategy_q_value - strategy_value
        if self.RMinmizer=='RM':
          self.accumulated_regret[i] = self.accumulated_regret[i] + self.eta*current_regret
        elif self.RMinmizer=='RM+':
          self.accumulated_regret[i] = self.accumulated_regret[i] + self.eta*current_regret
          self.accumulated_regret[i] = np.maximum(self.accumulated_regret[i], np.zeros(self.accumulated_regret[i].shape))
      for i in range(self.game.player_num):
        self.current_strategy[i] = self.RM(self.accumulated_regret[i])
        if not self.linear_average:
          self.average_strategy[i] = self.average_strategy[i] + self.current_strategy[i]
        else:
          self.average_strategy[i] = self.average_strategy[i] + self.iteration*self.current_strategy[i]

  def output_strategy(self):
    if not self.return_current_strategy:
      if self.iteration>0:
        res_average_strategy = []
        for i in range(self.game.player_num):
          res_average_strategy.append(self.average_strategy[i]/float(np.sum(self.average_strategy[i])))
      else:
        res_average_strategy = self.current_strategy
      return deepcopy(res_average_strategy), deepcopy(res_average_strategy)
    else:
      if self.iteration>0:
        res_average_strategy = []
        for i in range(self.game.player_num):
          res_average_strategy.append(self.average_strategy[i]/float(np.sum(self.average_strategy[i])))
      else:
        res_average_strategy = self.current_strategy
      return deepcopy(res_average_strategy), deepcopy(self.current_strategy)
    
  def RM(self, accumulated_regret):
    temp = np.maximum(accumulated_regret, np.zeros(accumulated_regret.shape)) #+ 0.1*self.eta
    if np.sum(temp)>0:
      res = temp/np.sum(temp)
    else:
      res = np.ones((accumulated_regret.shape))
      res = res/np.sum(res)
    return res

  def get_regret_from_accumalated_regret(self, player, accumalated_regret_ex, accumalated_regret_in, return_more = False):
    accumalated_regret_in_strategy = []
    for i in range(self.game.player_num):
      accumalated_regret_in_strategy.append(self.RM(accumalated_regret_in[i]))
    strategy_q_value = - self.game.get_loss_for_player(player, accumalated_regret_in_strategy)
    accumulated_regret_strategy = self.RM(accumalated_regret_ex[player])#self.current_strategy[i]#
    inter_strategy_value = np.matmul(accumulated_regret_strategy.T, strategy_q_value)
    inter_regret = strategy_q_value - inter_strategy_value
    if not return_more:
      return inter_regret
    else:
      return inter_regret, -strategy_q_value


#PRM+
class PRMSolver(RMSolver):
  def __init__(self, game, RMinmizer = 'RM', random_init_strategy=False, return_current_strategy=False, alternating_update=False) -> None:
    super().__init__(game=game, RMinmizer = RMinmizer, random_init_strategy=random_init_strategy, return_current_strategy=return_current_strategy, alternating_update=alternating_update)
    self.perturbed_weight = 0
    self.accumulated_regret = []
    for p in range(self.game.player_num):
      self.accumulated_regret.append(self.perturbed_weight*np.ones((self.game.dim_for_player[str(p)],1)))

  def evaluate_and_update_policy(self):
    self.iteration = self.iteration + 1
    if not self.alternating_update:
      current_regrets = []
      for i in range(self.game.player_num):
        current_regrets.append([])
      for i in range(self.game.player_num):
        strategy_q_value = - self.game.get_loss_for_player(i, self.current_strategy)
        strategy_value = np.matmul(self.current_strategy[i].T, strategy_q_value)
        current_regrets[i] = strategy_q_value - strategy_value 
        if self.RMinmizer=='RM':
          self.accumulated_regret[i] = self.accumulated_regret[i] + current_regrets[i]
        elif self.RMinmizer=='RM+':
          self.accumulated_regret[i] = self.accumulated_regret[i] + current_regrets[i]
          self.accumulated_regret[i] = np.maximum(self.accumulated_regret[i], self.perturbed_weight*np.ones(self.accumulated_regret[i].shape))
      for i in range(self.game.player_num):
        external_accumulated_regret = np.maximum(self.accumulated_regret[i] + current_regrets[i], self.perturbed_weight*np.ones(self.accumulated_regret[i].shape))
        self.current_strategy[i] = self.RM(external_accumulated_regret)
        if not self.linear_average:
          self.average_strategy[i] = self.average_strategy[i] + self.current_strategy[i]
        else:
          self.average_strategy[i] = self.average_strategy[i] + self.iteration*self.current_strategy[i]


#Smooth PRM+
class SPRMSolver(RMSolver):
  def __init__(self, game, RMinmizer = 'RM', eta=1, random_init_strategy=False, return_current_strategy=False, alternating_update=False, lower_z=1) -> None:
    super().__init__(game=game, RMinmizer = RMinmizer, random_init_strategy=random_init_strategy, return_current_strategy=return_current_strategy, alternating_update=alternating_update)
    self.eta = eta
    self.lower_z = lower_z
    self.accumulated_regret = []
    for p in range(self.game.player_num):
      self.accumulated_regret.append(np.ones((self.game.dim_for_player[str(p)],1))/np.sum(np.ones((self.game.dim_for_player[str(p)],1))))

  def evaluate_and_update_policy(self):
    self.iteration = self.iteration + 1
    if not self.alternating_update:
      current_regrets = []
      for i in range(self.game.player_num):
        current_regrets = []
        strategy_q_values = []
      for i in range(self.game.player_num):
        current_regrets.append([])
        strategy_q_values.append([])
      for i in range(self.game.player_num):
        strategy_q_value = - self.game.get_loss_for_player(i, self.current_strategy)
        strategy_value = np.matmul(self.current_strategy[i].T, strategy_q_value)
        current_regrets[i] = strategy_q_value - strategy_value
        strategy_q_values[i] = deepcopy(strategy_q_value)
        if self.RMinmizer=='RM':
          self.accumulated_regret[i] = self.accumulated_regret[i] + current_regrets[i]
        elif self.RMinmizer=='RM+':
          self.accumulated_regret[i] = self.accumulated_regret[i] + self.eta*current_regrets[i]
          if np.sum(np.maximum(self.accumulated_regret[i], np.zeros(self.accumulated_regret[i].shape)))<=self.lower_z:
            self.accumulated_regret[i][:, 0] = projection_simplex_sort(self.accumulated_regret[i][:, 0], z=self.lower_z)
          else:
            self.accumulated_regret[i] = np.maximum(self.accumulated_regret[i], np.zeros(self.accumulated_regret[i].shape))

      for i in range(self.game.player_num):
        self.temp_accumulated_regret = self.accumulated_regret[i] + self.eta*current_regrets[i]
        if np.sum(np.maximum(self.temp_accumulated_regret, np.zeros(self.accumulated_regret[i].shape)))<=self.lower_z:
          self.temp_accumulated_regret[:, 0] = projection_simplex_sort(self.temp_accumulated_regret[:, 0], z=self.lower_z)
        else:
          self.temp_accumulated_regret = np.maximum(self.temp_accumulated_regret, np.zeros(self.accumulated_regret[i].shape))
        self.current_strategy[i] = self.RM(self.temp_accumulated_regret)
        if not self.linear_average:
          self.average_strategy[i] = self.average_strategy[i] + self.current_strategy[i]
        else:
          self.average_strategy[i] = self.average_strategy[i] + self.iteration*self.current_strategy[i]


#SOGRM+
class SOGRMSolver(RMSolver):
  def __init__(self, game, RMinmizer = 'RM', eta=1, random_init_strategy=False, return_current_strategy=False, alternating_update=False, lower_z=1) -> None:
    super().__init__(game=game, RMinmizer = RMinmizer, random_init_strategy=random_init_strategy, return_current_strategy=return_current_strategy, alternating_update=alternating_update)
    self.eta = eta
    self.lower_z = lower_z
    self.accumulated_regret = []
    for p in range(self.game.player_num):
      self.accumulated_regret.append(np.ones((self.game.dim_for_player[str(p)],1))/np.sum(np.ones((self.game.dim_for_player[str(p)],1))))
    self.external_accumulated_regret = deepcopy(self.accumulated_regret)

    self.current_regrets = []
    self.new_regrets = []
    for i in range(self.game.player_num):
      self.current_regrets.append(0)
      self.new_regrets.append(0)

  def evaluate_and_update_policy(self):
    self.iteration = self.iteration + 1
    if not self.alternating_update:
      
      for i in range(self.game.player_num):
        strategy_q_value = - self.game.get_loss_for_player(i, self.current_strategy)
        strategy_value = np.matmul(self.current_strategy[i].T, strategy_q_value)
        self.current_regrets[i] = strategy_q_value - strategy_value
      
        self.external_accumulated_regret[i] = self.accumulated_regret[i] + self.eta*self.current_regrets[i]
        # self.external_accumulated_regret[i][:, 0] = projection_simplex_sort(self.external_accumulated_regret[i][:, 0], z=self.lower_z)
        if np.sum(np.maximum(self.external_accumulated_regret[i], np.zeros(self.accumulated_regret[i].shape)))<=self.lower_z:
          self.external_accumulated_regret[i][:, 0] = projection_simplex_sort(self.external_accumulated_regret[i][:, 0], z=self.lower_z)
        else:
          self.external_accumulated_regret[i] = np.maximum(self.external_accumulated_regret[i], np.zeros(self.accumulated_regret[i].shape))
          
      for i in range(self.game.player_num):
        self.current_strategy[i] = self.RM(self.external_accumulated_regret[i])

        if not self.linear_average:
          self.average_strategy[i] = self.average_strategy[i] + self.current_strategy[i]
        else:
          self.average_strategy[i] = self.average_strategy[i] + self.iteration*self.current_strategy[i]
    
      for i in range(self.game.player_num):
        strategy_q_value = - self.game.get_loss_for_player(i, self.current_strategy)
        strategy_value = np.matmul(self.current_strategy[i].T, strategy_q_value)
        self.new_regrets[i] = strategy_q_value - strategy_value

        self.accumulated_regret[i] = self.external_accumulated_regret[i] - self.eta*self.current_regrets[i] + self.eta*self.new_regrets[i]
            

if __name__ == '__main__':
    seed = 0
    setup_seed(seed)

    dim = 10
    game = TwoPlayerZeroSumGame(dim, dim)

    eta = 0.1
    alternating_update = False
    solvers = [
                SOGRMSolver(game, RMinmizer='RM+', eta=eta, return_current_strategy=True, alternating_update=alternating_update),
                PRMSolver(game, RMinmizer='RM+', return_current_strategy=True, alternating_update=alternating_update),
                RMSolver(game, RMinmizer='RM+', return_current_strategy=True, alternating_update=alternating_update),
            ]

    cexps = []
    for index, alg_name in enumerate(solvers):
        cexps.append([])

    times = []
    running_time = []
    cexp_index = 0
    total_epoch = int(1e5)


    for index, solver in enumerate(solvers):
        print_fq = 100
        select_interval = 100
        time_start = time.time()
        for epoch in range(total_epoch):
            astrategy, cstrategy = solver.output_strategy()
            solver.evaluate_and_update_policy()
            if (epoch+1)%print_fq==0 or epoch==0:
                cexp = game.compute_duality_gap(cstrategy)
                cexps[index].append([cexp])
                if index == 0:
                    times.append(epoch+1)
            if (epoch+1)%print_fq==0 or epoch==0:
                cexp_index = cexp_index + 1
            if (epoch+1)%select_interval==0:
                print_fq = print_fq*10
                select_interval = select_interval*10
        time_end = time.time() 
        running_time.append((time_end - time_start)/60.0)
        print(cexps[index][-1][0])

