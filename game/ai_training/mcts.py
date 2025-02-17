import numpy as np
import math
from collections import defaultdict
from joblib import Parallel, delayed
from state import GameState

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state.get_valid_moves()
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
        
    def best_child(self, c_param=1.414):
        choices = [(child,
                   child.value / (child.visits + 1e-10) + c_param * math.sqrt(2 * math.log(self.visits + 1) / (child.visits + 1e-10)))
                  for action, child in self.children.items()]
        return max(choices, key=lambda x: x[1])[0]
        
    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.make_move(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node
        
    def update(self, result):
        self.visits += 1
        self.value += result

def parallel_simulate(state, player):
    """并行模拟一局游戏"""
    current_state = state
    while not current_state.is_terminal():
        valid_moves = current_state.get_valid_moves()
        action = np.random.choice(valid_moves)
        current_state = current_state.make_move(action)
        
    winner = current_state.get_winner()
    if winner is None:
        return 0.0
    elif winner == player:
        return 1.0
    else:
        return -1.0

class MCTS:
    def __init__(self, player, simulation_limit=1000, n_jobs=4):
        self.player = player
        self.simulation_limit = simulation_limit
        self.n_jobs = n_jobs
        self.policy_values = {}  # 状态-动作值函数
        self.state_visits = {}  # 状态访问次数
        self.learning_rate = 0.2  # 提高学习率以加快收敛
        self.discount_factor = 0.99  # 增加折扣因子以更重视长期收益
        self.temperature = 1.0  # 用于控制探索程度
        self.min_visits = 5  # 最小访问次数阈值
        
    def get_move(self, state):
        root = MCTSNode(state)
        state_hash = state.get_state_hash()
        
        # 使用已学习的知识初始化节点值，并添加探索噪声
        if state_hash in self.policy_values:
            for action, value in self.policy_values[state_hash].items():
                if action in root.children:
                    noise = np.random.normal(0, 0.1)  # 添加高斯噪声
                    root.children[action].value = (value + noise) * root.children[action].visits
        
        # 将模拟分成多个批次
        batch_size = self.simulation_limit // self.n_jobs
        n_batches = self.n_jobs
        
        for _ in range(n_batches):
            nodes_to_simulate = []
            
            # 为每个批次选择节点
            for _ in range(batch_size):
                node = root
                # 选择
                while not node.state.is_terminal() and node.is_fully_expanded():
                    node = node.best_child(c_param=1.414 * self.temperature)
                    
                # 扩展
                if not node.state.is_terminal() and not node.is_fully_expanded():
                    node = node.expand()
                    
                nodes_to_simulate.append(node)
            
            # 并行执行模拟
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(parallel_simulate)(node.state, self.player)
                for node in nodes_to_simulate
            )
            
            # 更新结果
            for node, result in zip(nodes_to_simulate, results):
                while node is not None:
                    node.update(result)
                    node = node.parent
        
        # 根据访问次数和价值选择最佳移动
        valid_children = [(action, child) for action, child in root.children.items() 
                         if child.visits >= self.min_visits]
        
        if not valid_children:  # 如果没有足够访问次数的子节点，使用所有子节点
            valid_children = list(root.children.items())
            
        # 使用softmax选择动作
        values = np.array([child.value / (child.visits + 1e-10) for _, child in valid_children])
        visits = np.array([child.visits for _, child in valid_children])
        
        # 结合价值和访问次数的评分
        scores = values + 0.1 * np.log(visits + 1)
        scores = np.exp(scores / self.temperature)
        probs = scores / np.sum(scores)
        
        # 根据概率选择动作
        chosen_idx = np.random.choice(len(valid_children), p=probs)
        best_action = valid_children[chosen_idx][0]
        best_child = valid_children[chosen_idx][1]
        
        # 更新策略值
        self.update_policy(state_hash, best_action, best_child.value / best_child.visits)
        
        # 逐渐降低温度参数
        self.temperature = max(0.1, self.temperature * 0.999)
        
        return best_action
        
    def update_policy(self, state_hash, action, value):
        """更新策略值"""
        if state_hash not in self.policy_values:
            self.policy_values[state_hash] = {}
        if state_hash not in self.state_visits:
            self.state_visits[state_hash] = 0
            
        current_value = self.policy_values[state_hash].get(action, 0.0)
        visit_count = self.state_visits[state_hash]
        
        # 使用动态学习率
        effective_lr = self.learning_rate / (1 + visit_count * 0.01)
        
        # 使用时序差分学习更新值
        new_value = current_value + effective_lr * (value - current_value)
        self.policy_values[state_hash][action] = new_value
        self.state_visits[state_hash] += 1
        
    def update_from_episode(self, episode):
        """从完整的对局中学习"""
        rewards = []
        states = []
        actions = []
        
        # 收集对局信息
        current_state = GameState()
        for player, action in episode:
            if player == self.player:
                states.append(current_state.get_state_hash())
                actions.append(action)
                # 计算即时奖励
                next_state = current_state.make_move(action)
                reward = next_state.get_reward(self.player)
                rewards.append(reward)
            current_state = current_state.make_move(action)
            
        # 计算最终奖励
        final_reward = current_state.get_reward(self.player)
        rewards.append(final_reward)
        
        # 使用n步时序差分学习
        n_step = 3
        for i in range(len(states)):
            # 计算n步回报
            n_step_return = 0
            for j in range(n_step):
                if i + j < len(rewards):
                    n_step_return += (self.discount_factor ** j) * rewards[i + j]
                    
            # 更新策略
            self.update_policy(states[i], actions[i], n_step_return)
        
    def get_policy_stats(self):
        """获取策略统计信息"""
        if not self.policy_values:
            return {
                'total_states': 0,
                'total_state_visits': 0,
                'avg_value': 0
            }
            
        return {
            'total_states': len(self.policy_values),
            'total_state_visits': sum(self.state_visits.values()),
            'avg_value': np.mean([
                np.mean(list(actions.values()))
                for actions in self.policy_values.values()
            ])
        }
        
    def simulate(self, state):
        """随机模拟到游戏结束（单次模拟）"""
        return parallel_simulate(state, self.player) 