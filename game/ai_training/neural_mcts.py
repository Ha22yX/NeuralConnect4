import numpy as np
import torch
import torch.optim as optim
from collections import deque, defaultdict
import random
from neural_net import ConnectFourNet
from state import GameState
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
import logging
import psutil
import torch.multiprocessing as mp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0
        self.state_hash = state.get_state_hash()
        
    def expanded(self):
        return len(self.children) > 0
        
    def get_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class NeuralMCTS:
    def __init__(self, net, num_simulations=800, c_puct=1.0, num_threads=24):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.value_cache = {}
        self.policy_cache = {}
        self.cache_lock = Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def get_action_prob(self, state, temp=1.0):
        root = Node(state)
        futures = []
        
        # 并行运行MCTS模拟
        for _ in range(self.num_simulations):
            futures.append(self.executor.submit(self._run_simulation, root))
            
        # 等待所有模拟完成
        for future in futures:
            future.result()
            
        # 计算动作概率
        action_probs = np.zeros(7)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
            
        # 确保至少有一个有效的访问计数
        if np.sum(action_probs) > 0:
            action_probs = action_probs / np.sum(action_probs)
        else:
            valid_moves = state.get_valid_moves()
            action_probs = valid_moves / np.sum(valid_moves)
            
        # 根据温度参数调整概率分布
        if temp == 0:
            best_action = np.argmax(action_probs)
            action_probs = np.zeros_like(action_probs)
            action_probs[best_action] = 1
        else:
            action_probs = action_probs ** (1/temp)
            action_probs = action_probs / np.sum(action_probs)
            
        return action_probs
        
    def _run_simulation(self, root):
        """运行单次MCTS模拟"""
        node = root
        search_path = [node]
        
        # 选择
        while node.expanded():
            action, node = self.select_child(node)
            search_path.append(node)
            
        # 扩展和评估
        value = self.evaluate_and_expand(node)
        
        # 反向传播
        self.backpropagate(search_path, value, node.state.current_player)
        
    def select_child(self, node):
        """选择最有价值的子节点"""
        max_ucb = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            ucb = self.get_ucb(node, child)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action
                best_child = child
                
        return best_action, best_child

    def get_ucb(self, parent, child):
        """计算UCB值"""
        # 增加防守意识：当对手可能获胜时，提高防守动作的权重
        prior_score = self.c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.get_value() if parent.state.current_player != child.state.current_player else child.get_value()
        
        # 如果是防守动作（即可以阻止对手连成四子），增加其权重
        if self._is_defensive_move(parent.state, child.action):
            prior_score *= 1.5
            
        return value_score + prior_score
        
    def _is_defensive_move(self, state, action):
        """判断一个动作是否是防守动作"""
        # 模拟对手在这个位置下子
        opponent = 3 - state.current_player
        test_state = state.make_move(action)
        if test_state is None:
            return False
            
        # 检查如果对手在这里下子是否会赢
        test_board = state.board.copy()
        for row in range(5, -1, -1):
            if test_board[row][action] == 0:
                test_board[row][action] = opponent
                # 创建临时状态检查是否获胜
                temp_state = GameState(test_board, opponent)
                if temp_state.get_winner() == opponent:
                    return True
                break
        return False

    def evaluate_and_expand(self, node):
        """评估并扩展节点"""
        state = node.state
        
        if state.is_terminal():
            return 1 if state.get_winner() == state.current_player else -1
            
        state_hash = node.state_hash
        with self.cache_lock:
            if state_hash in self.value_cache:
                self.cache_hits += 1
                value = self.value_cache[state_hash]
                policy = self.policy_cache[state_hash]
            else:
                self.cache_misses += 1
                policy, value = self.net.predict(state.get_canonical_board())
                self.value_cache[state_hash] = value
                self.policy_cache[state_hash] = policy
                
        valid_moves = state.get_valid_moves()
        policy = policy * valid_moves
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = valid_moves / np.sum(valid_moves)
            
        for action in range(7):
            if valid_moves[action]:
                next_state = state.make_move(action)
                child = Node(next_state, parent=node, action=action)
                child.prior = policy[action]
                node.children[action] = child
                
        return value

    def backpropagate(self, search_path, value, original_player):
        """反向传播更新统计信息"""
        for node in reversed(search_path):
            node.value_sum += value if node.state.current_player == original_player else -value
            node.visit_count += 1

class NeuralMCTSTrainer:
    def __init__(self, batch_size=512, num_simulations=800, device='cpu', num_threads=24):
        self.device = torch.device(device)
        self.net = ConnectFourNet().to(self.device)
        
        # 使用学习率调度器
        self.base_lr = 0.001
        self.optimizer = optim.Adam(
            self.net.parameters(), 
            lr=self.base_lr,
            weight_decay=1e-4,
            eps=1e-7,
            amsgrad=True
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # 每5轮重启一次
            T_mult=2,  # 每次重启后周期翻倍
            eta_min=self.base_lr/10  # 最小学习率
        )
        
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        # 增加优先经验回放
        self.buffer = []
        self.max_buffer_size = 100000
        self.min_buffer_size = batch_size * 10
        self.mcts = NeuralMCTS(self.net, num_simulations=num_simulations, num_threads=num_threads)
        
        # 损失函数权重
        self.value_loss_weight = 1.0
        self.policy_loss_weight = 1.0
        
        self.training_stats = {
            'iterations': 0,
            'total_games': 0,
            'total_moves': 0,
            'avg_game_length': 0,
            'win_rates': defaultdict(float),
            'training_time': 0,
            'loss_history': [],
            'value_loss_history': [],
            'policy_loss_history': [],
            'avg_move_time_history': [],
            'cache_hit_rates': [],
            'learning_rates': []
        }
        
        # 设置CPU亲和性
        self.setup_cpu_affinity()
        
    def setup_cpu_affinity(self):
        """设置CPU亲和性以优化9950X的性能"""
        try:
            # 获取所有CPU核心
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count >= 48:  # 9950X有48个逻辑核心
                # 设置进程亲和性，避免使用SMT对核心
                p = psutil.Process()
                # 使用物理核心（0-23）而不是逻辑核心
                p.cpu_affinity(list(range(24)))
                logger.info(f"CPU亲和性已设置为物理核心: {p.cpu_affinity()}")
        except Exception as e:
            logger.warning(f"设置CPU亲和性时出错: {e}")
            
    def _play_single_game_worker(self):
        """工作进程的游戏函数"""
        mcts = NeuralMCTS(self.net, num_simulations=self.num_simulations)
        state = GameState()
        game_history = []
        moves = 0
        game_stats = {
            'move_times': [],
        }
        
        # 动态调整温度参数
        base_temp = 1.0
        temp_decay = 0.97
        
        while not state.is_terminal():
            start_time = time.time()
            
            # 根据游戏进程动态调整温度
            current_temp = base_temp * (temp_decay ** (moves // 2))
            
            # 获取当前玩家的动作概率
            pi = mcts.get_action_prob(state, temp=current_temp)
            game_history.append([state.get_canonical_board(), pi, None])
            
            # 选择动作（增加探索和防守意识）
            if moves < 10:  # 开局阶段增加随机性
                action = np.random.choice(len(pi), p=pi)
            else:
                # 检查是否有防守需求
                defensive_moves = []
                for action in range(7):
                    if pi[action] > 0 and mcts._is_defensive_move(state, action):
                        defensive_moves.append(action)
                
                if defensive_moves and np.random.random() < 0.9:  # 90%概率选择防守
                    action = np.random.choice(defensive_moves)
                else:
                    # 后期更倾向于选择最优动作，但保留一定随机性
                    if np.random.random() < 0.7:  # 降低最优动作的选择概率
                        action = np.argmax(pi)
                    else:
                        action = np.random.choice(len(pi), p=pi)
            
            move_time = time.time() - start_time
            game_stats['move_times'].append(move_time)
            
            state = state.make_move(action)
            moves += 1
            
        winner = state.get_winner()
        value = 0 if winner == 0 else (1 if winner == 1 else -1)
        game_stats['winner'] = winner
        game_stats['total_moves'] = moves
        game_stats['avg_move_time'] = np.mean(game_stats['move_times'])
        
        # 计算游戏质量分数（增加对平局的重视）
        quality_score = self._calculate_game_quality(moves, winner)
        
        # 更新训练数据
        for hist in game_history:
            hist[2] = value
            hist.append(quality_score)
            value = -value
            
        return game_history, moves, game_stats
        
    def _calculate_game_quality(self, moves, winner):
        """计算游戏质量分数用于优先经验回放"""
        quality = 1.0
        
        if winner != 0:  # 非平局
            # 降低快速获胜的权重，鼓励更长的对局
            quality *= max(0.6, 1.0 - (moves / 42))
        else:  # 平局
            # 提高平局的权重，特别是回合数较多的平局
            quality *= min(1.2, 0.8 + (moves / 42))
            
        return quality
        
    def _update_buffer(self, game_history):
        """更新经验回放缓冲区"""
        for experience in game_history:
            if len(self.buffer) >= self.max_buffer_size:
                # 随机移除低质量的经验
                indices = np.random.choice(len(self.buffer), size=len(game_history), p=self._get_removal_probs())
                for idx in sorted(indices, reverse=True):
                    self.buffer.pop(idx)
            self.buffer.append(experience)
            
    def _get_removal_probs(self):
        """计算经验移除概率（优先保留高质量经验）"""
        qualities = np.array([exp[3] for exp in self.buffer])
        probs = 1.0 / (qualities + 1e-6)
        return probs / np.sum(probs)
        
    def _sample_batch(self):
        """采样训练批次（优先采样高质量经验）"""
        if len(self.buffer) < self.min_buffer_size:
            return None
            
        qualities = np.array([exp[3] for exp in self.buffer])
        probs = qualities / np.sum(qualities)
        
        indices = np.random.choice(len(self.buffer), size=self.batch_size, p=probs)
        return [self.buffer[idx] for idx in indices]
        
    def train_step(self):
        """执行一次训练步骤"""
        batch = self._sample_batch()
        if batch is None:
            return None
            
        start_time = time.time()
        
        # 准备批次数据
        state_batch = torch.stack([
            torch.FloatTensor(item[0]).unsqueeze(0)
            for item in batch
        ]).to(self.device)
        pi_batch = torch.FloatTensor([item[1] for item in batch]).to(self.device)
        v_batch = torch.FloatTensor([item[2] for item in batch]).to(self.device)
        
        self.net.train()
        pi_pred, v_pred = self.net(state_batch)
        
        # 计算加权损失
        value_loss = self.value_loss_weight * torch.mean((v_batch - v_pred.squeeze()) ** 2)
        policy_loss = self.policy_loss_weight * -torch.mean(torch.sum(pi_batch * torch.log(pi_pred + 1e-8), dim=1))
        total_loss = value_loss + policy_loss
        
        # 梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        self.training_stats['learning_rates'].append(current_lr)
        
        # 动态调整损失权重
        if value_loss.item() > policy_loss.item() * 2:
            self.value_loss_weight *= 0.95
            self.policy_loss_weight *= 1.05
        elif policy_loss.item() > value_loss.item() * 2:
            self.value_loss_weight *= 1.05
            self.policy_loss_weight *= 0.95
            
        train_time = time.time() - start_time
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'train_time': train_time,
            'batch_size': self.batch_size,
            'learning_rate': current_lr
        }
        
    def self_play(self, num_games=100):
        """自我对弈收集训练数据"""
        start_time = time.time()
        games = []
        total_moves = 0
        game_statistics = []
        
        # 使用线程池而不是进程池
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = [executor.submit(self._play_single_game_worker) for _ in range(num_games)]
            for i, future in enumerate(futures, 1):
                try:
                    game_data, moves, game_stats = future.result()
                    games.extend(game_data)
                    total_moves += moves
                    game_statistics.append(game_stats)
                    
                    elapsed = time.time() - start_time
                    games_per_second = i / elapsed
                    estimated_remaining = (num_games - i) / games_per_second
                    
                    # 计算当前的胜率统计
                    current_stats = self._calculate_game_stats(game_statistics)
                    total_games = self.training_stats['total_games'] + i
                    
                    logger.info(f"\n完成第 {i}/{num_games} 局自我对弈 (总计: {total_games}局)")
                    logger.info(f"本局信息:")
                    logger.info(f"├─ 获胜者: {'平局' if game_stats['winner'] == 0 else f'玩家{game_stats['winner']}'}")
                    logger.info(f"├─ 总步数: {game_stats['total_moves']}")
                    logger.info(f"└─ 平均每步用时: {game_stats['avg_move_time']*1000:.2f}毫秒")
                    
                    logger.info(f"\n当前训练进度:")
                    logger.info(f"├─ 用时: {elapsed:.1f}秒")
                    logger.info(f"├─ 速度: {games_per_second:.2f}局/秒")
                    logger.info(f"└─ 预计剩余时间: {estimated_remaining:.1f}秒")
                    
                    logger.info(f"\n累计统计:")
                    logger.info(f"├─ 玩家1胜率: {current_stats['player1_win_rate']:.1f}%")
                    logger.info(f"├─ 玩家2胜率: {current_stats['player2_win_rate']:.1f}%")
                    logger.info(f"├─ 平局率: {current_stats['draw_rate']:.1f}%")
                    logger.info(f"└─ 平均每局步数: {current_stats['avg_game_length']:.1f}")
                    
                except Exception as e:
                    logger.error(f"游戏 {i} 发生错误: {str(e)}")
                    
        # 更新训练统计信息
        final_stats = self._calculate_game_stats(game_statistics)
        self.training_stats.update(final_stats)
        self.training_stats['total_games'] += num_games
        self.training_stats['total_moves'] += total_moves
        self.training_stats['avg_game_length'] = total_moves / num_games
        
        self._update_buffer(games)
        
    def _calculate_game_stats(self, game_statistics):
        """计算游戏统计信息"""
        total_games = len(game_statistics)
        if total_games == 0:
            return {
                'player1_win_rate': 0,
                'player2_win_rate': 0,
                'draw_rate': 0,
                'avg_game_length': 0,
                'avg_move_time': 0
            }
            
        player1_wins = sum(1 for stats in game_statistics if stats['winner'] == 1)
        player2_wins = sum(1 for stats in game_statistics if stats['winner'] == 2)
        draws = sum(1 for stats in game_statistics if stats['winner'] == 0)
        
        total_moves = sum(stats['total_moves'] for stats in game_statistics)
        total_move_times = sum(stats['avg_move_time'] for stats in game_statistics)
        
        return {
            'player1_win_rate': (player1_wins / total_games) * 100,
            'player2_win_rate': (player2_wins / total_games) * 100,
            'draw_rate': (draws / total_games) * 100,
            'avg_game_length': total_moves / total_games,
            'avg_move_time': total_move_times / total_games
        }
        
    def train(self, num_iterations=50, games_per_iteration=100, epochs_per_iteration=5):
        """完整的训练循环"""
        total_start_time = time.time()
        
        for iteration in range(num_iterations):
            iteration_start_time = time.time()
            logger.info(f"\n{'='*50}")
            logger.info(f"开始第 {iteration + 1}/{num_iterations} 轮训练迭代")
            logger.info(f"{'='*50}")
            
            # 自我对弈
            logger.info("\n[自我对弈阶段]")
            self.self_play(games_per_iteration)
            self_play_time = time.time() - iteration_start_time
            
            # 获取并记录缓存统计
            cache_stats = self.mcts.get_cache_stats()
            self.training_stats['cache_hit_rates'].append(cache_stats['hit_rate'])
            
            logger.info(f"\n[缓存统计]")
            logger.info(f"缓存命中率: {cache_stats['hit_rate']*100:.2f}%")
            logger.info(f"缓存命中次数: {cache_stats['hits']}")
            logger.info(f"缓存未命中次数: {cache_stats['misses']}")
            
            # 训练网络
            logger.info("\n[网络训练阶段]")
            epoch_losses = []
            epoch_value_losses = []
            epoch_policy_losses = []
            
            for epoch in range(epochs_per_iteration):
                stats = self.train_step()
                if stats:
                    epoch_losses.append(stats['total_loss'])
                    epoch_value_losses.append(stats['value_loss'])
                    epoch_policy_losses.append(stats['policy_loss'])
                    
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs_per_iteration} - "
                        f"损失: {stats['total_loss']:.4f} "
                        f"(价值损失: {stats['value_loss']:.4f}, "
                        f"策略损失: {stats['policy_loss']:.4f})"
                    )
            
            # 计算并记录平均损失
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_value_loss = np.mean(epoch_value_losses)
                avg_policy_loss = np.mean(epoch_policy_losses)
                self.training_stats['loss_history'].append(avg_loss)
                self.training_stats['value_loss_history'].append(avg_value_loss)
                self.training_stats['policy_loss_history'].append(avg_policy_loss)
            
            # 计算并记录统计信息
            iteration_time = time.time() - iteration_start_time
            total_time = time.time() - total_start_time
            avg_move_time = self_play_time / (games_per_iteration * self.training_stats['avg_game_length'])
            self.training_stats['avg_move_time_history'].append(avg_move_time)
            
            logger.info(f"\n[迭代统计]")
            logger.info(f"自我对弈阶段用时: {self_play_time:.2f}秒")
            logger.info(f"训练阶段用时: {iteration_time - self_play_time:.2f}秒")
            logger.info(f"总用时: {iteration_time:.2f}秒")
            logger.info(f"平均每局游戏长度: {self.training_stats['avg_game_length']:.1f}步")
            logger.info(f"平均每步用时: {avg_move_time*1000:.2f}毫秒")
            
            if epoch_losses:
                logger.info(f"\n[损失统计]")
                logger.info(f"平均总损失: {avg_loss:.4f}")
                logger.info(f"平均价值损失: {avg_value_loss:.4f}")
                logger.info(f"平均策略损失: {avg_policy_loss:.4f}")
            
            # 保存模型和训练状态
            if (iteration + 1) % 5 == 0:  # 每5轮保存一次
                self.save_model(f"model_iteration_{iteration + 1}.pth", {
                    'iteration': iteration + 1,
                    'training_stats': self.training_stats,
                    'train_time': time.time() - total_start_time,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                logger.info(f"\n[保存检查点] 模型已保存至 model_iteration_{iteration + 1}.pth")
            
            logger.info(f"\n[训练进度] 完成度: {(iteration + 1) / num_iterations * 100:.1f}%")
            logger.info(f"剩余预计时间: {(total_time / (iteration + 1)) * (num_iterations - iteration - 1):.1f}秒")
            logger.info(f"{'='*50}\n")
        
    def save_model(self, filename, stats):
        """保存模型"""
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': stats
        }, os.path.join('models', filename))
        
    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(os.path.join('models', filename))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 