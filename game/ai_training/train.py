import numpy as np
from state import GameState
from mcts import MCTS
import json
import os
from datetime import datetime
import pickle
import signal
import sys
import logging
import time
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, simulation_limit=1200, episodes=2000):
        self.simulation_limit = simulation_limit
        self.episodes = episodes
        self.player1 = MCTS(player=1, simulation_limit=simulation_limit)
        self.player2 = MCTS(player=2, simulation_limit=simulation_limit)
        self.stats = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'game_lengths': [],
            'current_episode': 0,
            'player1_policy': {'total_states': 0, 'total_state_visits': 0, 'avg_value': 0},
            'player2_policy': {'total_states': 0, 'total_state_visits': 0, 'avg_value': 0},
            'training_time': 0,
            'avg_move_time': 0
        }
        
        # 设置日志
        self.setup_logging()
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
    def setup_logging(self):
        """设置日志系统"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 修改日志目录路径
        self.log_dir = os.path.join("models", "training_logs", f"training_{self.timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置文件日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建模型保存目录
        self.model_dir = os.path.join(self.log_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 保存训练配置
        config = {
            'simulation_limit': self.simulation_limit,
            'episodes': self.episodes,
            'timestamp': self.timestamp
        }
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
    def handle_interrupt(self, signum, frame):
        """处理中断信号"""
        self.logger.info("\n收到中断信号，正在保存模型和统计信息...")
        self.save_checkpoint()
        sys.exit(0)
        
    def play_game(self):
        """进行一局游戏"""
        state = GameState()
        moves = []
        total_time = 0
        move_count = 0
        
        while not state.is_terminal():
            current_player = self.player1 if state.current_player == 1 else self.player2
            
            # 记录每步用时
            start_time = time.time()
            action = current_player.get_move(state)
            move_time = time.time() - start_time
            
            total_time += move_time
            move_count += 1
            
            moves.append((state.current_player, action))
            state = state.make_move(action)
            
            # 输出每一步的移动
            self.logger.debug(f"玩家{state.current_player}在第{len(moves)}步选择了列{action}，用时{move_time:.3f}秒")
            
        # 记录游戏结果
        winner = state.get_winner()
        if winner == 1:
            self.stats['player1_wins'] += 1
        elif winner == 2:
            self.stats['player2_wins'] += 1
        else:
            self.stats['draws'] += 1
            
        self.stats['game_lengths'].append(len(moves))
        if move_count > 0:
            self.stats['avg_move_time'] = (self.stats['avg_move_time'] * self.stats['current_episode'] + 
                                         total_time / move_count) / (self.stats['current_episode'] + 1)
        
        return moves, winner
        
    def train(self, start_episode=0):
        """训练AI"""
        self.logger.info("开始训练...")
        self.stats['current_episode'] = start_episode
        training_start_time = time.time()
        
        try:
            for episode in range(start_episode, self.episodes):
                self.stats['current_episode'] = episode
                self.logger.info(f"\n开始第 {episode + 1} 局训练")
                
                # 记录训练时间
                episode_start_time = time.time()
                moves, winner = self.play_game()
                episode_time = time.time() - episode_start_time
                
                # 更新总训练时间
                self.stats['training_time'] = time.time() - training_start_time
                
                # 输出本局结果
                if winner:
                    self.logger.info(f"第 {episode + 1} 局结束，玩家{winner}获胜，用时{len(moves)}步，耗时{episode_time:.2f}秒")
                else:
                    self.logger.info(f"第 {episode + 1} 局结束，平局，用时{len(moves)}步，耗时{episode_time:.2f}秒")
                
                # 更新两个AI
                self.player1.update_from_episode(moves)
                self.player2.update_from_episode(moves)
                
                # 每10局保存一次检查点
                if (episode + 1) % 10 == 0:
                    self.save_checkpoint()
                
                # 每100局打印详细统计信息
                if (episode + 1) % 100 == 0:
                    self.log_statistics()
                    
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            self.save_checkpoint()
            raise e
            
    def log_statistics(self):
        """输出详细的统计信息"""
        episode = self.stats['current_episode']
        total_games = episode + 1
        
        # 更新策略统计
        self.stats['player1_policy'] = self.player1.get_policy_stats()
        self.stats['player2_policy'] = self.player2.get_policy_stats()
        
        self.logger.info(f"\n第 {episode + 1} 局训练统计：")
        self.logger.info(f"总训练时间：{self.stats['training_time']:.2f}秒")
        self.logger.info(f"平均每步用时：{self.stats['avg_move_time']:.3f}秒")
        self.logger.info(f"总对局数：{total_games}")
        self.logger.info(f"玩家1胜利：{self.stats['player1_wins']} ({self.stats['player1_wins']/total_games*100:.1f}%)")
        self.logger.info(f"玩家2胜利：{self.stats['player2_wins']} ({self.stats['player2_wins']/total_games*100:.1f}%)")
        self.logger.info(f"平局：{self.stats['draws']} ({self.stats['draws']/total_games*100:.1f}%)")
        self.logger.info(f"平均对局长度：{np.mean(self.stats['game_lengths']):.2f}")
        
        # 输出策略统计
        self.logger.info("\n策略统计：")
        self.logger.info(f"玩家1已学习状态数：{self.stats['player1_policy']['total_states']}")
        self.logger.info(f"玩家1状态访问总次数：{self.stats['player1_policy']['total_state_visits']}")
        self.logger.info(f"玩家1平均状态值：{self.stats['player1_policy']['avg_value']:.3f}")
        self.logger.info(f"玩家2已学习状态数：{self.stats['player2_policy']['total_states']}")
        self.logger.info(f"玩家2状态访问总次数：{self.stats['player2_policy']['total_state_visits']}")
        self.logger.info(f"玩家2平均状态值：{self.stats['player2_policy']['avg_value']:.3f}")
        
        # 保存统计信息到JSON文件
        stats_file = os.path.join(self.log_dir, f"stats_episode_{episode+1}.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
            
        # 保存训练曲线图
        self.plot_training_curves()
            
    def save_checkpoint(self):
        """保存检查点（包括模型状态和训练统计）"""
        episode = self.stats['current_episode']
        checkpoint = {
            'episode': episode,
            'stats': self.stats,
            'player1': self.player1,
            'player2': self.player2,
            'timestamp': self.timestamp
        }
        
        # 保存到模型目录
        checkpoint_file = os.path.join(self.model_dir, f"checkpoint_episode_{episode+1}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.logger.info(f"检查点已保存到：{checkpoint_file}")
        
        # 同时保存最新的模型
        latest_file = os.path.join(self.model_dir, "latest_model.pkl")
        with open(latest_file, 'wb') as f:
            pickle.dump(checkpoint, f)
            
    def load_checkpoint(self, checkpoint_file):
        """加载检查点"""
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            
        self.stats = checkpoint['stats']
        self.player1 = checkpoint['player1']
        self.player2 = checkpoint['player2']
        start_episode = checkpoint['episode'] + 1
        
        self.logger.info(f"已加载检查点，将从第 {start_episode + 1} 局继续训练")
        return start_episode

    def plot_training_curves(self):
        """绘制训练曲线"""
        episode = self.stats['current_episode']
        
        # 创建图表目录
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 胜率曲线
        plt.figure(figsize=(12, 6))
        total_games = episode + 1
        win_rate1 = self.stats['player1_wins'] / total_games * 100
        win_rate2 = self.stats['player2_wins'] / total_games * 100
        draw_rate = self.stats['draws'] / total_games * 100
        
        plt.bar(['玩家1胜率', '玩家2胜率', '平局率'], [win_rate1, win_rate2, draw_rate])
        plt.title('胜率统计')
        plt.ylabel('百分比 (%)')
        plt.savefig(os.path.join(plots_dir, f'win_rates_episode_{episode+1}.png'))
        plt.close()
        
        # 状态值分布
        plt.figure(figsize=(12, 6))
        values1 = [value for actions in self.player1.policy_values.values() for value in actions.values()]
        values2 = [value for actions in self.player2.policy_values.values() for value in actions.values()]
        
        plt.hist(values1, bins=50, alpha=0.5, label='玩家1')
        plt.hist(values2, bins=50, alpha=0.5, label='玩家2')
        plt.title('状态值分布')
        plt.xlabel('状态值')
        plt.ylabel('频率')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'value_distribution_episode_{episode+1}.png'))
        plt.close()
        
        # 对局长度变化
        plt.figure(figsize=(12, 6))
        plt.plot(self.stats['game_lengths'])
        plt.title('对局长度变化')
        plt.xlabel('对局编号')
        plt.ylabel('步数')
        plt.savefig(os.path.join(plots_dir, f'game_lengths_episode_{episode+1}.png'))
        plt.close()
        
        # 每步平均用时变化
        plt.figure(figsize=(12, 6))
        plt.plot([self.stats['avg_move_time']])
        plt.title('每步平均用时')
        plt.xlabel('训练进度')
        plt.ylabel('时间(秒)')
        plt.savefig(os.path.join(plots_dir, f'avg_move_time_episode_{episode+1}.png'))
        plt.close()

def main():
    # 训练参数
    simulation_limit = 1200   # 增加每步模拟次数
    episodes = 2000          # 增加训练局数
    n_jobs = 8              # 增加并行进程数
    
    # 创建训练器
    trainer = Trainer(simulation_limit=simulation_limit, episodes=episodes)
    
    # 检查是否有检查点文件
    checkpoint_file = None
    if len(sys.argv) > 1:
        checkpoint_file = sys.argv[1]
        
    # 开始训练
    start_episode = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        start_episode = trainer.load_checkpoint(checkpoint_file)
    
    trainer.train(start_episode)
    
if __name__ == "__main__":
    main() 