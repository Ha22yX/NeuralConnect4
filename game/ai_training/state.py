import numpy as np

class GameState:
    def __init__(self, board=None, current_player=1):
        """初始化游戏状态"""
        if board is None:
            self.board = np.zeros((6, 7), dtype=np.int32)
        else:
            self.board = np.array(board, dtype=np.int32)
        self.current_player = current_player
        self.last_move = None
        
    def get_valid_moves(self):
        """获取所有合法移动"""
        valid_moves = np.zeros(7, dtype=np.int32)
        for col in range(7):
            if self.board[0][col] == 0:
                valid_moves[col] = 1
        return valid_moves
        
    def make_move(self, col):
        """执行移动并返回新的状态"""
        valid_moves = self.get_valid_moves()
        if valid_moves[col] == 0:
            return None
        
        new_state = GameState(self.board.copy(), 3 - self.current_player)
        
        # 从底部向上查找第一个空位
        for row in range(5, -1, -1):
            if new_state.board[row][col] == 0:
                new_state.board[row][col] = self.current_player
                new_state.last_move = (row, col)
                break
                
        return new_state
        
    def is_terminal(self):
        """检查游戏是否结束"""
        if self.get_winner() is not None:
            return True
        return len(self.get_valid_moves()) == 0
        
    def get_winner(self):
        """获取获胜者"""
        if self.last_move is None:
            return None
            
        row, col = self.last_move
        player = self.board[row][col]
        
        # 检查四个方向
        directions = [
            [(0, 1), (0, -1)],  # 水平
            [(1, 0), (-1, 0)],  # 垂直
            [(1, 1), (-1, -1)], # 对角线
            [(1, -1), (-1, 1)]  # 反对角线
        ]
        
        for dir_pair in directions:
            count = 1
            for direction in dir_pair:
                r, c = row, col
                dr, dc = direction
                while True:
                    r, c = r + dr, c + dc
                    if not (0 <= r < 6 and 0 <= c < 7):
                        break
                    if self.board[r][c] != player:
                        break
                    count += 1
            if count >= 4:
                return player
        return None
        
    def get_reward(self, player):
        """获取状态的奖励值"""
        winner = self.get_winner()
        if winner is None:
            return self._evaluate_position(player)
        elif winner == player:
            return 1.0
        else:
            return -1.0
            
    def _evaluate_position(self, player):
        """评估当前局面对指定玩家的有利程度"""
        score = 0
        
        # 评估所有可能的连线
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == 0:
                    continue
                    
                # 水平方向
                if col <= 3:
                    line = self.board[row, col:col+4]
                    score += self._evaluate_line(line, player)
                
                # 垂直方向
                if row <= 2:
                    line = self.board[row:row+4, col]
                    score += self._evaluate_line(line, player)
                
                # 对角线（右下）
                if row <= 2 and col <= 3:
                    line = np.array([self.board[row+i][col+i] for i in range(4)])
                    score += self._evaluate_line(line, player)
                
                # 对角线（左下）
                if row <= 2 and col >= 3:
                    line = np.array([self.board[row+i][col-i] for i in range(4)])
                    score += self._evaluate_line(line, player)
        
        return score / 100.0  # 归一化到[-1, 1]范围
        
    def _evaluate_line(self, line, player):
        """评估一条线的分数"""
        opponent = 3 - player
        player_count = np.sum(line == player)
        opponent_count = np.sum(line == opponent)
        empty_count = np.sum(line == 0)
        
        # 如果同时包含双方的棋子，这条线没有价值
        if player_count > 0 and opponent_count > 0:
            return 0
            
        # 根据己方棋子数量评分
        if player_count > 0:
            if player_count == 4:
                return 100  # 连成4个
            elif player_count == 3:
                return 5 if empty_count == 1 else 0  # 3个连续 + 1个空位
            elif player_count == 2:
                return 2 if empty_count == 2 else 0  # 2个连续 + 2个空位
            return 1  # 1个棋子
            
        # 根据对方棋子数量评分（防守）
        if opponent_count > 0:
            if opponent_count == 3 and empty_count == 1:
                return -80  # 对手差一步就赢了
            elif opponent_count == 2 and empty_count == 2:
                return -3  # 对手有潜在威胁
        
        return 0
        
    def get_state_hash(self):
        """获取状态的唯一标识"""
        return hash(str(self.board.tobytes()) + str(self.current_player))
        
    def get_canonical_board(self):
        """获取规范化的棋盘状态（从当前玩家的视角）"""
        canonical_board = self.board.copy()
        if self.current_player == 2:
            # 如果是玩家2，交换1和2的表示
            canonical_board[canonical_board == 1] = 3  # 临时值
            canonical_board[canonical_board == 2] = 1
            canonical_board[canonical_board == 3] = 2
        # 确保返回形状为 [height, width]
        return canonical_board.reshape(6, 7)
        
    def get_valid_moves_mask(self):
        """获取有效移动的掩码"""
        mask = np.zeros(7, dtype=np.int32)
        valid_moves = self.get_valid_moves()
        mask[valid_moves] = 1
        return mask
        
    def __str__(self):
        """返回棋盘的字符串表示"""
        return str(self.board) 