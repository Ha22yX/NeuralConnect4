import numpy as np

class Board:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.winner = None

    def is_valid_move(self, col):
        """检查在指定列落子是否合法"""
        return 0 <= col < self.cols and self.board[0][col] == 0

    def get_valid_moves(self):
        """获取所有合法的落子位置"""
        return [col for col in range(self.cols) if self.is_valid_move(col)]

    def make_move(self, col):
        """在指定列落子"""
        if not self.is_valid_move(col):
            return False

        # 从底部向上查找第一个空位
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)
                self.check_winner(row, col)
                self.current_player = 3 - self.current_player  # 切换玩家（1->2或2->1）
                return True
        return False

    def check_winner(self, row, col):
        """检查是否有玩家获胜"""
        player = self.board[row][col]
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
                    if (not (0 <= r < self.rows and 0 <= c < self.cols) or 
                        self.board[r][c] != player):
                        break
                    count += 1
            if count >= 4:
                self.winner = player
                return

    def is_full(self):
        """检查棋盘是否已满"""
        return len(self.get_valid_moves()) == 0

    def get_state(self):
        """获取当前棋盘状态"""
        return self.board.copy()

    def get_winner(self):
        """获取获胜者"""
        return self.winner

    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.winner = None 