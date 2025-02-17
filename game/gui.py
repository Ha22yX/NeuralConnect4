import pygame
import numpy as np
import math

class GUI:
    # 颜色定义
    BLUE = (0, 102, 204)
    DARK_BLUE = (0, 51, 153)
    BLACK = (0, 0, 0)
    RED = (255, 51, 51)
    YELLOW = (255, 204, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)

    def __init__(self, board, square_size=100):
        self.board = board
        self.square_size = square_size
        self.width = board.cols * square_size
        self.height = (board.rows + 1) * square_size
        self.radius = int(square_size/2 - 5)
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('四连棋')
        
        # 初始化字体
        self.font = pygame.font.Font(None, 48)
        
        # 动画相关
        self.falling_piece = None
        self.falling_y = 0
        self.target_y = 0
        self.animation_speed = 20
        
        # 悬停效果
        self.hover_col = None
        
        # 获胜动画
        self.winning_pieces = []
        self.winning_animation_frame = 0

    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(self.DARK_BLUE)
        
        # 绘制当前玩家提示
        color = self.RED if self.board.current_player == 1 else self.YELLOW
        text = self.font.render(f"玩家 {self.board.current_player} 回合", True, color)
        text_rect = text.get_rect(center=(self.width/2, self.square_size/2))
        self.screen.blit(text, text_rect)
        
        # 绘制悬停效果
        if self.hover_col is not None and self.board.is_valid_move(self.hover_col):
            x = self.hover_col * self.square_size + self.square_size//2
            y = self.square_size//2
            color = self.RED if self.board.current_player == 1 else self.YELLOW
            pygame.draw.circle(self.screen, color, (x, y), self.radius//2)
        
        # 绘制棋盘网格和棋子
        for c in range(self.board.cols):
            for r in range(self.board.rows):
                x = c * self.square_size + self.square_size//2
                y = (r + 1) * self.square_size + self.square_size//2
                
                # 绘制背景圆
                pygame.draw.circle(self.screen, self.BLUE, (x, y), self.radius)
                pygame.draw.circle(self.screen, self.DARK_BLUE, (x, y), self.radius, 3)
                
                # 绘制棋子
                piece = self.board.board[r][c]
                if piece > 0:
                    color = self.RED if piece == 1 else self.YELLOW
                    if (r, c) in self.winning_pieces:
                        # 获胜棋子的动画效果
                        scale = 1 + 0.1 * math.sin(self.winning_animation_frame * 0.1)
                        scaled_radius = int(self.radius * scale)
                        pygame.draw.circle(self.screen, color, (x, y), scaled_radius)
                        pygame.draw.circle(self.screen, self.WHITE, (x, y), scaled_radius, 2)
                    else:
                        pygame.draw.circle(self.screen, color, (x, y), self.radius-2)
                        # 添加高光效果
                        pygame.draw.circle(self.screen, self.WHITE, (x-5, y-5), 5)
        
        # 绘制掉落动画
        if self.falling_piece is not None:
            color = self.RED if self.board.current_player == 2 else self.YELLOW  # 注意这里用2是因为已经切换了玩家
            x = self.falling_piece * self.square_size + self.square_size//2
            pygame.draw.circle(self.screen, color, (x, int(self.falling_y)), self.radius-2)
            pygame.draw.circle(self.screen, self.WHITE, (x-5, int(self.falling_y)-5), 5)
            
            # 更新掉落动画
            self.falling_y += self.animation_speed
            if self.falling_y >= self.target_y:
                self.falling_piece = None
                # 检查是否获胜
                if self.board.get_winner() is not None:
                    self.find_winning_pieces()
        
        # 如果游戏结束，显示获胜信息
        winner = self.board.get_winner()
        if winner is not None:
            self.winning_animation_frame += 1
            color = self.RED if winner == 1 else self.YELLOW
            text = self.font.render(f"玩家 {winner} 获胜！", True, color)
            text_rect = text.get_rect(center=(self.width/2, self.square_size/2))
            self.screen.blit(text, text_rect)
        elif self.board.is_full():
            text = self.font.render("平局！", True, self.WHITE)
            text_rect = text.get_rect(center=(self.width/2, self.square_size/2))
            self.screen.blit(text, text_rect)
            
        pygame.display.update()

    def get_column_from_mouse(self, pos):
        """根据鼠标位置获取对应的列"""
        x = pos[0]
        col = x // self.square_size
        if 0 <= col < self.board.cols:
            return col
        return None

    def handle_click(self, pos):
        """处理鼠标点击事件"""
        col = self.get_column_from_mouse(pos)
        if col is not None and self.board.is_valid_move(col) and self.falling_piece is None:
            # 开始掉落动画
            self.falling_piece = col
            self.falling_y = self.square_size/2
            # 计算目标位置
            for row in range(self.board.rows-1, -1, -1):
                if self.board.board[row][col] == 0:
                    self.target_y = (row + 1) * self.square_size + self.square_size/2
                    break
            return self.board.make_move(col)
        return False

    def handle_mouse_motion(self, pos):
        """处理鼠标移动事件"""
        self.hover_col = self.get_column_from_mouse(pos)

    def find_winning_pieces(self):
        """找出获胜的四个棋子位置"""
        if not self.board.last_move:
            return
        
        row, col = self.board.last_move
        player = self.board.get_winner()
        if player is None:
            return
            
        directions = [
            [(0, 1), (0, -1)],  # 水平
            [(1, 0), (-1, 0)],  # 垂直
            [(1, 1), (-1, -1)], # 对角线
            [(1, -1), (-1, 1)]  # 反对角线
        ]
        
        for dir_pair in directions:
            pieces = [(row, col)]
            for direction in dir_pair:
                r, c = row, col
                dr, dc = direction
                while True:
                    r, c = r + dr, c + dc
                    if (not (0 <= r < self.board.rows and 0 <= c < self.board.cols) or 
                        self.board.board[r][c] != player):
                        break
                    pieces.append((r, c))
            if len(pieces) >= 4:
                self.winning_pieces = pieces
                return

    def quit(self):
        """退出游戏"""
        pygame.quit() 