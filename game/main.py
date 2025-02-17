import pygame
import sys
from board import Board
from gui import GUI

def main():
    # 初始化游戏
    board = Board()
    gui = GUI(board)
    clock = pygame.time.Clock()
    
    # 主游戏循环
    running = True
    game_over = False
    while running:
        gui.draw_board()
        clock.tick(60)  # 限制帧率为60FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
            if game_over:
                continue
                
            # 处理鼠标移动
            if event.type == pygame.MOUSEMOTION:
                gui.handle_mouse_motion(event.pos)
                
            # 处理鼠标点击
            if event.type == pygame.MOUSEBUTTONDOWN:
                gui.handle_click(event.pos)
        
        # 检查游戏是否结束
        if not game_over and (board.get_winner() is not None or board.is_full()):
            game_over = True
            if board.get_winner() is not None:
                winner = board.get_winner()
                print(f"玩家 {winner} 获胜！")
            else:
                print("游戏平局！")
            # 等待3秒后退出
            pygame.time.wait(3000)
            running = False
            
    gui.quit()

if __name__ == "__main__":
    main() 