
import numpy as np
import random
import pygame
import sys
import math
import matplotlib.pyplot as plt
from pygame.locals import *

# Importing the Dqn object 
from AI_Brain import Dqn

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(42,7,0.9) # 42 possible actions, 7 actions, gama = 0.9
brain2 = Dqn(42,7,0.9)
last_reward = 0 # initializing the last reward
scores = [] # initializing the mean score curve (sliding window of the rewards) with respect to time
scores2 = [] # initializing the mean score curve (sliding window of the rewards) with respect to time


def save(): # save button
    print('saving brain...')
    myfont.render("Saving Brain", 1, YELLOW)
    brain.save()
    plt.plot(scores)
    plt.show()

def load(): # load button
    print("loading last saved brain...")
    brain.load()

def save2(): # save button
    print('saving brain...')
    myfont.render("Saving Brain", 1, YELLOW)
    brain2.save2()
    plt.plot(scores2)
    plt.show()

def load2(): # load button
    print("loading last saved brain...")
    brain2.load2()


BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 10000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 60
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 50
    if window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 20
    if window.count(opp_piece) == 4:
        score -= 1000

    return score

def score_position(board, piece):
    score = 0

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 1.1

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else: # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def pick_best_move(board, piece):

    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):        
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == AI_PIECE: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

pygame.init()
pygame.font.init() # you have to call this at the start, 

SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)
screen = pygame.display.set_mode(size)
myfont = pygame.font.SysFont("monospace", 75)

myAI_wins = []
scoringAI_wins = [] 

def main():
       
    board = create_board()
    print_board(board)
    game_over = False
    

    draw_board(board)
    pygame.display.update()
    
    
    turn = random.randint(PLAYER, AI)
    
    
    
    load()
    load2()
    while not game_over:
#        screen.blit(myfont.render("Playing", 1, RED), (40,10))
        AI_board = np.array(np.ndarray.flatten(board.copy()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
    
#            if event.type == pygame.MOUSEMOTION:
#                pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
#                posx = event.pos[0]
#                if turn == PLAYER:
#                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
#    
#            pygame.display.update()
#    
#            if event.type == pygame.MOUSEBUTTONDOWN:
#                pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
                #print(event.pos)
                # Ask for Player 1 Input
        if turn == PLAYER:
#            col, minimax_score = minimax(board, 4, -math.inf, math.inf, True)
#                    posx = event.pos[0]
#                    col = int(math.floor(posx/SQUARESIZE))
            col = brain.update(score_position(board, PLAYER_PIECE), AI_board.tolist()) # playing the action from our ai (the object brain of the dqn class)
            scores.append(brain.score())

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
#                    label = myfont.render("Player 1 wins!!", 1, RED)
#                    screen.blit(label, (40,10))
                    game_over = True
                    scoringAI_wins.append(1)

                turn += 1
                turn = turn % 2

                print_board(board)
                draw_board(board)
    
    
        # # Ask for AI Input
        if turn == AI and not game_over:                
            
            #col = random.randint(0, COLUMN_COUNT-1)
            #col = pick_best_move(board, AI_PIECE)
            AI_valid_locations = get_valid_locations(board).copy()
            AI_valid_locations = [i if i in get_valid_locations(board) else 0 for i in range(COLUMN_COUNT)]
            for i in range(COLUMN_COUNT):
                if AI_valid_locations[i] > 1:
                    AI_valid_locations[i] = 1
            
            col = brain2.update(score_position(board, AI_PIECE), AI_board.tolist()) # playing the action from our ai (the object brain of the dqn class)
            scores2.append(brain2.score())
            
            if is_valid_location(board, col):
                #pygame.time.wait(500)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)
    
                if winning_move(board, AI_PIECE):
#                    label = myfont.render("Rob wins!!", 1, YELLOW)
#                    screen.blit(label, (40,10))
                    game_over = True
                    myAI_wins.append(1) 
    
                print_board(board)
                draw_board(board)
    
                turn += 1
                turn = turn % 2
    
        if game_over:
            save()
            save2()
            pygame.time.wait(50)
#            pygame.display.quit()
#            pygame.quit() 
#            sys.exit()

play_again = 0

while play_again != 50: 
    if play_again != 50:
        main()
    else:
        break
    play_again +=1
    
pygame.display.quit()
pygame.quit() 
sys.exit()