import numpy as np
from itertools import combinations, chain
from collections import defaultdict, Counter

"""
example run:

board = np.array([
       [0, 0, 0, 0, 1, 0, 0, 3, 0],
       [0, 0, 9, 0, 0, 5, 0, 0, 8],
       [8, 0, 4, 0, 0, 6, 0, 2, 5],
       [0, 0, 0, 0, 0, 0, 6, 0, 0],
       [0, 0, 8, 0, 0, 4, 0, 0, 0],
       [1, 2, 0, 0, 8, 7, 0, 0, 0],
       [3, 0, 0, 9, 0, 0, 2, 0, 0],
       [0, 6, 5, 0, 0, 8, 0, 0, 0],
       [9, 0, 0, 0, 0, 0, 0, 0, 0]])

refill_possible(board)

solve_sudoku(board,possible_board) # you may have to run a few times
"""

def refill_possible(board):
  """
    Refill the possible values for each cell in the Sudoku board.

    Args:
        board (np.array): A 9x9 numpy array representing the Sudoku board. 
                          Cells with a value of 0 are considered empty.

    Returns:
        None: Modifies the global possible_board variable, setting each cell to a set of possible values (1-9).
  """

  global possible_board
  #Create one set per cell with values 1 through 9
  cell_set = [set(range(1,10)) for _ in range(board.size)]

  #Reshape the cells like the board
  possible_board = np.array(cell_set).reshape(board.shape)

  for i in range(board.shape[0]):
    for j in range(board.shape[1]):
      if board[i][j] != 0:
        possible_board[i][j] = {board[i][j]}

def update_possible_values(board, possible_board):
  """
    Update the possible values for each cell based on current board state. Check the row, column, and grid intersections.

    Args:
        board (np.array): A 9x9 numpy array representing the Sudoku board. 
                          Cells with a value of 0 are considered empty.
        possible_board (np.array): A 9x9 numpy array of sets, where each set contains the possible values for that cell.

    Returns:
        bool: True if any cell's value was updated, False otherwise.
  """
  #Comparison to see if there was any change later
  updated_board = False
  init_board = board.copy()

  #Iterate through the board
  for row in range(board.shape[0]):
    for col in range(board.shape[1]):

      # Current cell to Look at
      cell = board[row,col]

      # If it’s already filled, then skip.
      if cell != 0:
        continue

      # First: Union all values in row i and column j.
      row_vals = set(board[row,:])
      col_vals = set(board[:,col])
      not_vals = row_vals.union(col_vals)

      # Next: check the subgrid and append to not_vals
      gx = (row // 3) * 3
      gy = (col // 3) * 3
      grid = board[gx:gx + 3, gy:gy + 3]

      grid_vals = set(grid.flatten())
      not_vals = not_vals.union(grid_vals)

      # Remove 0 from our possible values set
      not_vals.discard(0)

      # Update the possible values for each cell
      possible_board[row,col] = possible_board[row, col].difference(not_vals)

      # If there is only one value it can be, fill it in
      if len(possible_board[row, col]) == 1:
        board[row, col] = possible_board[row, col].pop()
        possible_board[row, col].clear()

        # The board was changed
        updated_board = True
        print(f'update at {row+1},{col+1}')

  #Did the board change
  return updated_board

def place_unique_row(board, possible_board):
    """
    Place unique values in each row (if a possible value only appears once for a row on the board).

    Args:
        board (np.array): A 9x9 numpy array representing the Sudoku board. 
                          Cells with a value of 0 are considered empty.
        possible_board (np.array): A 9x9 numpy array of sets, where each set contains the possible values for that cell.

    Returns:
        bool: True if any cell's value was updated, False otherwise.
    """

    updated_board = False

    for row in range(9):
        # Flatten the possible values in the current row
        poss_values = [val for cell in possible_board[row] for val in cell]

        # Count the number of time each number (1-9) appears
        counts = Counter(poss_values)

        # Identify unique values
        unique_vals = {num for num, count in counts.items() if count == 1}

        for unique in unique_vals:
            for col in range(9):
                if unique in possible_board[row, col]:
                    board[row, col] = unique
                    possible_board[row, col].clear()
                    print(f'updated {row+1}, {col+1}')
                    updated_board = True
                    break

    return updated_board

def place_unique_col(board, possible_board):
    """
      Place unique values in each column (if a possible value only appears once for a column on the board).

      Args:
          board (np.array): A 9x9 numpy array representing the Sudoku board. 
                            Cells with a value of 0 are considered empty.
          possible_board (np.array): A 9x9 numpy array of sets, where each set contains the possible values for that cell.

      Returns:
          bool: True if any cell's value was updated, False otherwise.
    """
    updated_board = False

    for col in range(9):
        # Flatten the possible values in the current column
        poss_values = [val for row in range(9) for val in possible_board[row, col]]

        # Count the number of times each number (1-9) appears
        counts = Counter(poss_values)

        # Identify unique values
        unique_vals = {num for num, count in counts.items() if count == 1}

        for unique in unique_vals:
            for row in range(9):
                if unique in possible_board[row, col]:
                    board[row, col] = unique
                    possible_board[row, col].clear()
                    print(f'updated {row+1}, {col+1}')
                    updated_board = True
                    break

    return updated_board

def place_unique_subgrid(board, possible_board):
  """
    Place unique values in each subgrid (if a possible value only appears once for a subgrid on the board).

    Args:
        board (np.array): A 9x9 numpy array representing the Sudoku board. 
                          Cells with a value of 0 are considered empty.
        possible_board (np.array): A 9x9 numpy array of sets, where each set contains the possible values for that cell.

    Returns:
        bool: True if any cell's value was updated, False otherwise.
  """
  updated_board = False

  for grid_row in range(3):
      for grid_col in range(3):
          gx, gy = grid_row * 3, grid_col * 3

          poss_values = [val for i in range(3) for j in range(3) for val in possible_board[gx + i, gy + j]]
          counts = Counter(poss_values)
          unique_vals = {num for num, count in counts.items() if count == 1}

          for unique in unique_vals:
              for i in range(3):
                  for j in range(3):
                      if unique in possible_board[gx + i, gy + j]:
                          board[gx + i, gy + j] = unique
                          possible_board[gx + i, gy + j].clear()
                          print(f'updated {gx + i+1}, {gy + j+1}')
                          updated_board = True
                          break

  return updated_board

def create_pairs(s,n_in_pair = 2):
    """
      Create pairs of elements from a set.

      Args:
          s (set): A set of elements to create pairs from.
          n_in_pair (int): The number of elements in each pair.

      Returns:
          list: A list of pairs (tuples) created from the set.
    """
    return list(combinations(s, n_in_pair))

def check_cells(possible_area,n_in_pair):

  """
    Check and update cells based on possible value pairs in a specified area.
    Look at hidden pairs, where we know n_in_pair elements can only go in n_in_pair cells,
      which means they must go there and not in another cells. Then, remove these hidden pairs
      from other cells.

    Args:
        possible_area (np.array): An array representing a row, column, or subgrid with possible values.
        n_in_pair (int): The number of elements in each pair.

    Returns:
        np.array: The updated possible_area array.
  """

  #Get the shape to adjust it if it is a grid later
  possible_shape = possible_area.shape
  possible_area = possible_area.flatten()

  #Create a list of lists, with the combinations of possible numbers for each cell index
  pairs_list = [[tuple(sorted(t)) for t in create_pairs(s, n_in_pair)] for s in possible_area]

  #Find cells that only have one possible pair in their pairs_list
  one_possible_pair = [p for p in pairs_list if len(p) == 1]

  #If the possible pair of size n_in_pair only shows up n_in_pair times, then 
  #remove that pair from all other cells
  how_many = Counter(sum(one_possible_pair,[]))

  for k,v in how_many.items():
    if v == n_in_pair:
      for n,p in enumerate(pairs_list):
        if [k] != p:
          possible_area[n] = possible_area[n].difference(set(k))
        else:
          possible_area[n] = set(k)

  return possible_area.reshape(possible_shape)

def check_row_in_subgrid(board,possible_board):
  """
    Check and update possible values in subgrids based on unique values in intersecting rows.

    Args:
        board (np.array): A 9x9 numpy array representing the Sudoku board. 
                          Cells with a value of 0 are considered empty.
        possible_board (np.array): A 9x9 numpy array of sets, where each set contains the possible values for that cell.

    Returns:
        None: Modifies the possible_board array in place.
  """
  can_update = False 

  for row_num in range(9):
    temp = possible_board[row_num,:]

    #Get all the unique possible numbers and make them keys in a dictionary. Make the values empty lists.
    unique_possible = sum([[i for i in v] for v in temp],[])

    which_subgrids = {num:[] for num in unique_possible}

    #For each number
    for num in unique_possible:

      #Go through which subgrid the values from the column are in
      for subgrid in range(3):

        #Combine all the values at the intersection of the column and a subgrid into one list
        eval_temp = temp[subgrid*3:subgrid*3+3]
        eval_temp = sum([[i for i in v] for v in eval_temp],[])

        #Add the index of the subgrid to the dictionary list value if the number is in the subgrid
        if num in eval_temp:
          which_subgrids[num].append(subgrid)

    #Only get unique subgrid indexes so they do not repeat
    which_subgrids = {num: np.unique(v) for num,v in which_subgrids.items()}

    #If there is only one subgrid that the number can go in
    for k,v in {num: v[0] for num,v in which_subgrids.items() if len(v) == 1}.items():
      
      #get the full subgrid that the column intersects
      grid_look = possible_board[(row_num // 3)*3:(row_num // 3)*3+3,v*3:v*3+3].copy()

      #Remove the column from the subgrid
      removed_row = grid_look[(row_num % 3),:]
      fixed = np.delete(grid_look, (row_num % 3), axis=0)

      #For the other columns in the subgrid, discard the value that appears in the segment of the column we are looking at
      for row in fixed:
        for item in row:
            if isinstance(item, set):
                item.discard(k)

      #recombine the columns to create the full subgrid. Set this as the subgrid for possible values
      result = np.insert(fixed, (row_num % 3), removed_row, axis=0)
      if (result != possible_board[(row_num // 3)*3:(row_num // 3)*3+3,v*3:v*3+3]).any():
        # print(result)
        # print(possible_board[(row_num // 3)*3:(row_num // 3)*3+3,v*3:v*3+3])
        possible_board[(row_num // 3)*3:(row_num // 3)*3+3,v*3:v*3+3] = result    
        can_update = True 

def check_col_in_subgrid(board,possible_board):
  """
    Check and update possible values in subgrids based on unique values in intersecting columns.

    Args:
        board (np.array): A 9x9 numpy array representing the Sudoku board. 
                          Cells with a value of 0 are considered empty.
        possible_board (np.array): A 9x9 numpy array of sets, where each set contains the possible values for that cell.

    Returns:
        None: Modifies the possible_board array in place.
  """
  can_update = False

  for col_num in range(9):
    temp = possible_board[:,col_num].copy()

    #Get all the unique possible numbers and make them keys in a dictionary. Make the values empty lists.
    unique_possible = sum([[i for i in v] for v in temp],[])

    which_subgrids = {num:[] for num in unique_possible}

    #For each number
    for num in unique_possible:

      #Go through which subgrid the values from the column are in
      for subgrid in range(3):

        #Combine all the values at the intersection of the column and a subgrid into one list
        eval_temp = temp[subgrid*3:subgrid*3+3]
        eval_temp = sum([[i for i in v] for v in eval_temp],[])

        #Add the index of the subgrid to the dictionary list value if the number is in the subgrid
        if num in eval_temp:
          which_subgrids[num].append(subgrid)

    #Only get unique subgrid indexes so they do not repeat
    which_subgrids = {num: np.unique(v) for num,v in which_subgrids.items()}

    #If there is only one subgrid that the number can go in
    for k,v in {num: v[0] for num,v in which_subgrids.items() if len(v) == 1}.items():
      
      #get the full subgrid that the column intersects
      grid_look = possible_board[v*3:v*3+3,(col_num // 3)*3:(col_num // 3)*3+3].copy()

      #Remove the column from the subgrid
      removed_col = grid_look[:, (col_num % 3)]
      fixed = np.delete(grid_look, (col_num % 3), axis=1)

      #For the other columns in the subgrid, discard the value that appears in the segment of the column we are looking at
      for row in fixed:
        for item in row:
            if isinstance(item, set):
                item.discard(k)

      #recombine the columns to create the full subgrid. Set this as the subgrid for possible values
      result = np.insert(fixed, (col_num % 3), removed_col, axis=1)
      if (possible_board[v*3:v*3+3,(col_num // 3)*3:(col_num // 3)*3+3] != result).any():
        # print(result)
        # print(possible_board[v*3:v*3+3,(col_num // 3)*3:(col_num // 3)*3+3])
        possible_board[v*3:v*3+3,(col_num // 3)*3:(col_num // 3)*3+3] = result      
        can_update = True    


def solve_sudoku(board,possible_board):
    """
      Solve the Sudoku puzzle by iteratively updating the possible values and placing unique values.

      Args:
          board (np.array): A 9x9 numpy array representing the Sudoku board. 
                            Cells with a value of 0 are considered empty.
          possible_board (np.array): A 9x9 numpy array of sets, where each set contains the possible values for that cell.

      Returns:
          np.array: The solved Sudoku board.
    """
    can_update = True

    #While we can update our grid
    while can_update:
        #Update the possible values in the grid
        can_update = update_possible_values(board, possible_board)

        #If nothing was updated in the previous step
        if not can_update:
            can_update = place_unique_row(board, possible_board)

        #If nothing was updated in the previous step
        if not can_update:
            can_update = place_unique_col(board, possible_board)

        #If nothing was updated in the previous step
        if not can_update:
            can_update = place_unique_subgrid(board, possible_board)

        #Iterate through columns
        if not can_update:
          #Let's check pairings of 2, 3 to start
          for n_in_pair in range(2,4):
            for i in range(9):
              #Get the area we are looking at
              check = possible_board[:,i]
              #If any values on the possible_board changed with our function, then update possible_board
              if (check_cells(check,n_in_pair) != check).any():
                possible_board[:,i] = check_cells(check,n_in_pair)
                can_update = True

        #Iterate through rows
        if not can_update:
          for n_in_pair in range(2,4):
            for i in range(9):
              check = possible_board[i,:]
              if (check_cells(check,n_in_pair) != check).any():
                possible_board[i,:] = check_cells(check,n_in_pair)
                can_update = True

        #Iterate through subgrids
        if not can_update:
          for n_in_pair in range(2,4):
            for grid_row in range(3):
              for grid_col in range(3):
                gx, gy = grid_row * 3, grid_col * 3
                check = possible_board[gx:gx + 3,gy:gy + 3]
                if (check_cells(check,n_in_pair) != check).any():
                  possible_board[gx:gx + 3,gy:gy + 3] = check_cells(check,n_in_pair)
                  can_update = True


        if not can_update:
          check_col_in_subgrid(board,possible_board)

        if not can_update:
          check_row_in_subgrid(board,possible_board)  

    return board
