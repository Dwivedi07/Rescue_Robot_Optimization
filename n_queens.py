from pulp import *

def solve_n_queens(n):
    # Create the LP problem object
    problem = LpProblem("n-queens", LpMinimize)

    # Define the decision variables
    queens = LpVariable.dicts("queen", [(i, j) for i in range(n) for j in range(n)], cat=LpBinary)

    # Define the constraints

    for i in range(n):
        # Only one queen in each row
        problem += lpSum([queens[(i, j)] for j in range(n)]) == 1

        # Only one queen in each column
        problem += lpSum([queens[(j, i)] for j in range(n)]) == 1

        # Only one queen in each diagonal
        for j in range(n):
            if i + j < n:
                problem += lpSum([queens[(i+k, j+k)] for k in range(n-i)]) <= 1
                if j > 0:
                    problem += lpSum([queens[(i+k, j-k)] for k in range(n-i)]) <= 1

    # Define the objective function (not really used here)
    problem += 0

    # Solve the problem
    problem.solve()

    # Print the solution
    for i in range(n):
        row = ""
        for j in range(n):
            if value(queens[(i, j)]) == 1:
                row += "Q "
            else:
                row += ". "
        print(row)


solve_n_queens(8)