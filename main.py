import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import datetime

# fits model and displays scatter plot
def fit_model(x, y, xlabel, ylabel):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    model = linear_model.LinearRegression()
    model.fit(x, y)

    y_predict = model.predict(x)

    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('NFL QBs 2010-2021')

    plt.plot(x, y_predict, color='r')
    plt.show()

class main:

    qbs = pd.read_csv("qb_stats_2010_2021.csv")
    exit_program = False

    print('\nWelcome to Advanced Football Analytics!\n'
          'Using data from 2010-2021, this model can predict NFL QB stats\n')

    while not exit_program:

        print('MAIN MENU\n\n'
              '1 - Completion Percentage and Yards Per Attempt Model\n'
              '2 - Rushing Yards Per Game and Passer Rating Model\n'
              '3 - Passing Attempts Per Game and Passer Rating Model\n'
              '4 - Predict Passer Rating\n'
              '5 - Exit Program\n\n'
              'Enter the number corresponding to your selection: ')

        choice = int(input())

        # selects Completion Percentage model
        if choice == 1:

            x = qbs['CmpP'].values
            y = qbs['Y/A'].values
            xlabel = 'Career Completion Percentage'
            ylabel = 'Career Yards Per Attempt'

            fit_model(x, y, xlabel, ylabel)

        # selects Rushing Yards Per Game model
        elif choice == 2:

            x = qbs['RushYardsPerGame'].values
            y = qbs['Rate'].values
            xlabel = 'Career Rushing Yards Per Game'
            ylabel = 'Career Passer Rating'

            fit_model(x, y, xlabel, ylabel)

        # selects Passing Attempts Per Game model
        if choice == 3:

            x = qbs['Pass_Att'].values
            g = qbs['G'].values
            apg = []

            # calculates passing attempts per game
            for i in range(0, len(x)):
                apg.append(x[i] / g[i])

            x = apg
            x = np.array(x)
            y = qbs['Rate'].values

            xlabel = 'Career Passing Attempts Per Game'
            ylabel = 'Career Passer Rating'

            fit_model(x, y, xlabel, ylabel)

        # selects passer rating prediction
        elif choice == 4:

            model = linear_model.LinearRegression()
            model.fit(qbs[['CmpP']], qbs.Rate)
            inp = float(input('Enter a career completion percentage. Just type the number: '))
            inp = [[inp]]
            y_predict = model.predict(inp)

            inp = str(inp[0][0])
            print(f'A player with a completion percentage of {inp}% is predicted to have a {y_predict[0]} '
                  f'passer rating.\n')


        # exits the program
        elif choice == 5:

            exit_program = True

        # prints error message and writes it to error log. Returns to main menu
        else:

            print('Invalid input. Please try again')
            file = open(r"error_log", "a+")
            ct = datetime.datetime.now()
            string = 'Invalid input: ' + str(choice) + ' at ' + str(ct) + '\n'
            file.write(string)
            file.close()





