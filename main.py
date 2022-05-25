import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import datetime


class main:

    qbs = pd.read_csv("qb_stats_2010_2021.csv")
    exit_program = False
    print('\nWelcome to Advanced Football Analytics!\n'
          'Using data from 2010-2021, this model can predict NFL QB stats\n')

    while not exit_program:

        print('Select the following statistics to see their correlation:\n\n'
              '1 - Completion Percentage and Yards Per Attempt\n'
              '2 - Rushing Yards Per Game and Passer Rating\n'
              '3 - Passing Attempts Per Game and Passer Rating\n'
              '4 - Predict Passer Rating\n'
              '5 - Exit Program\n\n'
              'Enter the number corresponding to your selection: ')

        choice = int(input())

        if choice == 1:

            x = qbs['CmpP'].values
            y = qbs['Y/A'].values

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            model = linear_model.LinearRegression()
            model.fit(x, y)
            y_predict = model.predict(x)

            plt.scatter(x, y)
            plt.xlabel('Career Completion Percentage')
            plt.ylabel('Career Yards Per Attempt')
            plt.title('NFL QBs 2010-2021')

            plt.plot(x, y_predict, color='r')
            plt.show()

        elif choice == 2:

            x = qbs['RushYardsPerGame'].values
            y = qbs['Rate'].values

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            model = linear_model.LinearRegression()
            model.fit(x, y)
            y_predict = model.predict(x)

            plt.scatter(x, y)
            plt.xlabel('Career Rushing Yards Per Game')
            plt.ylabel('Career Passer Rating')
            plt.title('NFL QBs 2010-2021')

            plt.plot(x, y_predict, color='r')
            plt.show()

        if choice == 3:

            x = qbs['Pass_Att'].values
            g = qbs['G'].values
            apg = []

            for i in range(0, len(x)):
                apg.append(x[i] / g[i])

            x = apg
            x = np.array(x)
            y = qbs['Rate'].values

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            model = linear_model.LinearRegression()
            model.fit(x, y)
            y_predict = model.predict(x)

            plt.scatter(x, y)
            plt.xlabel('Career Passing Attempts Per Game')
            plt.ylabel('Career Passer Rating')
            plt.title('NFL QBs 2010-2021')

            plt.plot(x, y_predict, color='r')
            plt.show()

        # elif choice == 4:
        #
        # elif choice == 5:

        else:

            print('Invalid input. Please try again')
            file = open(r"error_log", "a+")
            ct = datetime.datetime.now()
            string = 'Invalid input: ' + str(choice) + ' at ' + str(ct) + '\n'
            file.write(string)
            file.close()





