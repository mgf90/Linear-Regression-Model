import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model


class main:

    qbs = pd.read_csv("qb_stats_2010_2021.csv")
    # print(qbs)

    print('\nWelcome to Advanced Football Analytics!\n'
          'Using data from 2010-2021, this model can predict NFL QB passer rating\n'
          'Select the following statistic to see its correlative relationship to passer rating:\n\n'
          '1 - Rushing Yards\n2 - Rushing Yards Per Game\n3 - Passing Attempts Per Game\n\n'
          'Enter the number corresponding to your selection: ')

    choice = int(input())

    if choice == 1:

        x = qbs['Rush_Yds'].values
        y = qbs['Rate'].values

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        model = linear_model.LinearRegression()
        model.fit(x, y)
        y_predict = model.predict(x)

        plt.scatter(x, y)
        plt.xlabel('Career Rushing Yards')
        plt.ylabel('Career Passer Rating')
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


