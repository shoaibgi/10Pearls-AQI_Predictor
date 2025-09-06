import pandas as pd
import matplotlib.pyplot as plt


def avg_data_2020():
    temp_i = 0
    average = []
    for rows in pd.read_csv('Data/AQI/aqi2020.csv', chunksize=24):
        add_var = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data=rows)
        for index, row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if isinstance(i, (float, int)):
                add_var += i
            elif isinstance(i, str):
                if i not in ['NoData', 'PwrFail', '---', 'InVld', '-']:
                    temp = float(i)
                    add_var += temp
        avg = add_var / 24
        temp_i += 1
        average.append(avg)
    return average


def avg_data_2021():
    temp_i = 0
    average = []
    for rows in pd.read_csv('Data/AQI/aqi2021.csv', chunksize=24):
        add_var = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data=rows)
        for index, row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if isinstance(i, (float, int)):
                add_var += i
            elif isinstance(i, str):
                if i not in ['NoData', 'PwrFail', '---', 'InVld', '-']:
                    temp = float(i)
                    add_var += temp
        avg = add_var / 24
        temp_i += 1
        average.append(avg)
    return average


def avg_data_2022():
    temp_i = 0
    average = []
    for rows in pd.read_csv('Data/AQI/aqi2022.csv', chunksize=24):
        add_var = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data=rows)
        for index, row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if isinstance(i, (float, int)):
                add_var += i
            elif isinstance(i, str):
                if i not in ['NoData', 'PwrFail', '---', 'InVld', '-']:
                    temp = float(i)
                    add_var += temp
        avg = add_var / 24
        temp_i += 1
        average.append(avg)
    return average


def avg_data_2023():
    temp_i = 0
    average = []
    for rows in pd.read_csv('Data/AQI/aqi2023.csv', chunksize=24):
        add_var = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data=rows)
        for index, row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if isinstance(i, (float, int)):
                add_var += i
            elif isinstance(i, str):
                if i not in ['NoData', 'PwrFail', '---', 'InVld', '-']:
                    temp = float(i)
                    add_var += temp
        avg = add_var / 24
        temp_i += 1
        average.append(avg)
    return average


def avg_data_2024():
    temp_i = 0
    average = []
    for rows in pd.read_csv('Data/AQI/aqi2024.csv', chunksize=24):
        add_var = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data=rows)
        for index, row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if isinstance(i, (float, int)):
                add_var += i
            elif isinstance(i, str):
                if i not in ['NoData', 'PwrFail', '---', 'InVld', '-']:
                    temp = float(i)
                    add_var += temp
        avg = add_var / 24
        temp_i += 1
        average.append(avg)
    return average


if __name__ == "__main__":
    lst2020 = avg_data_2020()
    lst2021 = avg_data_2021()
    lst2022 = avg_data_2022()
    lst2023 = avg_data_2023()
    lst2024 = avg_data_2024()

    plt.plot(range(len(lst2020)), lst2020, label="2020 data")
    plt.plot(range(len(lst2021)), lst2021, label="2021 data")
    plt.plot(range(len(lst2022)), lst2022, label="2022 data")
    plt.plot(range(len(lst2023)), lst2023, label="2023 data")
    plt.plot(range(len(lst2024)), lst2024, label="2024 data")

    plt.xlabel('Day')
    plt.ylabel('PM 2.5')
    plt.legend(loc='upper right')
    plt.title("PM 2.5 Data for 2020–2024")
    plt.show()
