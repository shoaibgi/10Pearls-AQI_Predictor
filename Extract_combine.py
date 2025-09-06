from Plot_AQI import avg_data_2020, avg_data_2021, avg_data_2022, avg_data_2023, avg_data_2024
import requests
import sys
import pandas as pd
from bs4 import BeautifulSoup
import os
import csv


def met_data(month, year):
    file_html = open('Data/Html_Data/{}/{}.html'.format(year, month), 'rb')
    plain_text = file_html.read()

    tempD = []
    finalD = []

    soup = BeautifulSoup(plain_text, "lxml")

    # ✅ use find_all instead of findAll
    for table in soup.find_all('table', {'class': 'medias mensuales numspan'}):
        for tbody in table:
            for tr in tbody:
                a = tr.get_text()
                tempD.append(a)

    rows = len(tempD) / 15

    for times in range(round(rows)):
        newtempD = []
        for i in range(15):
            newtempD.append(tempD[0])
            tempD.pop(0)
        finalD.append(newtempD)

    length = len(finalD)

    if length > 1:
        finalD.pop(length - 1)
        finalD.pop(0)

    for a in range(len(finalD)):
        # drop unused columns
        for idx in [6, 13, 12, 11, 10, 9, 0]:
            if idx < len(finalD[a]):
                finalD[a].pop(idx)

    return finalD


def data_combine(year, cs):
    for a in pd.read_csv('Data/Real-Data/real_' + str(year) + '.csv', chunksize=cs):
        df = pd.DataFrame(data=a)
        mylist = df.values.tolist()
    return mylist


if __name__ == "__main__":
    if not os.path.exists("Data/Real-Data"):
        os.makedirs("Data/Real-Data")

    for year in range(2021, 2025):
        final_data = []
        with open('Data/Real-Data/real_' + str(year) + '.csv', 'w') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            wr.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])

        # collect weather data
        for month in range(1, 13):
            temp = met_data(month, year)
            final_data.extend(temp)

        # collect pm data
        pm = getattr(sys.modules[__name__], 'avg_data_{}'.format(year))()

        # ✅ handle mismatched lengths
        if len(pm) == 364:
            pm.insert(364, '-')

        max_len = max(len(final_data), len(pm))
        aligned_data = []

        for i in range(max_len):
            row = final_data[i] if i < len(final_data) else ["NA"] * 8
            pm_val = pm[i] if i < len(pm) else "NA"
            row.insert(8, pm_val)
            aligned_data.append(row)

        with open('Data/Real-Data/real_' + str(year) + '.csv', 'a') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            for row in aligned_data:
                if all(elem not in ("", "-") for elem in row):
                    wr.writerow(row)

    # combine yearly data
    data_2021 = data_combine(2021, 600)
    data_2022 = data_combine(2022, 600)
    data_2023 = data_combine(2023, 600)
    data_2024 = data_combine(2024, 600)

    total = data_2021 + data_2022 + data_2023 + data_2024

    with open('Data/Real-Data/AQI_Combine.csv', 'w') as csvfile:
        wr = csv.writer(csvfile, dialect='excel')
        wr.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])
        wr.writerows(total)

    df = pd.read_csv('Data/Real-Data/AQI_Combine.csv')
    print(df.head())
