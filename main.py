# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', False)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # data = pd.read_csv(r"E:\Datasets\WADI.A2\WADI.A2_19 Nov 2019\WADI_attackdataLABLE.csv", skiprows=1)
    # print(data.head())
    # print(data.describe())
    # print(data.shape)

    import pandas as pd

    train_new = pd.read_csv(r'E:\Datasets\WADI.A2\WADI.A2_19 Nov 2019\WADI_14days_new.csv')
    test_new = pd.read_csv(r'E:\Datasets\WADI.A2\WADI.A2_19 Nov 2019\WADI_attackdataLABLE.csv', skiprows=1)

    print(train_new.head())
    print(train_new.columns)

    print(test_new.head())
    print(test_new.columns)

    # test = pd.read_csv('./WADI.A1_9 Oct 2017/WADI_attackdata.csv')
    # train = pd.read_csv('./WADI.A1_9 Oct 2017/WADI_14days.csv', skiprows=4)
    #

    def recover_date(str1, str2):
        return str1 + " " + str2


    # train["datetime"] = train.apply(lambda x: recover_date(x['Date'], x['Time']), axis=1)
    # train["datetime"] = pd.to_datetime(train['datetime'])

    # train_time = train[['Row', 'datetime']]
    # train_new_time = pd.merge(train_new, train_time, how='left', on='Row')
    # del train_new_time['Row']
    # del train_new_time['Date']
    # del train_new_time['Time']
    # train_new_time.to_csv('./processing/WADI_train.csv', index=False)

    # test["datetime"] = test.apply(lambda x: recover_date(x['Date'], x['Time']), axis=1)
    # test["datetime"] = pd.to_datetime(test['datetime'])
    # test = test.loc[-2:, :]


    test_new_time = test_new.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'label'})
    test_new_time.loc[test_new_time['label'] == 1, 'label'] = 0
    test_new_time.loc[test_new_time['label'] == -1, 'label'] = 1

    label = test_new_time["label"]

    test_new_time.drop(columns=["Row ","Date ","Time","label"],inplace=True)

    print(train_new.head())
    train_new.drop(columns=["Row", "Date", "Time"],inplace=True)
    print(test_new_time.head())
    # test_new_time.to_csv('./processing/WADI_test.csv', index=False)

    test_new_time.to_csv("./Data/WADIV2/test/WADIV2.csv",index=False,header=False)
    train_new.to_csv("./Data/WADIV2/train/WADIV2.csv", index=False, header=False)
    label.to_csv("./Data/WADIV2/label/WADIV2.csv", index=False, header=False)



