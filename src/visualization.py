import matplotlib.pyplot as plt
import os

def plot_time_series(df, title, outpath=None, col_name='PJME_MW'):
    # If col_name not in df, try to use the first column
    if col_name not in df.columns and not df.empty:
        col_name = df.columns[0]
        
    plt.figure(figsize=(16, 4))
    plt.plot(df.index, df[col_name])
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel(col_name)
    if outpath:
        plt.savefig(outpath)
    plt.show()

def plot_train_test_split(train_df, test_df, outpath=None, col_name='PJME_MW'):
    # If col_name not in df, try to use the first column
    if col_name not in train_df.columns and not train_df.empty:
        col_name = train_df.columns[0]

    plt.figure(figsize=(16, 4))
    plt.plot(train_df.index, train_df[col_name], label='Train')
    plt.plot(test_df.index, test_df[col_name], label='Test')
    plt.title('Train/Test Split')
    plt.xlabel('Datetime')
    plt.ylabel(col_name)
    plt.legend()
    if outpath:
        plt.savefig(outpath)
    plt.show()

def plot_forecast_vs_actual(test_index, y_test, y_pred, title, outpath=None, ylabel='Load (MW)'):
    plt.figure(figsize=(16,6))
    plt.plot(test_index, y_test, label='Actual')
    plt.plot(test_index, y_pred, label='Forecast')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Datetime')
    plt.legend()
    if outpath:
        plt.savefig(outpath)
    plt.show()
