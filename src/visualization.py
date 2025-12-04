import matplotlib.pyplot as plt
import os

def plot_time_series(df, title, outpath=None):
    plt.figure(figsize=(16, 4))
    plt.plot(df.index, df['PJME_MW'])
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('PJME_MW')
    if outpath:
        plt.savefig(outpath)
    plt.show()

def plot_train_test_split(train_df, test_df, outpath=None):
    plt.figure(figsize=(16, 4))
    plt.plot(train_df.index, train_df['PJME_MW'], label='Train')
    plt.plot(test_df.index, test_df['PJME_MW'], label='Test')
    plt.title('Train/Test Split')
    plt.xlabel('Datetime')
    plt.ylabel('PJME_MW')
    plt.legend()
    if outpath:
        plt.savefig(outpath)
    plt.show()

def plot_forecast_vs_actual(test_index, y_test, y_pred, title, outpath=None):
    plt.figure(figsize=(16,6))
    plt.plot(test_index, y_test, label='Actual')
    plt.plot(test_index, y_pred, label='Forecast')
    plt.title(title)
    plt.ylabel('PJME_MW')
    plt.xlabel('Datetime')
    plt.legend()
    if outpath:
        plt.savefig(outpath)
    plt.show()
