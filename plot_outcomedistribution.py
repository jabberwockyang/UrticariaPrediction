
import matplotlib.pyplot as plt
import pandas as pd

def plot_outcome(df,targetvalue ):
    # calculate percentage of df[targetvalue] under 42 100 365
    perlessthan42 = df[df[targetvalue] < 42].shape[0] / df.shape[0]
    per42_90 = df[(df[targetvalue] >= 42) & (df[targetvalue] < 90)].shape[0] / df.shape[0]
    per90_365 = df[(df[targetvalue] >= 90) & (df[targetvalue] < 365)].shape[0] / df.shape[0]
    per365_5y = df[(df[targetvalue] >= 365) & (df[targetvalue] < 1825)].shape[0] / df.shape[0]
    permorethan5y = df[df[targetvalue] >= 1825].shape[0] / df.shape[0]

    # plot histogram of outcome
    fig, ax = plt.subplots()
    ax.hist(df[targetvalue], bins=50, color='blue', alpha=0.7)
    ax.set_title(f'{targetvalue} Distribution')
    ax.set_xlabel(targetvalue)
    ax.set_ylabel('Frequency')
    ax.axvline(x=42, color='red', linestyle='--', label='42 days')
    ax.axvline(x=90, color='green', linestyle='--', label='90 days')
    ax.axvline(x=365, color='purple', linestyle='--', label='365 days')
    ax.axvline(x=1825, color='orange', linestyle='--', label='5 years')
    ax.legend(
        title=f"Percentage of {targetvalue} \n < 6w: {perlessthan42*100:.2f}% \n 6w-3m: {per42_90*100:.2f}% \n 3m-1y: {per90_365*100:.2f}% \n 1-5 years: {per365_5y*100:.2f}% \n > 5 years: {permorethan5y*100:.2f}%",
        loc='upper right'
    )
    plt.savefig(f"outcome_distribution.png")

if __name__ == '__main__':
    df = pd.read_csv("output/dataforxgboost.csv")
    plot_outcome(df, 'VisitDuration')
