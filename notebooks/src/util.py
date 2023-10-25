import pandas as pd 

def distributionPreservingDownsample(df, column, sample_size):
    if sample_size > len(df):
        raise ValueError('Sample size must be less than the number of rows in the DataFrame.')

    # Calculate the number of rows for each flag value
    counts = df[column].value_counts()
    True_count = int(sample_size * counts[True] / len(df))
    False_count = sample_size - True_count

    # Split the DataFrame based on the flag value
    df_flag_A = df[df[column] == True]
    df_flag_B = df[df[column] == False]

    # Randomly sample rows for each flag value based on the desired count
    df_sampled_A = df_flag_A.sample(n=True_count)
    df_sampled_B = df_flag_B.sample(n=False_count)

    # Concatenate the sampled DataFrames
    df_sampled = pd.concat([df_sampled_A, df_sampled_B])

    # Display the resulting sampled DataFrame
    # print(df_sampled)
    return df_sampled