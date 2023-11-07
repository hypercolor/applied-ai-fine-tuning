import pandas as pd 
import time

def distributionPreservingDownsample(df, column, sample_size, random_state=42):
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
    df_sampled_A = df_flag_A.sample(n=True_count, random_state=random_state)
    df_sampled_B = df_flag_B.sample(n=False_count, random_state=random_state)

    # Concatenate the sampled DataFrames
    df_sampled = pd.concat([df_sampled_A, df_sampled_B])

    # Display the resulting sampled DataFrame
    # print(df_sampled)
    return df_sampled

def getTrainTestSplit(df, column, train_size, validation_size, random_state=42):
    if train_size + validation_size > len(df):
        raise ValueError('Train size + validation size must be less than the number of rows in the DataFrame.')
    
    train_data = distributionPreservingDownsample(df, column, train_size, random_state)
    test_data_universe = df[~df.index.isin(train_data.index)]
    test_data = distributionPreservingDownsample(test_data_universe, column, validation_size, random_state)

    return train_data, test_data


# <FineTuningJob fine_tuning.job id=ftjob-3xQGCgLB44R1C5jG0hrvcIbt at 0x12bcc42f0> JSON: {
#   "object": "fine_tuning.job",
#   "id": "ftjob-3xQGCgLB44R1C5jG0hrvcIbt",
#   "model": "gpt-3.5-turbo-0613",
#   "created_at": 1698726834,
#   "finished_at": null,
#   "fine_tuned_model": null,
#   "organization_id": "org-n3iT5I0sZST0QX1nSKkPHmb7",
#   "result_files": [],
#   "status": "validating_files",
#   "validation_file": null,
#   "training_file": "file-A9NBysw0N5tsSopAdpkcOF6S",
#   "hyperparameters": {
#     "n_epochs": "auto"
#   },
#   "trained_tokens": null,
#   "error": null
# }
def prettyPrintFineTuneJob(job):
    durationMin = (job.finished_at - job.created_at)/60
    print("Job ID: {}".format(job.id))
    print("Status: {}".format(job.status))
    print("Model: {}".format(job.model))
    print("Duration: {:.2f} min".format(durationMin))
    print("Trained tokens: {}".format(job.trained_tokens))
    print("Tokens per minute: {:.2f}".format(job.trained_tokens / durationMin))

def tersePrintFineTuneJobHeader():
    print("ID\t\t\t\tTraining File\t\t\tStatus\t\tDuration\tTrainedTokens\tTokensPerMinute")

def tersePrintFineTuneJob(job):
    if job.finished_at is None:
        durationMin = (time.time() - job.created_at)/60
    else:
        durationMin = (job.finished_at - job.created_at)/60
    trainedTokens = 0 if job.trained_tokens is None else job.trained_tokens
    print("{}\t{}\t{}\t{:.2f}\t\t{}\t\t{:.2f}".format(job.id, job.training_file, job.status, durationMin, trainedTokens, trainedTokens / durationMin))

def makeJobsDataframe(jobs):
    jobData = []
    for job in jobs:
        if job.finished_at is None:
            durationMin = (time.time() - job.created_at)/60
        else:
            durationMin = (job.finished_at - job.created_at)/60
        trainedTokens = 0 if job.trained_tokens is None else job.trained_tokens
        jobData.append([job.id, job.training_file, job.status, durationMin, trainedTokens, trainedTokens / durationMin, job.fine_tuned_model])
    df = pd.DataFrame(jobData, columns = ['ID', 'Training File', 'Status', 'Duration', 'TrainedTokens', 'TokensPerMinute', 'FT ID'])
    return df