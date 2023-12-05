import pandas as pd

df_real = pd.read_csv("data_files/gossipcop_real.csv")
df_fake = pd.read_csv("data_files/gossipcop_fake.csv")
gossip_df = pd.concat([df_real, df_fake])
print(gossip_df[['id', 'title']])
gossip_df[['id', 'title']].to_csv("data_files/gossipcop_title_no_ignore.tsv", sep = '\t', index = False)


