import pandas as pd
import os

print(" Finding your files in your raw folder...")

raw_dir = "raw"
processed_dir = "processed"

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

all_files = os.listdir(raw_dir)

case_file = next((f for f in all_files if 'case' in f.lower()), None)
vac_file = next((f for f in all_files if 'vac' in f.lower()), None)
mob_file = next((f for f in all_files if 'mobil' in f.lower()), None)

print(f" Found Case File: {case_file}")
print(f" Found Vaccine File: {vac_file}")
print(f" Found Mobility File: {mob_file}")

df_cases = pd.read_csv(os.path.join(raw_dir, case_file))
df_vaccines = pd.read_csv(os.path.join(raw_dir, vac_file))
df_mobility = pd.read_csv(os.path.join(raw_dir, mob_file), low_memory=False)

df_cases.columns = df_cases.columns.str.lower()
df_vaccines.columns = df_vaccines.columns.str.lower()
df_mobility.columns = df_mobility.columns.str.lower()

def find_col(df, options):
    for opt in options:
        if opt in df.columns:
            return opt
    return df.columns[0]

case_geo = find_col(df_cases, ['location', 'country', 'country_region', 'entity'])
vac_geo = find_col(df_vaccines, ['location', 'country', 'country_region', 'entity'])
mob_geo = find_col(df_mobility, ['country_region', 'location', 'country'])

case_date = find_col(df_cases, ['date', 'day'])
vac_date = find_col(df_vaccines, ['date', 'day'])
mob_date = find_col(df_mobility, ['date', 'day'])

df_cases[case_date] = pd.to_datetime(df_cases[case_date])
df_vaccines[vac_date] = pd.to_datetime(df_vaccines[vac_date])
df_mobility[mob_date] = pd.to_datetime(df_mobility[mob_date])

print(" Filtering for the United States (Year 2020)...")

df_cases_clean = df_cases[(df_cases[case_geo].isin(['United States', 'US'])) & (df_cases[case_date].dt.year == 2020)].copy()
df_vaccines_clean = df_vaccines[(df_vaccines[vac_geo].isin(['United States', 'US'])) & (df_vaccines[vac_date].dt.year == 2020)].copy()
df_mobility_clean = df_mobility[(df_mobility[mob_geo].isin(['United States', 'US'])) & (df_mobility[mob_date].dt.year == 2020)].copy()

df_cases_clean = df_cases_clean.rename(columns={case_date: 'date'})
df_vaccines_clean = df_vaccines_clean.rename(columns={vac_date: 'date'})
df_mobility_clean = df_mobility_clean.rename(columns={mob_date: 'date'})

case_metric = find_col(df_cases_clean, ['new_cases', 'cases'])
tot_case_metric = find_col(df_cases_clean, ['total_cases', 'cumulative_cases'])
df_cases_clean = df_cases_clean[['date', case_metric, tot_case_metric]]

vac_metric = find_col(df_vaccines_clean, ['people_fully_vaccinated', 'total_vaccinations', 'vaccinations'])
df_vaccines_clean = df_vaccines_clean[['date', vac_metric]]

mob_metric = find_col(df_mobility_clean, ['workplaces_percent_change_from_baseline'])
df_mobility_daily = df_mobility_clean.groupby('date')[[mob_metric]].mean().reset_index()

print(" Turning mobility percentages into multipliers...")
df_mobility_daily['workplace_multiplier'] = 1 + (df_mobility_daily[mob_metric] / 100)

print(" Merging all files into one master timeline spreadsheet...")
df_epi = pd.merge(df_cases_clean, df_vaccines_clean, on='date', how='outer')
master_spreadsheet = pd.merge(df_epi, df_mobility_daily[['date', 'workplace_multiplier']], on='date', how='inner')

master_spreadsheet = master_spreadsheet.fillna(0)

master_spreadsheet.to_csv(os.path.join(processed_dir, "calibrated_simulation_features.csv"), index=False)

print("\n SUCCESS! Step 2 is fully complete!")
print("Your clean matched spreadsheet is saved at: processed/calibrated_simulation_features.csv")