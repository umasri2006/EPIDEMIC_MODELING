import covasim as cv
import pandas as pd
import numpy as np
import os

print("INITIALIZING EXPANDED GLOBAL SIMULATION ENGINE (10 COUNTRIES)...")

raw_dir = "raw"
processed_dir = "processed"

all_files = os.listdir(raw_dir)
age_file = next((f for f in all_files if 'age-group' in f.lower() and f.endswith('.csv')), None)
pop_file = next((f for f in all_files if 'api_sp.pop.totl' in f.lower() and f.endswith('.csv')), None)
master_features_file = os.path.join(processed_dir, "calibrated_simulation_features.csv")

print("Reading global age demographics and total population frameworks...")
df_age_global = pd.read_csv(os.path.join(raw_dir, age_file))
df_pop_global = pd.read_csv(os.path.join(raw_dir, pop_file), skiprows=4)

df_age_global.columns = df_age_global.columns.str.lower()
df_pop_global.columns = df_pop_global.columns.str.lower()

countries_config = {
    'United States': {'covasim_code': 'usa', 'pop_wb_name': 'united states'},
    'United Kingdom': {'covasim_code': 'uk', 'pop_wb_name': 'united kingdom'},
    'Canada': {'covasim_code': 'canada', 'pop_wb_name': 'canada'},
    'Germany': {'covasim_code': 'germany', 'pop_wb_name': 'germany'},
    'France': {'covasim_code': 'france', 'pop_wb_name': 'france'},
    'Italy': {'covasim_code': 'italy', 'pop_wb_name': 'italy'},
    'Spain': {'covasim_code': 'spain', 'pop_wb_name': 'spain'},
    'Japan': {'covasim_code': 'japan', 'pop_wb_name': 'japan'},
    'Brazil': {'covasim_code': 'brazil', 'pop_wb_name': 'brazil'},
    'India': {'covasim_code': 'india', 'pop_wb_name': 'india'}
}

global_simulation_records = []

for country_real_name, config in countries_config.items():
    print(f"\n=========================================")
    print(f"PROCESSING COUNTRY: {country_real_name.upper()}")
    print(f"=========================================")

    try:
        country_pop_row = df_pop_global[df_pop_global['country_name'] == config['pop_wb_name']]
        real_total_population = int(country_pop_row['2020'].values[0])
        print(f"Real 2020 Population scale located: {real_total_population:,}")
    except:
        real_total_population = 60_000_000

    try:
        country_age_row = df_age_global[(df_age_global['entity'] == country_real_name) & (df_age_global['year'] == 2020)]
        age_0_14 = float(country_age_row['0-14 years'].values[0])
        age_15_64 = float(country_age_row['15-64 years'].values[0])
        age_65_plus = float(country_age_row['65+ years'].values[0])

        total_pct = age_0_14 + age_15_64 + age_65_plus
        my_age_brackets = {
            '0-14': age_0_14 / total_pct,
            '15-64': age_15_64 / total_pct,
            '65+': age_65_plus / total_pct
        }

        print(f"Age mix ratio -> Children: {age_0_14:.1f}%, Adults: {age_15_64:.1f}%, Elderly: {age_65_plus:.1f}%")
    except:
        my_age_brackets = {'0-19': 0.25, '20-64': 0.60, '65+': 0.15}

    master_data = pd.read_csv(master_features_file)
    total_days = len(master_data)
    workplace_trend = master_data['workplace_multiplier'].tolist()
    days_list = list(range(total_days))

    print(f"Simulating 40 unique outbreaks for {country_real_name}...")

    for world_id in range(1, 41):
        random_virus_strength = float(np.random.uniform(0.012, 0.024))
        random_vaccine_effectiveness = float(np.random.uniform(0.4, 0.9))
        vaccine_trend = np.linspace(1.0, 1.0 - random_vaccine_effectiveness, total_days).tolist()

        settings = dict(
            pop_size=12000,
            pop_type='hybrid',
            location=config['covasim_code'],
            n_days=total_days,
            beta=random_virus_strength
        )

        actions = [
            cv.change_beta(days=days_list, changes=workplace_trend, layers='w'),
            cv.change_beta(days=days_list, changes=vaccine_trend)
        ]

        sim = cv.Sim(settings, interventions=actions)
        sim.run(verbose=0)

        results = sim.results
        peak_day = int(np.argmax(results['new_infections']))
        max_sick_on_one_day = int(results['new_infections'][peak_day])
        total_sick = int(results['cum_infections'][-1])

        global_simulation_records.append({
            'country': country_real_name,
            'country_total_population': real_total_population,
            'pct_population_elderly': my_age_brackets['65+'],
            'virus_strength_input': random_virus_strength,
            'vaccine_effectiveness_input': random_vaccine_effectiveness,
            'predicted_total_cases': total_sick,
            'predicted_peak_day': peak_day,
            'predicted_max_daily_cases': max_sick_on_one_day
        })

        if world_id % 20 == 0:
            print(f"   Progress: {world_id}/40 worlds calculated...")

df_global_ml_set = pd.DataFrame(global_simulation_records)
df_global_ml_set.to_csv(os.path.join(processed_dir, "global_multi_country_ml_dataset.csv"), index=False)

print("\nGLOBAL MULTI-COUNTRY DATA MERGE COMPLETE!")
print(f"Total dataset generated: {len(df_global_ml_set)} highly detailed training rows across 10 countries.")
print("Saved directly to: processed/global_multi_country_ml_dataset.csv")