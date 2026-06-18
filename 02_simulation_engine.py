import covasim as cv
import pandas as pd
import numpy as np
import os

print("Loading your freshly cleaned master spreadsheet...")
processed_dir = "processed"
master_data = pd.read_csv(os.path.join(processed_dir, "calibrated_simulation_features.csv"))
total_days = len(master_data)

workplace_trend = master_data['workplace_multiplier'].tolist()
days_list = list(range(total_days))

simulation_records = []

print("\nRunning 100 simulation worlds to generate AI data... Please wait.")

for world_id in range(1, 101):

    random_virus_strength = float(np.random.uniform(0.012, 0.022))

    random_vaccine_effectiveness = float(np.random.uniform(0.5, 0.9))

    vaccine_trend = np.linspace(1.0, 1.0 - random_vaccine_effectiveness, total_days).tolist()

    settings = dict(
        pop_size=20000,
        pop_type='hybrid',
        location='usa',
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
    total_total_sick = int(results['cum_infections'][-1])

    world_summary = {
        'scenario_id': world_id,
        'virus_strength_input': random_virus_strength,
        'vaccine_speed_input': random_vaccine_effectiveness,
        'predicted_total_cases': total_total_sick,
        'predicted_peak_day': peak_day,
        'predicted_max_daily_cases': max_sick_on_one_day
    }

    simulation_records.append(world_summary)

    if world_id % 10 == 0:
        print(f"Generated {world_id}/100 virtual worlds...")

ai_training_ready_df = pd.DataFrame(simulation_records)
ai_training_ready_df.to_csv(os.path.join(processed_dir, "simulation_ml_training_set.csv"), index=False)

print("\nSUCCESS! Step 3 is fully complete!")
print("You just generated 100 complete simulation rows for your AI model!")
print("Saved at: processed/simulation_ml_training_set.csv")