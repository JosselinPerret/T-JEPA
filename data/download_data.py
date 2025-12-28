import openml

# Récupère la suite de tâches
benchmark_suite = openml.study.get_study('OpenML-CC18', 'tasks')

# Exemple pour récupérer le premier dataset de la suite
task_id = benchmark_suite.tasks[0]
dataset = openml.tasks.get_task(task_id).get_dataset()
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
print(f"Dataset chargé : {dataset.name}, Shape : {X.shape}")