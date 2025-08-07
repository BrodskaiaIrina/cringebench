
try:
    import sys 
    import yaml
    import argparse
except Exception as e:
    print("Ошибка:", e)

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            print('ok')
            return yaml.safe_load(f)
    except Exception as e:
        print("Ошибка загрузки конфигурации:", e)
        return None


def test_lakefs_connection(config):
    print("Проверка подключения к LakeFS")

    try:
        import lakefs_sdk
        from lakefs_sdk.configuration import Configuration
        from lakefs_sdk.api_client import ApiClient
        from lakefs_sdk.api import repositories_api

        lakefs_config = config["lakefs"]

        configuration = Configuration(
            host=lakefs_config['endpoint'],
            username=lakefs_config['access_key'],
            password=lakefs_config['secret_key']
        )

        api_client = ApiClient(configuration)
        repos_api = repositories_api.RepositoriesApi(api_client)

        repos = repos_api.list_repositories()
        print("Подключение к LakeFS успешно")
        print("Найдено репозиториев:", {len(repos.results) if repos.results else 0})

        target_repo = lakefs_config['repository']
        repo_exists = any(repo.id == target_repo for repo in (repos.results or []))
        if repo_exists:
            print("Репозиторий '{target_repo}' найден")
        else:
            print("Репозиторий '{target_repo}' не найден")
            print("Доступные репозитории:", {repo.id for repo in (repos.results or [])})

        return True, repo_exists

    except ImportError:
        print("lakefs-sdk не установлен: pip install lakefs-sdk")
        return False, False
    except Exception as e:
        print("Ошибка подключения к lakefs:", e)
        return False, False


def test_mlflow_connection(config):
    print("Проверка подключения к MLFlow")

    try:
        import mlflow

        mlflow_config = config["mlflow"]

        tracking_uri = mlflow_config.get('tracking_uri')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print("  Tracking uri:", tracking_uri)

        import os
        username = mlflow_config.get('username')
        password = mlflow_config.get('password')

        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password

        experiments = mlflow.search_experiments()
        print("Подключение к MLFlow успешно")
        print("Найдено экспериментов:", {len(experiments)})

        experiment_name = mlflow_config.get('experiment_name', "spec-bench-evaluation")
        existing_experiment = None

        for exp in experiments:
            if exp.name == experiment_name:
                existing_experiment = exp
                break

        if existing_experiment:
            print("Эксперимент '{existing_experiment}' найден")
            print(f"ID: {existing_experiment.experiment_id}")
        else:
            print("Эксперимент '{existing_experiment}' будет создан автоматически")

        return True, existing_experiment is not None

    except ImportError:
        print("mlflow не установлен: pip install mlflow")
        return False, False
    except Exception as e:
        print("Ошибка подключения к mlflow:", e)
        return False, False




parser = argparse.ArgumentParser(description="Проверка подключения к lakefs, mlfolw")
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()

config = load_config(args.config)
#print(config)
if not config:
    print('AAAAAA')
    sys.exit(1)

print("Конфигурация загружена из:", args.config)

lakefs_ok, lakefs_repo_ok = test_lakefs_connection(config)
mlflow_ok, mlflow_repo_ok = test_mlflow_connection(config)

print(f"  LakeFS подключение: {'OK' if lakefs_ok else 'ОШИБКА'}")
print(f"  LakeFS репозиторий: {'OK' if lakefs_repo_ok else 'ОШИБКА'}")
print(f"  MLFlow подключение: {'OK' if mlflow_ok else 'ОШИБКА'}")
print(f"  MLFlow эксперимент: {'OK' if mlflow_repo_ok else 'ОШИБКА'}")