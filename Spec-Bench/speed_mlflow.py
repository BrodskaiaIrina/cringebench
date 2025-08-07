import os
import sys
import yaml
import logging
import argparse
from typing import Dict, Any

import mlflow
import mlflow.pytorch

sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))

from evaluation.speed import speed


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f'❌ Конфигурационный файл {config_path} не найден')
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f'❌ Ошибка парсинга YAML файла: {e}')
        sys.exit(1)


class MLflowSpeedLogger:
    def __init__(self, config: Dict[str, Any]):
        self.enabled = True
        mlflow_config = config.get('mlflow', {})
        
        # Настройка подключения к MLflow
        tracking_uri = mlflow_config.get('tracking_uri')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        username = mlflow_config.get('username')
        password = mlflow_config.get('password')

        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password

        self.experiment_name = mlflow_config.get('experiment_name', 'spec-bench-evaluation')
        #self.default_tags = config.get('experiment', {}).get('default_tags', {})
        
        # Создать или получить эксперимент
        try:
            self.experiment = mlflow.set_experiment(self.experiment_name)
            logging.info(f'MLflow эксперимент: {self.experiment_name}')
        except Exception as e:
            logging.error(f'Ошибка настройки MLflow эксперимента: {e}')
            self.enabled = False


    def log_speed_metrics(self, model_name: str, baseline_file: str, model_file: str, tokenizer_path: str):
        if not self.enabled:
            return

        if not os.path.exists(model_file):
            print(f"❌ Файл модели не найден: {model_file}")
            return

        if not os.path.exists(baseline_file):
            print(f"❌ Файл baseline не найден: {baseline_file}")
            return

        try:
            run_name = f"speed_analysis_{model_name}"
            run = mlflow.start_run(run_name=run_name)
            print(f"Начат анализ скорости для {model_name}")

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_file", os.path.basename(model_file))
            mlflow.log_param("baseline_file", os.path.basename(baseline_file))
            mlflow.log_param("tokenizer_path", tokenizer_path)

            tasks = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]

            for task in tasks:
                try:
                    print(f"    Анализ задачи: {task}")
                    tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list = speed(
                        model_file, baseline_file, tokenizer_path, task=task, report=False
                    )

                    mlflow.log_metric(f"{task}_tokens_per_second", tokens_per_second)
                    mlflow.log_metric(f"{task}_tokens_per_second_baseline", tokens_per_second_baseline)
                    mlflow.log_metric(f"{task}_speedup_ratio", speedup_ratio)

                    if accept_lengths_list:
                        import numpy as np

                        mean_accept_length = np.mean(accept_lengths_list)
                        mlflow.log_metric(f"{task}_mean_accept_length", mean_accept_length)
                        mlflow.log_metric(f"{task}_max_accept_length", max(accept_lengths_list))

                        acceptance_rate = sum(1 for al in accept_lengths_list if al > 0) / len(accept_lengths_list)
                        mlflow.log_metric(f"{task}_acceptance_rate", acceptance_rate)

                    print(f"    ✅ {task}: speedup {speedup_ratio:.2f}x, {tokens_per_second:.1f} tokens/sec")

                except Exception as e:
                    print(f"    ❌ Ошибка анализа {task}: {e}")

            mlflow.end_run()
            print(f"✅ Анализ скорости завершен для {model_name}")

        except Exception as e:
            print(f"❌ Ошибка логирования метрик скорости {e}")
            try:
                mlflow.end_run()
            except:
                pass


def analyze_model_speed(config_path: str, model_name: str, baseline_file: str, model_file: str, tokenizer_path: str):
    config = load_config(config_path)

    if not config:
        print("❌ Не удалось загрузить конфигурацию, пропуск MLflow логирования")
        return False

    logger = MLflowSpeedLogger(config)

    logger.log_speed_metrics(model_name, baseline_file, model_file, tokenizer_path)

    return True


def main():
    parser = argparse.ArgumentParser(description="Анализ скорости модели с логированием в MLflow")
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Путь к конфигурационному файлу"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Название модели для логирования"
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="Путь к файлу результатов модели (.jsonl)"
    )
    parser.add_argument(
        "--baseline-file",
        required=True,
        help="Путь к файлу результатов baseline модели (.jsonl)"
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Путь к токенизатору"
    )
    
    args = parser.parse_args()
    
    success = analyze_model_speed(
        config_path=args.config,
        model_name=args.model_name,
        baseline_file=args.baseline_file,
        model_file=args.model_file,
        tokenizer_path=args.tokenizer_path
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()