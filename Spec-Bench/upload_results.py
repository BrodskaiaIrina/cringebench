import os
import glob
import json
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import lakefs_sdk
    from lakefs_sdk.configuration import Configuration
    from lakefs_sdk.api_client import ApiClient
    from lakefs_sdk.api import objects_api, branches_api, commits_api
except ImportError:
    print('❌ LakeFS SDK не установлен. Установите его командой: pip install lakefs-sdk')
    sys.exit(1)

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    print('❌ MLflow не установлен. Установите его командой: pip install mlflow')
    MLFLOW_AVAILABLE = False


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


def setup_logging(config: Dict[str, Any]):
    log_config = config.get('logging', {})

    # Создать директорию логов
    log_file = log_config.get('file', 'logs/benchmark_execution.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Настроить уровень логирования
    level = getattr(logging, log_config.get('level', 'INFO').upper())

    # Настроить форматтер
    formatter = logging.Formatter(
        log_config.get('format', '[%(asctime)s] %(levelname)s - %(message)s')
    )

    # Настроить логгер
    logger = logging.getLogger()
    logger.setLevel(level)

    # Консольный хэндлер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый хэндлер с ротацией
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=log_config.get('max_file_size', 100) * 1024 * 1024, # MB в байты
        backupCount=log_config.get('backup_count', 5)
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class LakeFSUploader:
    def __init__(self, config: Dict[str, Any]):
        lakefs_config = config['lakefs']
        
        self.endpoint = lakefs_config['endpoint']
        self.repository = lakefs_config['repository']
        self.branch_prefix = lakefs_config.get('branch_prefix', 'experiment')

        access_key = lakefs_config.get('access_key')
        secret_key = lakefs_config.get('secret_key')

        if not access_key or not secret_key:
            raise ValueError("Access key и secret key обязательны для подключения к LakeFS")

        configuration = Configuration(
            host=self.endpoint,
            username=access_key,
            password=secret_key
        )
        
        api_client = ApiClient(configuration)
        self.objects_api = objects_api.ObjectsApi(api_client)
        self.branches_api = branches_api.BranchesApi(api_client)
        self.commits_api = commits_api.CommitsApi(api_client)
        
    def upload_file(self, repo, branch, local_path, remote_path):
        try:
            import os
            file_size = os.path.getsize(local_path)
            print(f"Uploading {local_path} ({file_size} bytes) -> {remote_path}")

            with open(local_path, 'rb') as f:
                file_content = f.read()

            #import io 
            #file_stream = io.BytesIO(file_content)
            #print(type(file_content))

            self.objects_api.upload_object(
                repository=repo,
                branch=branch,
                path=remote_path,
                content=file_content
            )
            
            print(f'✅ Uploaded {local_path} -> {remote_path}')
            return True
        except Exception as e:
            print(f'❌ Failed to upload {local_path}: {e}')
            #print(f'❌ Error type: {type(e).__name__}')
            return False
            
    def create_branch(self, repo, branch_name, source_branch='main'):
        try:
            self.branches_api.create_branch(
                repository=repo,
                branch_creation=lakefs_sdk.BranchCreation(
                    name=branch_name,
                    source=source_branch
                )
            )
            print(f'✅ Created branch: {branch_name}')
            return True
        except Exception as e:
            print(f'⚠️ Branch creation failed: {e}')
            return False
            
    def commit_changes(self, repo, branch, message):
        try:
            commit = lakefs_sdk.CommitCreation(message=message)
            result = self.commits_api.commit(
                repository=repo,
                branch=branch,
                commit_creation=commit
            )
            print(f'✅ Committed changes: {result.id}')
            return result.id
        except Exception as e:
            print(f'❌ Failed to commit: {e}')
            return None

    def check_connection(self):
        try:
            # Пытаемся получить список веток для проверки подключения
            from lakefs_sdk.api import repositories_api
            repos_api = repositories_api.RepositoriesApi(self.objects_api.api_client)
            repos_api.list_repositories()
            return True
        except Exception as e:
            logging.error(f'Не удается подключиться к LakeFS: {e}')
            return False


class MLflowLogger:
    def __init__(self, config: Dict[str, Any]):
        if not MLFLOW_AVAILABLE:
            logging.warning('MLflow недоступен, пропуск логирования')
            self.enabled = False
            return
            
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
        self.default_tags = config.get('experiment', {}).get('default_tags', {})
        
        # Создать или получить эксперимент
        try:
            self.experiment = mlflow.set_experiment(self.experiment_name)
            logging.info(f'MLflow эксперимент: {self.experiment_name}')
        except Exception as e:
            logging.error(f'Ошибка настройки MLflow эксперимента: {e}')
            self.enabled = False
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> Optional[str]:
        if not self.enabled:
            return None
            
        try:
            # Объединить теги
            all_tags = self.default_tags.copy()
            if tags:
                all_tags.update(tags)
                
            run = mlflow.start_run(run_name=run_name, tags=all_tags)
            logging.info(f'Начат MLflow run: {run.info.run_id}')
            return run.info.run_id
        except Exception as e:
            logging.error(f'Ошибка создания MLflow run: {e}')
            return None
    
    def log_params(self, params: Dict[str, Any]):
        if not self.enabled:
            return
            
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            logging.error(f'Ошибка логирования параметров в MLflow: {e}')
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        if not self.enabled:
            return
            
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logging.error(f'Ошибка логирования метрик в MLflow: {e}')
    
    def log_artifacts(self, artifact_path: str, local_path: str = None):
        if not self.enabled:
            return
            
        try:
            if local_path:
                mlflow.log_artifact(local_path, artifact_path)
            else:
                mlflow.log_artifacts(artifact_path)
        except Exception as e:
            logging.error(f'Ошибка логирования артефактов в MLflow: {e}')
    
    def log_benchmark_results(self, results_file: str, benchmark_name: str):
        if not self.enabled or not os.path.exists(results_file):
            return
            
        try:
            # Парсинг результатов бенчмарка
            metrics = self._parse_benchmark_results(results_file, benchmark_name)
            
            if metrics:
                self.log_metrics(metrics)
                logging.info(f'Залогированы метрики для {benchmark_name}: {len(metrics)} значений')
        except Exception as e:
            logging.error(f'Ошибка логирования результатов {benchmark_name}: {e}')


    def _parse_benchmark_results(self, results_file: str, benchmark_name: str) -> Dict[str, float]:
        metrics = {}
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            if not results:
                return metrics
            
            # Извлечение базовых метрик
            total_questions = len(results)
            metrics[f'{benchmark_name}_total_questions'] = total_questions
            
            # Анализ времени выполнения
            wall_times = []
            decoding_steps = []
            new_tokens = []
            accept_lengths = []
            
            for result in results:
                choices = result.get('choices', [{}])
                if choices:
                    choice = choices[0]  # Берем первый choice
                    
                    # Время выполнения
                    if 'wall_time' in choice:
                        wall_times.extend(choice['wall_time'])
                    
                    # Количество шагов декодирования
                    if 'decoding_steps' in choice:
                        decoding_steps.extend(choice['decoding_steps'])
                    
                    # Новые токены
                    if 'new_tokens' in choice:
                        new_tokens.extend(choice['new_tokens'])
                    
                    # Принятые длины (для speculative decoding)
                    if 'accept_lengths' in choice:
                        accept_lengths.extend(choice['accept_lengths'])
            
            # Вычисление статистик
            if wall_times:
                metrics[f'{benchmark_name}_avg_wall_time'] = sum(wall_times) / len(wall_times)
                metrics[f'{benchmark_name}_total_wall_time'] = sum(wall_times)
                metrics[f'{benchmark_name}_max_wall_time'] = max(wall_times)
                metrics[f'{benchmark_name}_min_wall_time'] = min(wall_times)
            
            if decoding_steps:
                metrics[f'{benchmark_name}_avg_decoding_steps'] = sum(decoding_steps) / len(decoding_steps)
                metrics[f'{benchmark_name}_total_decoding_steps'] = sum(decoding_steps)
            
            if new_tokens:
                metrics[f'{benchmark_name}_avg_new_tokens'] = sum(new_tokens) / len(new_tokens)
                metrics[f'{benchmark_name}_total_new_tokens'] = sum(new_tokens)
                
                # Токены в секунду
                if wall_times and len(wall_times) == len(new_tokens):
                    tokens_per_sec = [nt / wt if wt > 0 else 0 for nt, wt in zip(new_tokens, wall_times)]
                    metrics[f'{benchmark_name}_avg_tokens_per_sec'] = sum(tokens_per_sec) / len(tokens_per_sec)
                    metrics[f'{benchmark_name}_max_tokens_per_sec'] = max(tokens_per_sec)
            
            if accept_lengths:
                metrics[f'{benchmark_name}_avg_accept_length'] = sum(accept_lengths) / len(accept_lengths)
                metrics[f'{benchmark_name}_max_accept_length'] = max(accept_lengths)
                
                # Acceptance rate для speculative decoding
                if len(accept_lengths) > 0:
                    acceptance_rate = sum(1 for al in accept_lengths if al > 0) / len(accept_lengths)
                    metrics[f'{benchmark_name}_acceptance_rate'] = acceptance_rate
        
        except Exception as e:
            logging.error(f'Ошибка парсинга результатов {results_file}: {e}')
        
        return metrics
    
    def end_run(self):
        if self.enabled:
            try:
                mlflow.end_run()
                logging.info('MLflow run завершен')
            except Exception as e:
                logging.error(f'Ошибка завершения MLflow run: {e}')


def sanitize_branch_name(name: str) -> str:
    import re

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    if safe_name.startswith('-'):
        safe_name = 'model_' + safe_name[1:]
    return safe_name


def get_system_info():
    '''Получить информацию о системе для метаданных'''
    import platform
    import torch
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info


def create_metadata_file(results_dir, uploaded_files, config):
    '''Создать файл с метаданными эксперимента'''
    metadata = {
        'experiment_info': get_system_info(),
        'results_directory': str(results_dir),
        'uploaded_files': uploaded_files,
        'file_count': len(uploaded_files),
        'config_snapshot': {
            'lakefs_endpoint': config['lakefs']['endpoint'],
            'lakefs_repository': config['lakefs']['repository'],
            'mlflow_tracking_uri': config.get('mlflow', {}).get('tracking_uri'),
            'experiment_tags': config.get('experiment', {}).get('default_tags', {})
        }
    }
    
    # Анализ результатов для метаданных
    total_size = 0
    for file_path in uploaded_files:
        local_path = os.path.join(results_dir, os.path.basename(file_path.split('/')[-1]))
        if os.path.exists(local_path):
            total_size += os.path.getsize(local_path)
    
    metadata['total_size_bytes'] = total_size
    metadata['total_size_mb'] = round(total_size / (1024 * 1024), 2)
    
    return metadata


def upload_single_result(config: Dict[str, Any], result_file: str, model_name: str = None):
    """Загрузить один файл с результатами в LakeFS"""
    
    if not os.path.exists(result_file):
        logging.error(f"Файл с результатами не найден: {result_file}")
        return False
    
    logging.info(f"🚀 Загрузка результата в LakeFS: {os.path.basename(result_file)}")
    
    # Инициализация LakeFS клиента
    try:
        uploader = LakeFSUploader(config)
    except Exception as e:
        logging.error(f"Ошибка инициализации LakeFS клиента: {e}")
        return False
    
    # Проверка подключения
    if not uploader.check_connection():
        return False
    
    # Создать ветку для этого результата
    timestamp = datetime.now().strftime("%Y%m%d")
    if model_name:
        safe_model_name = sanitize_branch_name(model_name)
        experiment_branch = f"{uploader.branch_prefix}_{safe_model_name}_{timestamp}"
    else:
        experiment_branch = f"{uploader.branch_prefix}_single_{timestamp}"
    
    # Попытка создать ветку
    uploader.create_branch(uploader.repository, experiment_branch, "main")
    
    # Загрузить файл
    filename = os.path.basename(result_file)
    remote_path = f"results/{timestamp}/{filename}"
    
    if uploader.upload_file(uploader.repository, experiment_branch, result_file, remote_path):
        # Создать коммит
        commit_message = f"Single result upload: {filename}\nTimestamp: {timestamp}"
        if model_name:
            commit_message += f"\nModel: {model_name}"
        
        commit_id = uploader.commit_changes(uploader.repository, experiment_branch, commit_message)
        
        if commit_id:
            logging.info(f"✅ Результат загружен успешно!")
            logging.info(f"📊 Ветка: {experiment_branch}")
            logging.info(f"📝 Коммит: {commit_id}")
            logging.info(f"🌐 Просмотр: {uploader.endpoint}/repositories/{uploader.repository}/objects?ref={experiment_branch}")
            return True
        else:
            logging.error("❌ Ошибка создания коммита")
            return False
    else:
        logging.error(f"❌ Ошибка загрузки файла: {result_file}")
        return False


def upload_benchmark_results(config: Dict[str, Any], results_dir: str = None, mlflow_logger: MLflowLogger = None):
    '''Загрузить все результаты бенчмарков в LakeFS и логировать в MLflow'''
    
    if results_dir is None:
        results_dir = config.get('benchmark', {}).get('results_dir', 'data/spec_bench/model_answer')
    
    logging.info('🚀 Начинаю загрузку результатов в LakeFS...')
    
    # Инициализация LakeFS клиента
    try:
        uploader = LakeFSUploader(config)
    except Exception as e:
        logging.error(f'Ошибка инициализации LakeFS клиента: {e}')
        return False
    
    # Проверка подключения
    if not uploader.check_connection():
        return False
    
    # Проверка директории результатов
    if not os.path.exists(results_dir):
        logging.error(f'Директория результатов не найдена: {results_dir}')
        return False
    
    # Найти все файлы результатов
    pattern = os.path.join(results_dir, '*.jsonl')
    result_files = glob.glob(pattern)
    
    if not result_files:
        logging.warning(f'Не найдено файлов результатов в {results_dir}')
        logging.info(f'Ищу файлы по паттерну: {pattern}')
        return False
    
    logging.info(f'📁 Найдено {len(result_files)} файлов результатов')
    for f in result_files:
        logging.info(f'  - {os.path.basename(f)}')
    
    # Создать ветку для этого эксперимента
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_branch = f'{uploader.branch_prefix}_{timestamp}'
    
    # Попытка создать ветку
    uploader.create_branch(uploader.repository, experiment_branch, 'main')
    
    # Логирование в MLflow
    if mlflow_logger:
        run_tags = {
            'experiment_timestamp': timestamp,
            'lakefs_branch': experiment_branch,
            'lakefs_repository': uploader.repository,
            'results_count': str(len(result_files))
        }
        mlflow_logger.start_run(run_name=f'benchmark_{timestamp}', tags=run_tags)
        
        # Логировать системную информацию
        system_info = get_system_info()
        mlflow_logger.log_params(system_info)
    
    # Загрузить файлы
    uploaded_files = []
    failed_files = []
    
    for local_file in result_files:
        filename = os.path.basename(local_file)
        remote_path = f'results/{timestamp}/{filename}'
        
        if uploader.upload_file(uploader.repository, experiment_branch, local_file, remote_path):
            uploaded_files.append(remote_path)
            
            # Логировать метрики бенчмарка в MLflow
            if mlflow_logger:
                benchmark_name = filename.replace('.jsonl', '').replace('-', '_')
                mlflow_logger.log_benchmark_results(local_file, benchmark_name)
        else:
            failed_files.append(local_file)
    
    # Создать файл метаданных
    if uploaded_files:
        metadata = create_metadata_file(results_dir, uploaded_files, config)
        metadata_path = f'results/{timestamp}/experiment_metadata.json'
        
        # Сохранить метаданные локально и загрузить
        temp_metadata_file = os.path.join(results_dir, 'temp_metadata.json')
        with open(temp_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if uploader.upload_file(uploader.repository, experiment_branch, temp_metadata_file, metadata_path):
            uploaded_files.append(metadata_path)
        
        # Логировать метаданные как артефакт в MLflow
        if mlflow_logger:
            mlflow_logger.log_artifacts('metadata', temp_metadata_file)
        
        # Удалить временный файл
        os.remove(temp_metadata_file)
    
    # Создать коммит
    if uploaded_files:
        commit_message = f'Benchmark results from {timestamp}\n\n'
        commit_message += f'Uploaded {len(uploaded_files)} files:\n'
        commit_message += '\n'.join(f'- {f}' for f in uploaded_files)
        
        if failed_files:
            commit_message += f'\n\nFailed to upload {len(failed_files)} files:\n'
            commit_message += '\n'.join(f'- {os.path.basename(f)}' for f in failed_files)
        
        commit_id = uploader.commit_changes(uploader.repository, experiment_branch, commit_message)
        
        if commit_id:
            logging.info(f'🎉 Успешно загружено {len(uploaded_files)} файлов!')
            logging.info(f'📊 Ветка: {experiment_branch}')
            logging.info(f'📝 Коммит: {commit_id}')
            logging.info(f'🌐 Просмотр: {uploader.endpoint}/repositories/{uploader.repository}/objects?ref={experiment_branch}')
            
            # Логировать итоговые метрики в MLflow
            if mlflow_logger:
                final_metrics = {
                    'uploaded_files_count': len(uploaded_files),
                    'failed_files_count': len(failed_files),
                    'total_files_count': len(result_files),
                    'upload_success_rate': len(uploaded_files) / len(result_files) if result_files else 0
                }
                mlflow_logger.log_metrics(final_metrics)
                mlflow_logger.log_params({
                    'lakefs_commit_id': commit_id,
                    'lakefs_branch': experiment_branch,
                    'lakefs_repository': uploader.repository
                })
            
            if failed_files:
                logging.warning(f'Не удалось загрузить {len(failed_files)} файлов:')
                for f in failed_files:
                    logging.warning(f'  - {os.path.basename(f)}')
            
            return True
    
    logging.error('Не удалось загрузить файлы в LakeFS')
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузить результаты бенчмарков в LakeFS и MLflow')
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Путь к конфигурационному файлу (по умолчанию: config.yaml)'
    )
    parser.add_argument(
        '--results-dir', 
        help='Директория с результатами (переопределяет конфигурацию)'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Отключить логирование в MLflow'
    )
    parser.add_argument(
        "--single-file",
        help="Загрузить один конкретный файл вместо всей директории"
    )
    parser.add_argument(
        "--model-name",
        help="Название модели (для загрузки одного файла)"
    )
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = load_config(args.config)
    
    # Настроить логирование
    logger = setup_logging(config)
    
    try:
        if args.single_file:
            # Загрузка одного файла
            success = upload_single_result(
                config=config,
                result_file=args.single_file,
                model_name=args.model_name
            )
        else:
        # Инициализация MLflow логгера
            mlflow_logger = None
            if not args.no_mlflow:
                mlflow_logger = MLflowLogger(config)
            
            # Загрузить результаты
            success = upload_benchmark_results(
                config=config,
                results_dir=args.results_dir,
                mlflow_logger=mlflow_logger
            )
            
            # Завершить MLflow run
            if mlflow_logger:
                mlflow_logger.end_run()
        
        if success:
            logging.info('🎉 Все операции завершены успешно!')
        else:
            logging.error('❌ Операции завершены с ошибками')
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logging.info('Операция прервана пользователем')
        if mlflow_logger:
            mlflow_logger.end_run()
        sys.exit(130)
    except Exception as e:
        logging.error(f'Неожиданная ошибка: {e}')
        if mlflow_logger:
            mlflow_logger.end_run()
        sys.exit(1)


if __name__ == '__main__':
    main()