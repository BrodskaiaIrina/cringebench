#!/bin/bash

check_config() {
    local config_file="${1:-config.yaml}"
    
    if [[ ! -f "$config_file" ]]; then
        log "❌ Конфигурационный файл $config_file не найден"
        log "Создайте файл на основе config.yaml.example"
        return 1
    fi
    
    return 0
}

# Конфигурация моделей
Vicuna_PATH=./models/vicuna-7b-v1.3
Eagle_PATH=./models/EAGLE-Vicuna-7B-v1.3
Eagle3_PATH=./models/EAGLE3-Vicuna1.3-13B
Medusa_PATH=./models/medusa-vicuna-7b-v1.3
Hydra_PATH=./models/hydra-vicuna-7b-v1.3
Drafter_PATH=./models/vicuna-68m
Space_PATH=./models/vicuna-v1.3-7b-space
datastore_PATH=./model/rest/datastore/datastore_chat_large.idx

Vicuna_PATH_tknz="lmsys/vicuna-7b-v1.3"

# Параметры эксперимента
MODEL_NAME=vicuna-7b-v1.3
TEMP=0.0
GPU_DEVICES=0
bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# Создать директорию для результатов если её нет
RESULTS_DIR="data/${bench_NAME}/model_answer"
mkdir -p "${RESULTS_DIR}"

# Логирование
LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Логирование в файл: ${LOG_FILE}"

# Функция для логирования
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}


# Функция для запуска одного бенчмарка с пошаговым логированием
run_benchmark_with_logging() {
    local cmd="$1"
    local name="$2"
    local model_id="$3"
    local config_file="$4"
    local start_time
    local end_time
    local duration
    
    log "🚀 Запуск бенчмарка: ${name}"
    start_time=$(date +%s)
    
    # Выполнить команду
    eval $cmd
    local exit_code=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log "✅ ${name} завершен успешно за ${duration} секунд"
        
        # Найти файл результата для этой модели
        local result_file="${RESULTS_DIR}/${model_id}.jsonl"
        if [[ -f "$result_file" ]]; then
            log "📁 Найден файл результата: $(basename "$result_file")"
            
            # Загрузить результат в LakeFS
            upload_single_result "$config_file" "$result_file" "$model_id"
            
            # Найти baseline файл для анализа скорости
            local baseline_file="${RESULTS_DIR}/${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}.jsonl"
            if [[ -f "$baseline_file" ]]; then
                log "📊 Найден baseline для сравнения: $(basename "$baseline_file")"
                
                # Определить путь к токенизатору (используем Vicuna_PATH)
                local tokenizer_path="$Vicuna_PATH"
                
                # Выполнить анализ скорости и логирование в MLflow
                analyze_speed "$config_file" "$model_id" "$result_file" "$baseline_file" "$tokenizer_path"
            else
                log "⚠️ Baseline файл не найден для анализа скорости: $(basename "$baseline_file")"
                log "   Убедитесь, что baseline (vanilla) модель была запущена первой"
            fi
        else
            log "❌ Файл результата не найден: $(basename "$result_file")"
        fi
        
        return 0
    else
        log "❌ ${name} завершен с ошибкой (код: ${exit_code}) за ${duration} секунд"
        return $exit_code
    fi
}


# Функция для запуска одного бенчмарка (старая версия без логирования)
run_benchmark() {
    local cmd="$1"
    local name="$2"
    local start_time
    local end_time
    local duration
    
    log "🚀 Запуск бенчмарка: ${name}"
    start_time=$(date +%s)
    
    # Выполнить команду
    eval $cmd
    local exit_code=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log "✅ ${name} завершен успешно за ${duration} секунд"
        return 0
    else
        log "❌ ${name} завершен с ошибкой (код: ${exit_code}) за ${duration} секунд"
        return $exit_code
    fi
}

# Функция для обработки модели (проверка существующего результата или запуск inference)
process_model() {
    local model_name="$1"
    local model_id="$2"
    local result_file="$3"
    local inference_cmd="$4"
    local config_file="$5"
    local run_speed_analysis="${6:-true}"  # по умолчанию true
    
    if [[ -f "$result_file" ]]; then
        log "📁 Найден существующий результат: $model_name"
        log "⚡ Пропуск inference, загрузка и анализ существующего файла"
        upload_single_result "$config_file" "$result_file" "$model_id"
        
        # Анализ скорости для не-baseline моделей
        if [[ "$run_speed_analysis" == "true" ]]; then
            local baseline_file="${RESULTS_DIR}/${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}.jsonl"
            if [[ -f "$baseline_file" ]]; then
                analyze_speed "$config_file" "$model_id" "$result_file" "$baseline_file" "$Vicuna_PATH_tknz"
            else
                log "⚠️ Baseline файл не найден для анализа скорости"
            fi
        fi
    else
        log "🚀 Запуск: $model_name"
        eval "$inference_cmd"
        if [ $? -eq 0 ]; then
            log "✅ $model_name завершен успешно"
            upload_single_result "$config_file" "$result_file" "$model_id"
            
            # Анализ скорости для не-baseline моделей
            if [[ "$run_speed_analysis" == "true" ]]; then
                local baseline_file="${RESULTS_DIR}/${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}.jsonl"
                if [[ -f "$baseline_file" ]]; then
                    analyze_speed "$config_file" "$model_id" "$result_file" "$baseline_file" "$Vicuna_PATH_tknz"
                else
                    log "⚠️ Baseline файл не найден для анализа скорости"
                fi
            fi
        else
            log "❌ $model_name завершен с ошибкой"
        fi
    fi
}


# Функция для загрузки одного результата в LakeFS
upload_single_result() {
    local config_file="$1"
    local result_file="$2"
    local model_name="$3"
    
    if check_config "$config_file"; then
        log "📤 Загрузка результата в LakeFS: $(basename "$result_file")"
        python3 upload_results.py --config "$config_file" --single-file "$result_file" --model-name "$model_name"
        local upload_exit_code=$?
        
        if [ $upload_exit_code -eq 0 ]; then
            log "✅ Результат успешно загружен в LakeFS"
        else
            log "❌ Ошибка загрузки (код: ${upload_exit_code})"
        fi
        return $upload_exit_code
    else
        log "⚠️ Пропуск загрузки (конфигурация недоступна)"
        return 1
    fi
}

# Функция для анализа скорости модели
analyze_speed() {
    local config_file="$1"
    local model_name="$2"
    local model_file="$3"
    local baseline_file="$4"
    local tokenizer_path="$5"
    
    if check_config "$config_file"; then
        log "📊 Анализ скорости модели: $model_name"
        python3 speed_mlflow.py --config "$config_file" --model-name "$model_name" \
            --model-file "$model_file" --baseline-file "$baseline_file" \
            --tokenizer-path "$tokenizer_path"
        local speed_exit_code=$?
        
        if [ $speed_exit_code -eq 0 ]; then
            log "✅ Анализ скорости завершен успешно"
        else
            log "❌ Ошибка анализа скорости (код: ${speed_exit_code})"
        fi
        return $speed_exit_code
    else
        log "⚠️ Пропуск анализа скорости (конфигурация недоступна)"
        return 1
    fi
}


# Функция для загрузки результатов в LakeFS и MLflow (старая версия)
upload_to_services() {
    local config_file="${1:-config.yaml}"
    
    if check_config "$config_file"; then
        log "📤 Загрузка всех результатов в LakeFS и логирование в MLflow..."
        python3 upload_results.py --config "$config_file" --results-dir "${RESULTS_DIR}"
        local upload_exit_code=$?
        
        if [ $upload_exit_code -eq 0 ]; then
            log "✅ Результаты успешно загружены в LakeFS и MLflow"
        else
            log "❌ Ошибка загрузки (код: ${upload_exit_code})"
        fi
        return $upload_exit_code
    else
        log "⚠️ Пропуск загрузки (конфигурация недоступна)"
        return 1
    fi
}

# Функция для проверки доступности модели
check_model_path() {
    local path="$1"
    local name="$2"
    
    if [[ "$path" == "/your_own_path/"* ]]; then
        log "⚠️ ${name}: Путь не настроен (${path})"
        return 1
    elif [[ ! -d "$path" ]]; then
        log "❌ ${name}: Директория не найдена (${path})"
        return 1
    else
        log "✅ ${name}: Модель найдена (${path})"
        return 0
    fi
}

# Главная функция
main() {
    local config_file="${1:-config.yaml}"
    
    log "🎯 Начало выполнения Spec-Bench Evaluation"
    log "📊 Параметры:"
    log "   - Модель: ${MODEL_NAME}"
    log "   - Температура: ${TEMP}"
    log "   - Тип данных: ${torch_dtype}"
    log "   - GPU устройства: ${GPU_DEVICES}"
    log "   - Бенчмарк: ${bench_NAME}"
    log "   - Директория результатов: ${RESULTS_DIR}"
    log "   - Конфигурационный файл: ${config_file}"
    
    # Проверка конфигурации
    if check_config "$config_file"; then
        log "✅ Конфигурация загружена успешно"
    else
        log "⚠️ Проблемы с конфигурацией, продолжаем без загрузки в внешние сервисы"
    fi
    
    # Счетчики
    local total_benchmarks=0
    local successful_benchmarks=0
    local failed_benchmarks=0
    
    # Выполнение бенчмарков с пошаговым логированием
    # Baseline (Vanilla)
    if check_model_path "$Vicuna_PATH" "Baseline"; then
        process_model "Baseline (Vanilla)" \
                     "${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}" \
                     "${RESULTS_DIR}/${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}.jsonl" \
                     "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline -model-path $Vicuna_PATH -model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
                     "$config_file" \
                     "false"
    fi
    
    # SPS (Speculative Sampling)
    # if check_model_path "$Vicuna_PATH" "SPS" && check_model_path "$Drafter_PATH" "Drafter"; then
    #     process_model "SPS (Speculative Sampling)" \
    #                  "${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps -model-path $Vicuna_PATH -drafter-path $Drafter_PATH -model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # Medusa
    # if check_model_path "$Medusa_PATH" "Medusa"; then
    #     process_model "Medusa" \
    #                  "${MODEL_NAME}-medusa-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-medusa-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa -model-path $Medusa_PATH -base-model $Vicuna_PATH -model-id ${MODEL_NAME}-medusa-${torch_dtype} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # # EAGLE
    # if check_model_path "$Eagle_PATH" "EAGLE"; then
    #     process_model "EAGLE" \
    #                  "${MODEL_NAME}-eagle-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-eagle-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle -ea-model-path $Eagle_PATH -base-model-path $Vicuna_PATH -model-id ${MODEL_NAME}-eagle-${torch_dtype} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # EAGLE2
    # if check_model_path "$Eagle_PATH" "EAGLE2"; then
    #     process_model "EAGLE2" \
    #                  "${MODEL_NAME}-eagle2-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-eagle2-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 -ea-model-path $Eagle_PATH -base-model-path $Vicuna_PATH -model-id ${MODEL_NAME}-eagle2-${torch_dtype} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # # EAGLE3
    # if check_model_path "$Eagle3_PATH" "EAGLE3"; then
    #     process_model "EAGLE3" \
    #                  "${MODEL_NAME}-eagle3-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-eagle3-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 -ea-model-path $Eagle3_PATH -base-model-path $Vicuna_PATH -model-id ${MODEL_NAME}-eagle3-${torch_dtype} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # LADE (Lookahead)
    # if check_model_path "$Vicuna_PATH" "LADE"; then
    #     process_model "LADE (Lookahead)" \
    #                  "${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead -model-path $Vicuna_PATH -model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} -level 5 -window 7 -guess 7 -bench-name $bench_NAME -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # # PLD (Parallel Decoding)
    # if check_model_path "$Vicuna_PATH" "PLD"; then
    #     process_model "PLD (Parallel Decoding)" \
    #                  "${MODEL_NAME}-pld-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-pld-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld -model-path $Vicuna_PATH -model-id ${MODEL_NAME}-pld-${torch_dtype} -bench-name $bench_NAME -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # REST (Retrieval-based Speculative Decoding)
    # if check_model_path "$Vicuna_PATH" "REST" && [[ -f "$datastore_PATH" ]]; then
    #     process_model "REST (Retrieval-based Speculative Decoding)" \
    #                  "${MODEL_NAME}-rest-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-rest-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest -model-path $Vicuna_PATH -model-id ${MODEL_NAME}-rest-${torch_dtype} -datastore-path $datastore_PATH -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # # Hydra
    # if check_model_path "$Hydra_PATH" "Hydra"; then
    #     process_model "Hydra" \
    #                  "${MODEL_NAME}-hydra-${torch_dtype}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-hydra-${torch_dtype}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra -model-path $Hydra_PATH -base-model $Vicuna_PATH -model-id ${MODEL_NAME}-hydra-${torch_dtype} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # # SPACE
    # if check_model_path "$Space_PATH" "SPACE"; then
    #     process_model "SPACE" \
    #                  "${MODEL_NAME}-space-${torch_dtype}-temp-${TEMP}" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-space-${torch_dtype}-temp-${TEMP}.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_space -model-path $Space_PATH -model-id ${MODEL_NAME}-space-${torch_dtype}-temp-${TEMP} -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # Recycling
    # if check_model_path "$Vicuna_PATH" "Recycling"; then
    #     process_model "Recycling" \
    #                  "${MODEL_NAME}-recycling" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-recycling.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling -model-path $Vicuna_PATH -model-id ${MODEL_NAME}-recycling -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype" \
    #                  "$config_file"
    # fi
    
    # SAMD
    # if check_model_path "$Vicuna_PATH" "SAMD" && check_model_path "$Eagle_PATH" "SAMD tree model"; then
    #     process_model "SAMD" \
    #                  "${MODEL_NAME}-samd" \
    #                  "${RESULTS_DIR}/${MODEL_NAME}-samd.jsonl" \
    #                  "CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd -model-path $Vicuna_PATH -model-id ${MODEL_NAME}-samd -bench-name $bench_NAME -temperature $TEMP -dtype $torch_dtype -samd_n_predicts 40 -samd_len_threshold 5 -samd_len_bias 5 -tree_method eagle2 -attn_implementation sdpa -tree_model_path $Eagle_PATH" \
    #                  "$config_file"
    # fi

    # log "Выполнение завершено!"
    # log "Полный лог сохранен в ${LOG_FILE}"
}

# Обработка сигналов
cleanup() {
    log "🛑 Получен сигнал прерывания. Завершение работы..."
    exit 130
}

trap cleanup SIGINT SIGTERM

# Проверка зависимостей
check_dependencies() {
    local missing_deps=()
    
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python")
    fi
    
    if ! python3 -c "import torch" &> /dev/null; then
        missing_deps+=("torch")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "❌ Отсутствуют зависимости: ${missing_deps[*]}"
        log "Установите их перед запуском"
        exit 1
    fi
}

# Точка входа
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "🚀 Spec-Bench Evaluation Script с LakeFS интеграцией"
    echo "=================================================="
    
    # Проверить зависимости
    check_dependencies
    
    # Запустить главную функцию
    main "$@"
fi 