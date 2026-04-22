                           COUNTDOWN PUZZLE SOLVER

Задача: Обучить модель google/gemma-3-1b-it (студент) решать задачу Countdown, 
используя знания модели-учителя из другого семейства
(например, Qwen3-8B, Llama-3.1-8B-Instruct, Mistral-7B-Instruct или 
любую другую открытую модель до 8B параметров).

Пример:
  Числа: 75, 80, 90, 24
  Цель: 61
  Ответ: 90 - 80 + 75 - 24


ПОДХОД

1. Синтетические данные (1_generate.py)
   - 300k примеров, сгенерированных брутфорсом с проверкой через Fraction-арифметику
   - Сложность: 2 оператора (40%), 3 (40%), 4 (20%)
   - Числа от 1 до 99, таргеты от 1 до 999 — соответствует тестовому распределению
   - Глобальная дедупликация по паре (числа, таргет)

2. Дообучение модели (2_train.py)
   - Базовая модель: google/gemma-3-1b-it
   - Учитель для дистилляции: Qwen/Qwen3-8B (другое семейство, разные токенизаторы)
   - Метод: SFT + LoRA (r=128, alpha=128)
   - Loss только на уравнениях (completion-only), промпт маскируется
   - 150k примеров, 2 эпохи, LR=2e-4 с cosine scheduler

3. Инференс (3_inference.py)
   - Первый проход: beam search (5 лучей), температура 0.1
   - При неудаче: 15 сэмплов с температурой 0.7, majority voting по результату eval()
   - Fallback: исчерпывающий brute-force с таймаутом 5 секунд


ВОСПРОИЗВЕДЕНИЕ

Требования:
  - GPU с 10+ GB VRAM (NVIDIA, CUDA 12+)
  - Python 3.10+
  - pip install -r requirements.txt

Запуск:

  1. python 1_generate.py
     -> train_verified.jsonl
     Время: ~20 минут

  2. python 2_train.py
     -> gemma-countdown/
     Время: ~10 часов на RTX 4060 Ti 16 GB

  3. python 3_inference.py
     -> submission.csv
     Время: ~2 часа (с ретраями)


СТРУКТУРА ФАЙЛОВ

  1_generate.py        — генерация обучающих данных
  2_train.py           — дообучение модели
  3_inference.py       — решение тестовых задач
  requirements.txt     — зависимости
  train_verified.jsonl — обучающий датасет (генерируется)
  test_public.csv      — тестовый набор (2000 задач)
  submission.csv       — итоговый файл для сдачи
  gemma-countdown/     — обученный LoRA-адаптер


РЕЗУЛЬТАТЫ

Публичный тест (600 задач):

  Модель (first pass + retries): ~47%
  Brute-force fallback: остальные задачи
  Итоговая точность: ~99.7%