# План Выполнения

Этот документ фиксирует рабочий план доработки проекта от текущего MVP-состояния до полного прогона на реальном датасете, сохранения паттернов и sequence dataset в файлы, обучения модели и визуализации результатов.

## Общий прогресс

`27%`

Правило обновления:
- после каждого содержательного шага обновляется процент общего прогресса;
- у каждого пункта должен быть статус: `done`, `in_progress`, `pending`, `blocked`;
- этот документ считается основным трекером выполнения плана.

## 1. Архитектурная фиксация границ stage

Статус: `in_progress`

Цель:
- явно разделить data stage, discovery stage, sequence dataset stage, model stage, decoding stage и visualization stage;
- зафиксировать, какие из них должны выполняться только на CUDA.

Подзадачи:
- описать stage boundaries в коде и документации;
- убрать двусмысленность между preprocessing и model/discovery stage;
- привести CLI к этим границам.

## 2. Перенос discovery в GPU-only stage

Статус: `pending`

Цель:
- перестать считать `pattern discovery` CPU-only baseline для финального workflow;
- реализовать CUDA-совместимый discovery backend;
- падать с явной ошибкой, если discovery stage запрошен без CUDA.

Подзадачи:
- определить backend для GPU discovery;
- переписать структурный discovery слой под CUDA;
- подключить runtime validation для discovery stage.

## 3. Материализация паттернов на диск

Статус: `done`

Цель:
- хранить найденные паттерны не только в памяти, но и в явном артефакте на диске.

Результат:
- добавлен экспорт `pattern_prototypes.jsonl`.

## 4. Материализация pattern flow на диск

Статус: `done`

Цель:
- хранить временную последовательность найденных паттернов на окнах.

Результат:
- добавлен экспорт `pattern_flow.jsonl`.

## 5. Материализация sequence dataset на диск

Статус: `done`

Цель:
- хранить sequence dataset, на котором затем учится модель.

Результат:
- добавлен экспорт `forecast_sequence_dataset.jsonl`.

## 6. Слой export/import для артефактов

Статус: `done`

Цель:
- сделать работу с артефактами воспроизводимой;
- дать возможность отдельно переиспользовать найденные паттерны и sequence dataset без полного пересчёта.

Результат:
- сериализация `.jsonl` вынесена в отдельный модуль;
- добавлено обратное чтение:
  - `pattern_prototypes.jsonl`
  - `pattern_flow.jsonl`
  - `forecast_sequence_dataset.jsonl`

Следующий шаг:
- подключить использование этих артефактов в training/discovery workflow, а не только их запись и чтение.

## 7. Полный прогон на всём 5+ лет датасете

Статус: `pending`

Цель:
- получить полный pattern corpus на всём `hly4935_subset.csv`;
- зафиксировать количество паттернов, размер pattern flow и полный sequence dataset.

Подзадачи:
- прогнать full discovery;
- прогнать full training;
- прогнать full evaluation;
- зафиксировать итоговые артефакты полного run.

## 8. Проверка качества найденных паттернов

Статус: `pending`

Цель:
- убедиться, что найденные паттерны устойчивы и содержательно полезны.

Подзадачи:
- проверить распределение размеров паттернов;
- выявить вырожденные и слишком редкие паттерны;
- проверить интерпретируемость паттернов по реальным погодным окнам.

## 9. Приведение model pipeline к воспроизводимому workflow

Статус: `in_progress`

Цель:
- сделать `train`, `predict`, `evaluate` более воспроизводимыми и удобными для полного run.

Результат на текущий момент:
- добавлены CLI summary/output улучшения;
- добавлены JSON summary outputs для model команд.
- `train-sequence-model` умеет принимать сохранённый `forecast_sequence_dataset.jsonl`.

Осталось:
- отвязать обучение от обязательного пересчёта всех артефактов при каждом запуске;
- переключить training workflow на сохранённый pattern dataset.

## 10. Полная GPU-only оценка модели

Статус: `pending`

Цель:
- получить валидные метрики модели только на CUDA run.

Подзадачи:
- запустить full train/evaluate на полном датасете;
- собрать поканальные и агрегированные метрики;
- сравнить с baseline.

## 11. Доведение decoding до полноценного прогноза

Статус: `pending`

Цель:
- возвращать из предсказанных паттернов нормальный прогноз weather channels, time placeholders и hazards.

Подзадачи:
- проверить reconstruction values;
- проверить reconstruction time placeholders;
- проверить reconstruction hazard layer;
- валидировать физическую правдоподобность результата.

## 12. Визуализация

Статус: `in_progress`

Цель:
- добавить графическое представление паттернов, pattern flow и sequence dataset.

Подзадачи:
- timeline of `pattern_id` over time;
- raw weather + pattern overlay;
- prototype heatmaps;
- history-to-target sequence matrix;
- сохранение графиков в файлы.

Текущее состояние:
- добавлен CLI `plot-patterns`;
- реализованы:
  - `pattern_flow_timeline.png`
  - `pattern_prototypes_heatmap.png`
  - `weather_pattern_overlay.png`
- требуется дополнительная верификация сохранения файлов в целевом runtime без ограничений текущего sandbox.

## 13. Единый reproducible run

Статус: `pending`

Цель:
- один сценарий должен давать полный набор артефактов:
  - паттерны
  - pattern flow
  - sequence dataset
  - обученную модель
  - evaluation report
  - prediction outputs
  - visualization outputs

## 14. Критерий завершения

Статус: `pending`

План считается выполненным, когда:
- pattern discovery выполняется в корректном GPU workflow;
- найденные паттерны и pattern flow сохраняются в `.jsonl`;
- sequence dataset сохраняется в `.jsonl`;
- полный прогон на всём датасете завершён;
- модель обучена и оценена на полном датасете;
- декодирование выдаёт обычный погодный прогноз;
- есть визуализация найденных паттернов и последовательностей;
- прогресс в этом документе обновлён до `100%`.
