# План Выполнения

Этот документ фиксирует рабочий план доработки проекта от текущего MVP-состояния до полного прогона на реальном датасете, сохранения паттернов и sequence dataset в файлы, обучения модели и визуализации результатов.

## Общий прогресс

`92%`

Правило обновления:
- после каждого содержательного шага обновляется процент общего прогресса;
- у каждого пункта должен быть статус: `done`, `in_progress`, `pending`, `blocked`;
- этот документ считается основным трекером выполнения плана.

## 1. Архитектурная фиксация границ stage

Статус: `done`

Цель:
- явно разделить data stage, discovery stage, sequence dataset stage, model stage, decoding stage и visualization stage;
- зафиксировать, какие из них должны выполняться только на CUDA.

Подзадачи:
- описать stage boundaries в коде и документации;
- убрать двусмысленность между preprocessing и model/discovery stage;
- привести CLI к этим границам.

Результат:
- в постановке проекта явно зафиксировано, что discovery относится к GPU-only stage;
- план и прогресс вынесены в отдельный трекер;
- workflow разделён на artifact stage, sequence dataset stage, model stage и visualization stage.

Уточнение после пересмотра workflow:
- `run-pipeline` в старом виде больше не соответствует требованию, если discovery должен быть строго GPU-only;
- workflow нужно разделить на:
  - `prepare-pattern-windows` (CPU-only)
  - `discover-patterns` (GPU-only)
  - `train/evaluate/predict` (GPU-only)

## 2. Перенос discovery в GPU-only stage

Статус: `done`

Цель:
- перестать считать `pattern discovery` CPU-only baseline для финального workflow;
- реализовать CUDA-совместимый discovery backend;
- падать с явной ошибкой, если discovery stage запрошен без CUDA.

Подзадачи:
- определить backend для GPU discovery;
- переписать структурный discovery слой под CUDA;
- подключить runtime validation для discovery stage.

Результат на текущий момент:
- `StructuralPatternDiscovery` переведён на `torch` backend с обязательным `cuda`;
- runtime validation для discovery stage подключена;
- короткий discovery run на CUDA успешно проходит;
- discovery вынесен в отдельную CLI-команду `discover-patterns`;
- подготовка окон вынесена в отдельную CLI-команду `prepare-pattern-windows`.

Осталось:
- подтвердить стабильность полного 5+ лет прогона;
- при необходимости доработать производительность и memory profile.

Наблюдение по full run:
- `prepare-pattern-windows` на полном датасете завершился примерно за `10.5` минут;
- downstream `forecast_sequence_dataset.jsonl` на полном датасете оказался очень тяжёлым по размеру, что требует учитывать стоимость записи/чтения и memory profile.

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
- для тяжёлых full-run bundle добавлена совместимая запись/чтение в gzip-формате `forecast_sequence_dataset.jsonl.gz`.

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

Статус: `done`

Цель:
- получить полный pattern corpus на всём `hly4935_subset.csv`;
- зафиксировать количество паттернов, размер pattern flow и полный sequence dataset.

Подзадачи:
- прогнать full discovery;
- прогнать full training;
- прогнать full evaluation;
- зафиксировать итоговые артефакты полного run.
- исключить повторный full `prepare` при перезапуске полного workflow, если `artifacts/prepare/*` уже существуют и валидны.

Результат:
- полный 5+ лет run завершён на артефактах в `/mnt/ml/weather_forecast_artifacts`;
- получены и сохранены `prepare`, `discovery`, `training`, `prediction` и `evaluation` артефакты;
- historical OOM на этапе `evaluate` воспроизведён, локализован и устранён;
- финальный full `evaluate` успешно завершён в streaming/lightweight режиме без повторного memory blow-up.

## 8. Проверка качества найденных паттернов

Статус: `pending`

Цель:
- убедиться, что найденные паттерны устойчивы и содержательно полезны.

Подзадачи:
- проверить распределение размеров паттернов;
- выявить вырожденные и слишком редкие паттерны;
- проверить интерпретируемость паттернов по реальным погодным окнам.

## 9. Приведение model pipeline к воспроизводимому workflow

Статус: `done`

Цель:
- сделать `train`, `predict`, `evaluate` более воспроизводимыми и удобными для полного run.

Результат на текущий момент:
- добавлены CLI summary/output улучшения;
- добавлены JSON summary outputs для model команд.
- `train-sequence-model` умеет принимать сохранённый `forecast_sequence_dataset.jsonl`.
- основная CLI-документация переведена на workflow `prepare -> discover -> train`.
- `predict-sequence` умеет работать от сохранённых `prepared_pattern_windows.jsonl` и `pattern_prototypes.jsonl`;
- `evaluate-sequence-model` умеет работать от сохранённых `prepared_pattern_windows.jsonl`, `pattern_prototypes.jsonl` и `forecast_sequence_dataset.jsonl`.
- `run-split-workflow` получил флаг `--reuse-prepare` для повторного использования готового `prepare` bundle без пересчёта CPU-стадии.

Осталось:
- расширить загрузку сохранённых discovery-артефактов для остальных downstream-сценариев;
- дополнительно сократить operational cost записи/чтения полного `forecast_sequence_dataset.jsonl.gz`, либо заменить формат на более компактный бинарный;
- вычистить legacy-path, чтобы он не был основным operational workflow.

Уточнение по итогу полного прогона:
- `evaluate-sequence-model` переведён на streaming evaluation от сохранённого dataset;
- full evaluation использует lightweight загрузку артефактов без materialization `forecast_samples` и полного списка `PatternWindow`;
- в workflow добавлены process boundaries для `predict` и `evaluate`, memory/timing logs и RSS guardrail.

## 10. Полная GPU-only оценка модели

Статус: `done`

Цель:
- получить валидные метрики модели только на CUDA run.

Подзадачи:
- запустить full train/evaluate на полном датасете;
- собрать поканальные и агрегированные метрики;
- сравнить с baseline.

Результат:
- full GPU-backed evaluation завершён;
- итоговый отчёт сохранён в `artifacts/model/sequence_evaluation.json`;
- по агрегированному среднему RMSE модель лучше baseline на validation и test, но по агрегированному среднему MAE baseline пока лучше;
- memory profile full evaluation стабилизирован на уровне около `1.15 GB RSS` вместо прежнего OOM-сценария.

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
  - `forecast_sequence_matrix.png`
  - `weather_pattern_overlay.png`
- сохранение файлов подтверждено в целевом runtime вне sandbox.

## 13. Единый reproducible run

Статус: `done`

Цель:
- один сценарий должен давать полный набор артефактов:
  - паттерны
  - pattern flow
  - sequence dataset
  - обученную модель
  - evaluation report
  - prediction outputs
  - visualization outputs

Результат на текущий момент:
- добавлен staged orchestrator `run-split-workflow`;
- короткий run подтверждён на схеме `prepare -> discover -> train -> predict -> visualization`;
- stage outputs пишутся раздельно в `prepare/`, `discovery/`, `model/`, `plots/`.
- добавлен `--reuse-prepare` для повторных запусков после успешного `prepare`.

Осталось:
- подтвердить полный 5+ лет прогон в этом режиме.

Результат:
- staged orchestrator доведён до полного 5+ лет прогона;
- первый end-to-end run завершён с выпуском всех основных model-артефактов;
- повторный запуск может переиспользовать `prepare` bundle и downstream saved artifacts без повторного полного CPU stage.

## 13A. Стабилизация Хоста И Storage Hygiene

Статус: `in_progress`

Цель:
- исключить повторные зависания хоста во время полного run;
- убрать лишнюю нагрузку с системного диска и Docker storage;
- вынести тяжёлые runtime-артефакты на ёмкий диск.

Наблюдения:
- полный `forecast_sequence_dataset.jsonl` уже достиг размера порядка `9.9 GB`;
- `prepared_pattern_windows.jsonl` достиг размера порядка `2.6 GB`;
- суммарный локальный каталог `artifacts/` уже достиг порядка `12 GB`;
- в Docker есть большой объём reclaimable data, который не нужен для текущего workflow.

План действий:
- очистить неиспользуемые Docker images, stopped containers и build cache;
- перенести `artifacts/` на `/mnt/ml`, чтобы полный run не нагружал системный диск;
- при необходимости затем вынести `.venv` и/или Docker data-root на отдельный диск;
- отдельно сократить размер `forecast_sequence_dataset.jsonl` или заменить формат на более компактный.

Критерий завершения:
- `artifacts/` физически размещены на `/mnt/ml` и доступны из проекта прозрачно;
- reclaimable Docker storage очищен;
- повторный full/staged run не упирается в локальный storage pressure как в первичный риск.

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
