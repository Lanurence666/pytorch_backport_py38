# Бэкпорт PyTorch для Python 3.8

[**Русский**](README_ru.md) | [中文](README_zh.md) | [English](README.md)

Бэкпорт **PyTorch 2.13.0a0** (последняя ветка main, коммит `03855a7`) для Python 3.8, обеспечивающий современные возможности PyTorch на устаревшей среде выполнения Python 3.8.

> **Примечание:** Последняя официальная версия PyTorch с поддержкой Python 3.8 — **PyTorch 2.0.x**. Этот бэкпорт предоставляет последние возможности PyTorch (torch.compile, улучшения Transformer, новые API квантования и т.д.) пользователям Python 3.8.

## Что это такое?

Это модифицированная версия исходного кода PyTorch, которая компилируется и работает на **Python 3.8** (Windows x64). Оригинальная ветка main PyTorch требует Python 3.10+, поэтому мы применили комплексный набор исправлений совместимости для работы на Python 3.8.

## Применённые исправления совместимости

Следующие проблемы синтаксиса и API Python 3.9+ были исправлены для компиляции на Python 3.8:

### Исправления исходного кода Python

| № | Проблема | Версия Python | Исправление |
|---|----------|--------------|-------------|
| 1 | Синтаксис объединения типов `X \| Y` | 3.10+ | Замена на `Union[X, Y]` через `from __future__ import annotations` или `typing.Union` |
| 2 | Встроенный синтаксис дженериков `list[X]`, `dict[X, Y]` | 3.9+ | Замена на `List[X]`, `Dict[X, Y]` из `typing` |
| 3 | `str.removeprefix()` / `str.removesuffix()` | 3.9+ | Реализация полифилла или использование альтернатив |
| 4 | `typing.TypeGuard` | 3.10+ | Замена на тип возврата `bool` |
| 5 | Использование `typing.ParamSpec` | 3.10+ | Использование `typing_extensions.ParamSpec` |
| 6 | Операторы `match` / `case` | 3.10+ | Переписывание как цепочки `if` / `elif` |
| 7 | `zip(strict=True)` | 3.10+ | Реализация ручной проверки длины |
| 8 | `functools.cache` | 3.9+ | Использование `functools.lru_cache(maxsize=None)` |
| 9 | `typing.TypeAlias` | 3.10+ | Использование простого присваивания или `typing_extensions.TypeAlias` |
| 10 | Именованные аргументы `AttributeError(msg, name=..., obj=...)` | 3.10+ | Удаление `name=None`/`obj=None` или использование вспомогательной функции `_AttributeError_compat()` |

### Исправления исходного кода C/C++

| № | Проблема | Версия Python | Исправление |
|---|----------|--------------|-------------|
| 1 | `PyType_GetSlot()` | 3.9+ | Реализация совместимой обёртки через `tp_as_number`, `tp_as_sequence`, `tp_as_mapping` |
| 2 | `_PyEval_SetProfile()` | 3.9+ | Прямое присвоение полей `PyThreadState` для 3.8 |
| 3 | `_PyInterpreterState_GetEvalFrameFunc()` / `SetEvalFrameFunc()` | 3.9+ | Заглушки no-op для 3.8 (C-shim Dynamo отключён) |
| 4 | `Py_TPFLAGS_HAVE_VECTORCALL` | 3.12+ | Маппинг на `_Py_TPFLAGS_HAVE_VECTORCALL` (3.8-3.11) |
| 5 | Несоответствие версии Flatbuffers в `mobile_bytecode_generated.h` | Н/Д | Обновлён `static_assert` для соответствия фактической версии (25.12.19) |
| 6 | Порядок включения `opcode.h` (требуется `Py_LT` и др. из `object.h`) | Н/Д | Перемещён `#include <opcode.h>` после включений `Python.h` |
| 7 | Защита `#ifndef` для каждой функции в `pythoncapi_compat.h` | Н/Д | Предотвращение ошибок переопределения при включении копий из нескольких проектов |

## Основные возможности

- **Полный набор возможностей PyTorch 2.13.0a0** на Python 3.8
- **Поддержка CUDA 12.4** (Windows x64)
- **torch.compile** (Dynamo) — функциональность на уровне Python работает; C-уровневый shim оценки кадров отключён на 3.8
- **Autograd** — полностью функционален
- **Модули нейронных сетей** — полностью функциональны
- **torch.profiler** — функционален (с совместимостью 3.8 для `_PyEval_SetProfile`)
- **Квантование** — функционально
- **Экспорт ONNX** — функционален
- **Распределённое обучение** — базовая функциональность доступна

## Результаты тестирования

По сравнению с последней официальной версией с поддержкой Python 3.8 (PyTorch 2.0.x):

| Возможность | PyTorch 2.0.x (Официальный) | PyTorch 2.13.0a0 (Этот бэкпорт) |
|-------------|---------------------------|--------------------------------|
| Тензорные операции | ✅ | ✅ |
| Autograd | ✅ | ✅ |
| Поддержка CUDA | ✅ (CUDA 11.x) | ✅ (CUDA 12.4) |
| nn.Module | ✅ | ✅ |
| Оптимизаторы | ✅ | ✅ (больше оптимизаторов) |
| torch.compile | ❌ (недоступен) | ⚠️ (только на уровне Python, C-shim отключён) |
| Модели Transformer | ✅ (базовые) | ✅ (улучшенная архитектура) |
| Квантование | ✅ (базовое) | ✅ (новые API) |
| torch.profiler | ✅ | ✅ |
| Экспорт ONNX | ✅ | ✅ |
| AMP (Смешанная точность) | ✅ | ✅ |
| torch.distributed | ✅ (базовый) | ✅ (улучшенный) |

## Файл тестов

Комплексный набор тестов включён как `test_pytorch_functions.py`. Запустите его:

```bash
python test_pytorch_functions.py
```

Тесты охватывают:
- Основные тензорные операции (создание, индексация, математика, вещание)
- Autograd (градиенты, пользовательские функции, контрольные точки градиентов)
- Модули нейронных сетей (Linear, Conv2d, LSTM, Transformer и др.)
- Функции потерь и оптимизаторы
- Сохранение/загрузка моделей
- Операции CUDA (при наличии GPU)
- Расширенные возможности (torch.compile, JIT, квантование, профилировщик, AMP)

## Как собрать

### Предварительные требования

- **Python 3.8** (64-бит, Windows)
- **Visual Studio 2022** с поддержкой C++20
- **CUDA Toolkit 12.4** (для поддержки GPU)
- **CMake** >= 3.25
- **Ninja** система сборки
- **NumPy** (совместимая с Python 3.8 версия)

### Шаги сборки

```bash
# 1. Клонируйте этот репозиторий
git clone https://github.com/Lanurence666/pytorch_backport_py38.git
cd pytorch_backport_py38

# 2. Создайте и активируйте среду conda
conda create -n py38 python=3.8
conda activate py38

# 3. Установите зависимости сборки
pip install numpy cmake ninja pybind11 typing_extensions

# 4. Установите переменные окружения
set MAX_JOBS=2
set USE_CUDA=1
set TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0

# 5. Соберите и установите (режим редактирования для разработки)
pip install -e . --no-build-isolation

# 6. Или соберите wheel-пакет
pip wheel --no-build-isolation -w dist .
```

### Важные замечания по сборке

- Установите `MAX_JOBS=2` для предотвращения ошибок памяти компоновщика (LNK1102) при линковке `torch_cpu.dll`
- Полная сборка занимает примерно 2-4 часа на современной машине
- Результирующий wheel-пакет занимает приблизительно 160 МБ

## Установка

### Из Wheel-пакета (Рекомендуется)

Скачайте wheel из [GitHub Releases](https://github.com/Lanurence666/pytorch_backport_py38/releases) и установите:

```bash
pip install torch-2.13.0a0+git03855a7-cp38-cp38-win_amd64.whl
```

### Из исходного кода

```bash
pip install -e . --no-build-isolation
```

## Известные ограничения

1. **C-shim torch.compile**: C-уровневый shim оценки кадров (`_PyInterpreterState_GetEvalFrameFunc`/`SetEvalFrameFunc`) недоступен на Python 3.8. Трассировка Dynamo на уровне Python всё ещё работает, но оптимизация производительности на C-уровне отключена.

2. **Только Windows**: Этот бэкпорт тестировался только на Windows x64 с CUDA 12.4. Сборки для Linux могут потребовать дополнительных исправлений.

3. **Окончание поддержки Python 3.8**: Python 3.8 достиг конца поддержки в октябре 2024 года. Используйте этот бэкпорт на свой страх и риск.

## Связанные проекты

Этот бэкпорт стал возможен благодаря [python38_compat_fix_suite](https://github.com/Lanurence666/python38_compat_fix_suite) — комплексному набору инструментов для портирования проектов Python 3.9+ на Python 3.8.

Другие бэкпорты для Python 3.8:
- [numpy_backport_py38](https://github.com/Lanurence666/numpy_backport_py38) — NumPy 2.x для Python 3.8
- [scipy_backport_py38](https://github.com/Lanurence666/scipy_backport_py38) — SciPy 1.x для Python 3.8

## Лицензия

PyTorch лицензирован под BSD 3-Clause License. Подробности см. в файле [LICENSE](LICENSE).

Исправления совместимости в этом бэкпорте также предоставляются под той же лицензией BSD 3-Clause.
