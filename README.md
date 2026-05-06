# ML Project — Предсказание типа события в гамма-телескопе

**Студент:** Меджидова Камила Гусейновна

**Группа:** БИВ237

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

Проект посвящён задаче бинарной классификации событий, зарегистрированных гамма-телескопом MAGIC.
По физическим характеристикам события необходимо определить, относится ли оно к gamma-сигналу или к hadron-шуму.

**Задача:** Классификация

**Датасет:** MAGIC Gamma Telescope Dataset (UCI Machine Learning Repository)

**Целевая метрика:** ROC-AUC (основная), дополнительно Accuracy, F1-score

## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы (magic04.data, magic04.names)
├── models                      # Сохранённые модели 
├── notebooks
│   └── cp1_magic_baseline.ipynb  # EDA + baseline + модели
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   └── data_preprocessing.py   # Предобработка данных
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# Установка зависимостей
pip install -r requirements.txt
```

Проект выполнялся в Jupyter Notebook / Google Colab.

## Данные

* `data/raw/` — исходные файлы (без изменений)
* `data/processed/` — обработанные данные (будут добавлены позже)


## Результаты

| Модель              | Accuracy | F1-score | ROC-AUC | Примечание    |
| ------------------- | -------- | -------- | ------- | ------------- |
| Logistic Regression | 0.79     | 0.84     | 0.84    | baseline      |
| KNN                 | 0.83     | 0.87     | 0.87    |               |
| Decision Tree       | 0.80     | 0.85     | 0.79    |               |
| Random Forest       | 0.88     | 0.91     | 0.93    | лучшая модель |


## Отчёт

---

## CP2 Experiments

Additional machine learning experiments were conducted using advanced ensemble methods and hyperparameter tuning.

### Tested models

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees
- Support Vector Machine (SVM)
- XGBoost

### Hyperparameter tuning

GridSearchCV was applied to Random Forest for parameter optimization.

Best parameters:

```python
{'max_depth': None, 'n_estimators': 200}
```

### Final Results

| Model | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Extra Trees | 0.8883 | 0.8276 | 0.9410 |
| Random Forest tuned | 0.8883 | 0.8306 | 0.9409 |
| XGBoost | 0.8885 | 0.8331 | 0.9406 |
| Random Forest | 0.8877 | 0.8304 | 0.9404 |
| Gradient Boosting | 0.8767 | 0.8082 | 0.9274 |
| SVM | 0.8289 | 0.7098 | 0.8698 |
| Logistic Regression | 0.7981 | 0.6792 | 0.8446 |

The best result was achieved by the Extra Trees model.
Финальный отчёт будет находиться в `report/report.md`.
