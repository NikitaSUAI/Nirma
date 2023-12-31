# NIRMA

Данный проект производит обработку речи двух и более дикторов:
* производит транскрибацию текста
* производит диаризацию диткоров
* определяет эммоции
* определяет уровень доминантности и валинтности дикторов

В качестве входных параметров используется аудиозапись в формате wav. Желательно без применения кодирования сигнала.

Результатом работы данного репозитория - является файл в формате json. Данный файл содержит транскрипцию речи, метки дикторов для каждого слова и предложения, класс эммоций и уровень доминантности и валинтности диктора. Каздая запись в даннм файле имеет метку начала и конца произнесения.

# Установка:

## Установка через Poetry:
* [Установить git](https://git-scm.com/book/ru/v2/%D0%92%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5-%D0%A3%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-Git) 
* [Установить Poetry](https://python-poetry.org/docs/)
* Склонировать репозиторий:
    ~~~bash
    ~$ git clone https://github.com/NikitaSUAI/Nirma.git
    ~~~
* Перейти в папку с проектом и установить зависимости. 
    ~~~bash
    ~$ cd nirma
    ~$ poetry install
    ~~~
* Необходимо получить токен для скачивания моделей с [hugging-face](https://huggingface.co/settings/tokens) и добавить его в конфиг src/pipelines/pipeline_cofigs/test_pipeline.yaml:acces_token
* Установка завершена!

# Запуск пайплайна

Для запуска пайлпайна из командной строки необходимо воспользоваться командой:
~~~bash
~$ poetry run pipeline \
    --wav_file <path to wav> \
    --output <path to output file>\ # default = "test.json"
    --conf <path to config> # default = "src/pipelines/pipeline_cofigs/test_pipeline.yaml"
~~~

Помимо возможности запуска из командной строки, доступен бот в [telegram](https://t.me/testing_digital_bot).
# Структура репозитория

```
├── README.md               <- Главный readme
├── env                     <- Раздер настройки окружения для работы с проектом
│   ├── requirements.txt    <- Файл с зависимостями окружения
│   ├── Dockerfile          <- Файл для сборки докер контейнера
|   ├── README.md           <- Readme по настройке окружения
├── cache                   <- Кэш для файлов моделей
├── src                     <- Исходниые файлы проекта
│   ├── services            <- дополнительные утилиты библиотеки (telegram бот)
│   ├── base                <- Базовые интерфейсы и классы
│   ├── utils               <- Утилы библиотеки
│   ├── tasks               <- Задачи, исполняемые пайплайном (распознавание речи, диаризация, оценка эмоциональности, аугментации и т.д.)
│      ├── task_configs     <- Конфигурационные файлы задач
│   ├── pipelines           <- Выстроенные из задач пайплайны, для получения сложного конечного результата
│      ├── pipline_configs  <- Конфигурационные файлы пайлпайнов
```
# Философия репозитория
- Разнородные модели "живут" каждая в своих task
- Общий код делится между разными моделями
- Задачи имеют универсальный интерфейс
- Из задач можно строить общий пайплайн
- Основной фреймворк HuggingFace
- Мультимодальный вход задачи сводится к текстовому описанию
- Текстовое описание prompt - программированием задачет задачу в LLM


## CheckList 1.0.0
- [x] Упрощение структуры репозитория
- [x] Добавить универсальный интерфейс исполнения задач
- [x] Добавить констурктор пайплайнов из задач
- [x] Добавить задачу распознавания речи
- [x] Добавить задачу диаризации дикторов
- [x] Добавить задачу оценки эмоций по аудио
- [ ] Добавить задачу шаблон LLM
- [x] Завести репо моделей
- [x] Написать документацию к библиотеке
- [x] Свести общий requirement
- [x] Телеграм бот для демонтрации как сервис
- [ ] Добавить задачу аугментации текста (опционально)
- [ ] Добавить задачу аугментации аудио (опционально)




