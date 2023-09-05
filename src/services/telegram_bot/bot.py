"""Бот по мотивам  https://github.com/tochilkinva/tg_bot_stt_tts/tree/main"""

import logging
import os
from pathlib import Path
import sys

from src.pipelines.test_pipeline import TestPipeline

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types.input_file import InputFile
from dotenv import load_dotenv
import librosa



load_dotenv()

TELEGRAM_TOKEN = "383402153:AAFwMryOo8W_mI56z6BRFRnwFWYjaZzNpmo"
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)  # Объект бота
dp = Dispatcher(bot)  # Диспетчер для бота


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

cfg_path = "src/pipelines/pipeline_cofigs/test_pipeline.yaml"

# Хэндлер на команду /start , /help
@dp.message_handler(commands=["start", "help"])
async def cmd_start(message: types.Message):
    await message.reply(
        """Это сервис для демонстрации функционала билиотеки в рамкапх НИРМА
        Библиотеки реализует алгоритмы распознавание валентности и доминантности дикторов в полилогах
        Для работы загрузите аудиофайл с полилогом
        """
    )


# Хэндлер на команду /test
@dp.message_handler(commands="base_pipeline")
async def cmd_test(message: types.Message):
    """
    Обработчик команды /test
    """
    ms_text = "Пайплайн режима успешно инициализирован"
    ms_text = "Пайплайн режима недоступен"
    await message.answer(f"{ms_text}")


# Хэндлер на получение текста
@dp.message_handler(content_types=[types.ContentType.TEXT])
async def cmd_text(message: types.Message):
    """
    Обработчик на получение текста
    """
    await message.reply("Текст получен, спасибо за текст")


# Хэндлер на получение голосового и аудио сообщения
@dp.message_handler(content_types=[
    types.ContentType.VOICE,
    types.ContentType.AUDIO,
    types.ContentType.DOCUMENT
    ]
)
async def voice_message_handler(message: types.Message):
    """
    Обработчик на получение голосового и аудио сообщения.
    """
    if message.content_type == types.ContentType.VOICE:
        file_id = message.voice.file_id
    elif message.content_type == types.ContentType.AUDIO:
        file_id = message.audio.file_id
    elif message.content_type == types.ContentType.DOCUMENT:
        file_id = message.document.file_id
    else:
        await message.reply("Формат документа не поддерживается")
        return

    pipeline = TestPipeline.init_from_config(cfg_path)
    file = await bot.get_file(file_id)
    file_path = file.file_path
    file_on_disk = Path("", f"{file_id}.tmp")
    await bot.download_file(file_path, destination=file_on_disk)
    await message.reply("Аудио получено")

    text = ""
    try:
        audio, sr = librosa.load(file_on_disk, sr=16000)
        res = pipeline.process({"audio": audio})
        for idx, (emo, vd) in enumerate(zip(res["segments_emotions"], res["segments_valence_dominance"])):
            segment = res["segments"][idx]
            text += f'{segment["speaker"]}: {segment["text"]} (эмоция: {emo} валентность: {str(vd[0])} доминантность:{str(vd[1])})'
            text += '\n'
    except Exception as e:
        del pipeline

    max_answer_len = 2000

    if len(text) == 0:
        text = "Извините, произошла ошибка при обработке документа."

    cur_text = ''
    if len(text) > max_answer_len:
        for idx in range(len(text) // max_answer_len):
            cur_text = text[idx * max_answer_len: (idx+1) * max_answer_len]
        await message.answer(cur_text)

    os.remove(file_on_disk)  # Удаление временного файла


if __name__ == "__main__":
    # Запуск бота
    logging.info("Запуск сервиса")
    try:
        executor.start_polling(dp, skip_updates=True)
    except (KeyboardInterrupt, SystemExit):
        pass
