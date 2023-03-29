## Нейросеть для классификации текста (рукописный или печатный)
Разработана в рамках хакатона ГУАП

### Картинки с текстом
Картинки (сканы текстов) для обучения и тестирования необходимо загрузить отдельно:  
https://disk.yandex.ru/d/Yu3wUYX7xdqXHQ  
https://disk.yandex.ru/d/HSJXmD9HXelQUQ  
https://disk.yandex.ru/d/Rz1uawh-DMEhhQ  
https://disk.yandex.ru/d/nI2TnkqbSw2usA  
Распаковать в data/imgs/

### Использование нейросети
#### Обучение
- В скрипте `neural.py` установить `USE_SAVED_MODEL = False`
- По желанию поменять настройки `NUMBER_OF_IMAGES_FOR_EVALUATE`, `CHUNK_SIZE`, `TEST_CHUNK_SIZE`, `X_RESOLUTION`, `Y_RESOLUTION`
- Запустить `neural.py`. Ждать.
- После завершения обучения, модель будет сохранена в папку с проектом

#### Уже обученная модель
- В скрипте `neural.py` установить `USE_SAVED_MODEL = True`
- Значения `X_RESOLUTION` и `Y_RESOLUTION` должны совпадать с теми, что использовались при обучении модели
- По желанию поменять `TEST_CHUNK_SIZE`

После обучения модели или загрузки уже обученной, на вход нейросети будут поданы тестовые картинки. Будет создан файл `testAnswers.csv`, куда будут 
записаны результаты классификация тестов на картинках.  
Альтернативно,  можно раскомментировать строки `while 1:`. Тогда по одной будут открываться картинки, а в консоль будет выводиться результат
классификации каждой картинки

### Вспомогательные скрипты
- **`getSmallData.py`** берет из датасета некоторое количество картинок для тренировки (`TRAIN_IMAGES_COUNT`) и тестирования (`TEST_IMAGES_COUNT`).
Картинки копируются в папку `PATH_TO_SMALL_DATA`. Это удобно при тестировании разных настрок модели нейросети, чтобы загрузка и обучение на маленьком
датасете происходили быстрее