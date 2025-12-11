import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {} # словарь, используемый для преобразования меток в числа
        self.num2label = {} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        self.label2num = {"ham": 0, "spam": 1}
        self.num2label = {0: "ham", 1: "spam"}
        
        cleaned_messages = []
        for message in self._x:
            cleaned = re.sub(r'[^a-zA-Z\s]', '', str(message).lower())
            cleaned = ' '.join(cleaned.split())
            cleaned_messages.append(cleaned)
        
        self._x = np.array(cleaned_messages)
        
        self._y = np.array([self.label2num[label] for label in self._y])
        # Конец вашего кода

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        # Начало вашего кода
        indices = np.arange(len(self._x))
        np.random.shuffle(indices)
        
        X_shuffled = self._x[indices]
        y_shuffled = self._y[indices]
        
        n = len(self._x)
        test_size = int(n * test)
        val_size = int(n * val)
        train_size = n - test_size - val_size
        
        X_train = X_shuffled[:train_size]
        y_train = y_shuffled[:train_size]
        
        X_val = X_shuffled[train_size:train_size + val_size]
        y_val = y_shuffled[train_size:train_size + val_size]
        
        X_test = X_shuffled[train_size + val_size:]
        y_test = y_shuffled[train_size + val_size:]
        
        self.train = (X_train, y_train)
        self.val = (X_val, y_val)
        self.test = (X_test, y_test)
        # Конец вашего кода
