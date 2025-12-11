import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None # словарь, используемый для преобразования меток в числа
        self.num2label = None # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        self._train_X, self._train_y = dataset.train
        self._val_X, self._val_y = dataset.val
        self._test_X, self._test_y = dataset.test
        
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        
        for i, message in enumerate(self._train_X):
            words = message.split()
            label = self._train_y[i]
            
            for word in words:
                self.vocab.add(word)
                
                if label == self.label2num["spam"]:
                    self.spam[word] = self.spam.get(word, 0) + 1
                else:
                    self.ham[word] = self.ham.get(word, 0) + 1
        
        self.Nvoc = len(self.vocab)
        self.Nspam = sum(self.spam.values())
        self.Nham = sum(self.ham.values())
        # Конец вашего кода
    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        import re
        cleaned = re.sub(r'[^a-zA-Z\s]', '', str(message).lower())
        cleaned = ' '.join(cleaned.split())
        words = cleaned.split()
        
        total_messages = len(self._train_y)
        p_spam_prior = np.sum(self._train_y == self.label2num["spam"]) / total_messages
        p_ham_prior = np.sum(self._train_y == self.label2num["ham"]) / total_messages
        
        log_pspam = np.log(p_spam_prior)
        log_pham = np.log(p_ham_prior)
        
        for word in words:
            spam_count = self.spam.get(word, 0)
            p_word_spam = (spam_count + self.alpha) / (self.Nspam + self.alpha * self.Nvoc)
            
            ham_count = self.ham.get(word, 0)
            p_word_ham = (ham_count + self.alpha) / (self.Nham + self.alpha * self.Nvoc)
            
            log_pspam += np.log(p_word_spam)
            log_pham += np.log(p_word_ham)
        
        pspam = log_pspam
        pham = log_pham
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        correct = 0
        total = len(self._val_y)
        
        for i, message in enumerate(self._val_X):
            prediction = self.inference(message)
            true_label = self.num2label[self._val_y[i]]
            
            if prediction == true_label:
                correct += 1
        
        val_acc = correct / total
        # Конец вашего кода
        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        correct = 0
        total = len(self._test_y)
        
        for i, message in enumerate(self._test_X):
            prediction = self.inference(message)
            true_label = self.num2label[self._test_y[i]]
            
            if prediction == true_label:
                correct += 1
        
        test_acc = correct / total
        # Конец вашего кода
        return test_acc


