class Item:
    def __init__(self, count=3, max_count=16):
        self._count = count
        self._max_count = max_count
        
    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        else:
            return False
        
    # Свойство объекта. Не принимает параметров кроме self, вызывается без круглых скобок
    # Определяется с помощью декоратора property
    @property
    def count(self):
        return self._count
    
    # Ещё один способ изменить атрибут класса
    @count.setter
    def count(self, val):
        self._count = val
        if val <= self._max_count:
            self._count = val
        else:
            pass
    
    @staticmethod
    def static():
        print('I am function')
    
    @classmethod
    def my_name(cls):
        return cls.__name__
    
    def __normalize(self, num):
        if num > self._max_count:
            return self._max_count
        elif num < 0:
            return 0
        else: 
            return num
    
    def __add__(self, num):
        """ Сложение с числом """
        return self._count + num
    
    def __sub__(self, num):
        return self._count - num
    
    def __mul__(self, num):
        """ Умножение на число """
        return self._count * num
    
    def __iadd__(self, num):
        self._count = self.__normalize(self._count + num)
        return self

    def __isub__(self, num):
        self._count = self.__normalize(self._count - num)
        return self
  
    def __imul__(self, num):
        self._count = self.__normalize(self._count * num)
        return self
    
    def __eq__(self, num):
        return self._count == num
    
    def __ne__(self, num):
        return self._count != num
    
    def __gt__(self, num):
        return self._count > num
    
    def __ge__(self, num):
        return self._count >= num
    
    def __lt__(self, num):
        """ Сравнение меньше """
        return self._count < num
    
    def __le__(self, num):
        return self._count <= num
    
    def __len__(self):
        """ Получение длины объекта """
        return self._count
    
class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe

class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation
        
    @property
    def eatable(self):
        return self._saturation > 0

class Peach(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=10, color='yellow_red', saturation=8):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property 
    def color(self):
        return self._color
    
    @property
    def eatable(self):
        return super().eatable and self._ripe
    
class Pear(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=5, color='yellow', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property 
    def color(self):
        return self._color
    
    @property
    def eatable(self):
        return super().eatable and self._ripe
    
class Bread(Food):
    def __init__(self, count=1, max_count=3, color='brown', saturation=30):
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self._color = color

    @property 
    def color(self):
        return self._color
    
class Cake(Food):
    def __init__(self, count=1, max_count=6, color='black', saturation=15):
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self._color = color

    @property 
    def color(self):
        return self._color
    
class Inventory:
    def __init__(self, size=1):
        self._list = [None] * size
        self._size = size

    def __check_idx(self, idx):
        return 0 <= idx and idx <= self._size
    
    def __getitem__(self, idx):
        if not self.__check_idx(idx):
            raise IndexError(f'Index {idx} out of range')
        return self._list[idx]

    def insert(self, i, obj):
        if self.__check_idx(i):
            if obj.eatable and (not obj.count == 0):
                self._list[i] = obj
                return True
            else:
                return False
        else:
            return False
    
    def decrease(self, i, n):
        obj = self._list[i] 
        if self.__check_idx(i) and obj != None:
            obj -= n
            if obj.count == 0:
                self._list[i] = None
                return 0
            return obj.count
        else:
            return -1

def main():
    print("Test_1:")
    table = Item()
    table *= 10
    print(table.count)
    table -= 17
    print(table.count)
    table += 5
    print(table.count)
    print('')

    print("Test_2")
    bread = Bread()
    print(bread.color)
    bread += 3
    print(bread.count)
    print(bread.eatable)
    pear = Pear(ripe=True, count=4)
    print(pear.eatable)
    print('')

    print("Test_3")
    backpack = Inventory(3)
    print(backpack.insert(0, bread))
    bad_pear = Pear(ripe=0)
    print(backpack.insert(1, pear))
    print(backpack.insert(2, bad_pear))
    print(backpack.decrease(1, 2))
    print(backpack.decrease(1, 2))
    print(backpack.decrease(1, 2))
    print(backpack[0])
main()