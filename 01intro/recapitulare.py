# my_list = [8, 2, -3, 4, 2, "6", -20.74, True, None]
# print(type(my_list))
# print(len(my_list))
# print(my_list[1:4])
# print(my_list[11:12])
# print(my_list[-3])

# my_tuple = (1, 7)
# my_tuple = ('1',)
# print(type(my_tuple))

# dictionar = {"key": "value", "key2": "value2", "key3": "value3"}
# print(dictionar['key3'])
# print(dictionar.get('key3', "nu a gasit valoare"))
# print(dictionar)
# dictionar["key4"] = "value4"
# dictionar.update({"key5": "value5", "key6": "value6"})
# print(dictionar)
# for i in dictionar.keys():
# for i in dictionar.values():
# for i, v in dictionar.items():
#     print(i, v)
# print(dictionar.items())

# set_1 = set()
# my_list = [8, 2, -3, 4, 2, "6", -20.74, True, None]
# my_set = {1, 2, 3, 2}
# print(my_set[2])
# print(my_set)
# my_list = list(set(my_list))
# print(list(set(my_list)))
# print(my_list[2])

# my_var = 6
# variabila = None
# # result = "rezultat" if my_var < 6 else "alt rezultat"
# # print(result)
# result = "else"
# if my_var < 6 and (variabila := 10):
#     result = "6"
#     if variabila:
#         print('exista')
# elif my_var < 10:
#     result = "mai mic ca 10"
# print(variabila)
# else:
#     print('else')

# var = 7
# while var < 10:
#     print(var)
#     if var % 2:
#         continue
#     var += 1


# for i in range(0, 10, 2):
#     print(i)
# my_list = [8, 2, -3, 4, 2, "6", -20.74, True, None]
# for i, v in enumerate(my_list):
#     print(i, v)


def my_function(a=3, b=6, *args, **kwargs):
# def my_function(**kwargs):
    suma = a + b
    print(type(kwargs))
    for i in args:
        suma += i
    print(kwargs)
    for i in kwargs.values():
        suma += i
    return suma

suma = my_function(1, 2, 3, 4, 5, 7, 8, 10, c= 4, d= 5)
# suma = my_function(c= 4, d= 5)
print(suma)
