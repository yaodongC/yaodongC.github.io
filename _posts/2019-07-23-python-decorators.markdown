---
layout: post
title: "Decorators - Dynamically Alter The Functionality Of Your Functions"
subtitle:   A Beginner's Guid to Python 01
date:       2019-07-23
author:     Yaodong Cui
header-img: img/pytorch-logo-dark.svg
header-mask: 0.5
catalog: true
tags:
    - Python
---

# Related knowledge

## Python have First-class functions

**Functions in Python are first-class objects.** This allows you to define functions within function and assign attributes to functions like objects.

```python
def foo(): pass
foo.score = 20
print(type(foo))
print(foo.score)

out:
[1] <type 'function'>
[2] 20
```

A programming language is said to have First-class functions when functions in that language are treated like any other variable. For example, in such a language, a function can be **passed as an argument** to other functions, can be **returned by another function** and can be **assigned as a value** to a variable.
Functions in python:
- have types
- can be passed as an argument
- can be used in expression
- can become part of various data structures like dictionaries

## Python Variable Scope: The LEGB rule

Python check variables in the L->E->G->B order:
- Local: variables definded within a function
- Enclosing: variables in Local scope of enclosing functions
- Global: variables defined at the top-level of a module or explicitly declared using the 'global' keyword
- Built-in: variables/names that are pre-assigned in Python

An example of local and global rule:
```python
x='global x'

def test():
  x = 'local x'
  print(x)

test()
print(x)

out:
[1] 'local x'  
[2] 'global x'  
```
An example of enclosing rule:
```python
def outer():
  x = 'outer x'
  def inner():
    x = 'local x'
    print(x)
  inner()
  print(x)
  return inner

outer()
out:
[1] 'inner x'  
[2] 'outer x'  
```
## Closures:
A function defined inside another function is called a nested function. Nested functions can access variables (non-local variables) of the enclosing scope. In Python, these non-local variables are read-only by default and we must declare them explicitly as non-local (using nonlocal keyword) in order to modify them.

```python
def logger(func):
  def log_func(*args):
    logging.info('running "{}" with arguments {}'.format(func.__name__,args))
    print('running "{}" with arguments {}'.format(func.__name__,args))
    print(func(*args))
  return log_func

def add(x,y):
  return x+y

add_logger=logger(add)
add_logger(3,4)
out:
[1] running "add" with arguments 3,4
[2] 7  
```

# Python Decorators

## Use function as Decorators
Adding a decorator '@your_decorator_function' makes 'add( , )=add_logger( , )'.
```python
def logger(func):
  def log_func(*args,**kwargs):
    logging.info('running "{}" with arguments {}'.format(func.__name__,args))
    print(func(*args,**kwargs))
  return log_func

@logger
def add(x,y):
  return x+y

add(3,4)
out:
[1] running "add" with arguments 3,4
[2] 7
```

Note: *args and **kwargs are mostly used in function definitions. *args and **kwargs allow you to pass an unspecified number of arguments to a function, so when writing the function definition, you do not need to know how many arguments will be passed to your function.
*args is used to send a non-keyworded variable length argument list to the function.
```python
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')
out:
[1] first normal arg: yasoob
[2] another arg through *argv: python
[3] another arg through *argv: eggs
[4] another arg through *argv: test
```

**kwargs allows you to pass keyworded variable length of arguments to a function. You should use **kwargs if you want to handle named arguments in a function.

```python
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))

greet_me(name="yasoob",age="17",gender='X',race="lizard")
out:
[1] name = yasoob
[2] age = 17
[3] gender = X
[4] race=lizard
```

## Use class as Decorators

```python
class logger(object):
  def __init__(self,original_func):
    self.original_func=original_func
  def __call__(self,*args,**kwargs)  
    logging.info('running "{}" with arguments {}'.format(self.original_func.__name__,args))
    print(self.original_func(*args,**kwargs))
    return self.original_func

@logger
def add(x,y):
  return x+y

add(3,4)
out:
[1] running "add" with arguments 3,4
[2] 7
```
